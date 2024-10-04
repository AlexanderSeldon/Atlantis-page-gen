import streamlit as st
import time
import os
import numpy as np
from twelvelabs import TwelveLabs
from dotenv import load_dotenv
import base64
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from twelvelabs import APIStatusError, AuthenticationError
from twelvelabs.models.embed import EmbeddingsTask
from openai import OpenAI
from googleapiclient.discovery import build
import re
import difflib


# Load environment variables
load_dotenv("Atlantis.env")

# API Keys
TWELVE_LABS_API_KEY = os.getenv('TWELVE_LABS_API_KEY')

# Initialize Twelve Labs client
client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

google_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)


# Streamlit session state initialization
if 'video_index_id' not in st.session_state:
    st.session_state['video_index_id'] = None
if 'video_id' not in st.session_state:
    st.session_state['video_id'] = None
if 'video_file_path' not in st.session_state:
    st.session_state['video_file_path'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def create_index():
    try:
        engines = [
            {
                "name": "marengo2.6",
                "options": ["visual", "conversation", "text_in_video", "logo"]
            },
            {
                "name": "pegasus1.1",
                "options": ["visual", "conversation"]
            }
        ]
        video_index = client.index.create(
            name=f"video_index_{int(time.time())}",
            engines=engines
        )
        st.session_state['video_index_id'] = video_index.id
        st.success(f"Created index: id={video_index.id}, name={video_index.name}")
    except Exception as e:
        st.error(f"Failed to create index: {e}")

def upload_video(index_id, video_bytes):
    try:
        # Save the uploaded video to a temporary file
        video_file_path = "uploaded_video.mp4"
        with open(video_file_path, "wb") as f:
            f.write(video_bytes)
        st.session_state['video_file_path'] = video_file_path

        # Upload video file to Twelve Labs
        task = client.task.create(
            index_id=index_id,
            file=video_file_path  # Provide the path to the video file
        )
        st.success(f"Created task: id={task.id}")

        # Wait for indexing to complete
        def on_task_update(task):
            st.text(f"Upload Status: {task.status}")

        task.wait_for_done(sleep_interval=5, callback=on_task_update)

        if task.status == 'ready':
            st.success(f"Uploaded video. The unique identifier of your video is {task.video_id}")
            st.session_state['video_id'] = task.video_id  # Set the video_id in session state
            return task.video_id
        else:
            st.error(f"Indexing failed with status {task.status}")
            return None
    except Exception as e:
        st.error(f"Failed to upload video: {str(e)}")
        return None


def generate_gist(video_id, types=["title", "topic", "hashtag"]):
    try:
        res = client.generate.gist(video_id, types=types)
        return f"Title = {res.title}\nTopics = {res.topics}\nHashtags = {res.hashtags}"
    except Exception as e:
        st.error(f"Error generating gist: {str(e)}")
        return ""



def chat_with_bot(user_input):
    try:
        messages = st.session_state['chat_history']
        messages.append({"role": "user", "content": user_input})
        is_first_answer = len(messages) == 1

        if 'video_embeddings' in st.session_state:
            search_results = search_video(
                client,
                st.session_state['video_index_id'],
                user_input,
                st.session_state['video_embeddings']
            )
        else:
            st.error("Video embeddings not found. Please ensure the video is properly processed.")
            return "I'm sorry, but I couldn't find the video embeddings. Please make sure the video has been properly processed."

        if search_results:
            # Sort search results by start time to ensure chronological order
            search_results.sort(key=lambda x: x['start'])
            
            # Create context from search results in chronological order
            context = ""
            for result in search_results[:5]:
                context += f"From {result['start']} to {result['end']} seconds (score: {result['score']:.2f})\n"

            # Use the context to generate the open-ended text
            prompt = f"""Based on the following video segments:

{context}

{'Identify the game being played and the specific level or guide shown in this video, then ' if is_first_answer else ''}
Respond to this user message in detail (about 200-250 words):
{user_input}

Include specific references to the video content where relevant, and provide a comprehensive answer that covers the main points while still being concise."""

            twelve_labs_response = generate_open_ended_text(
                video_id=st.session_state['video_id'],
                prompt=prompt,
                is_first_answer=is_first_answer
            )

            if is_first_answer:
                # Split the response into lines
                response_lines = twelve_labs_response.split('\n')
                # Extract game and level info
                game_info = '\n'.join(response_lines[:2])
                # The actual answer starts from the third line
                answer = '\n'.join(response_lines[2:])
            else:
                game_info = ""
                answer = twelve_labs_response

            enhanced_response = chatgpt_rag_analysis(user_input, answer, game_info)
        else:
            enhanced_response = "I apologize, but I couldn't find any relevant content in the video to answer your question. Could you please rephrase your query or ask about a different aspect of the video?"

        messages.append({"role": "assistant", "content": enhanced_response})
        st.session_state['chat_history'] = messages
        return enhanced_response
    except Exception as e:
        st.error(f"Error during chatbot conversation: {str(e)}")
        return "I'm sorry, but an error occurred while processing your request. Please try again or rephrase your question."

def create_video_embeddings(video_file_path, relevant_clips):
    try:
        task = client.embed.task.create(
            engine_name="Marengo-retrieval-2.6",
            video_file=video_file_path,
            video_embedding_scopes=["clip", "video"]  # Create embeddings for clips and the entire video
        )
        st.success(f"Created embedding task: id={task.id}")

        def on_task_update(task):
            st.text(f"Embedding Status: {task.status}")

        task.wait_for_done(sleep_interval=2, callback=on_task_update)

        if task.status == 'ready':
            st.success("Video embedding task completed successfully")
            return task.id
        else:
            st.error(f"Embedding task failed with status {task.status}")
            return None
    except Exception as e:
        st.error(f"An error occurred while creating video embeddings: {str(e)}")
        return None

def retrieve_embeddings(task_id):
    try:
        task = client.embed.task.retrieve(task_id)
        if task.video_embeddings is not None:
            st.success("Retrieved video embeddings successfully")
            return task.video_embeddings
        else:
            st.warning("No embeddings found for the task")
            return None
    except Exception as e:
        st.error(f"Failed to retrieve embeddings: {str(e)}")
        return None



def google_search(query):
    try:
        res = google_service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=5).execute()
        return res.get('items', [])
    except Exception as e:
        st.error(f"Error performing Google search: {str(e)}")
        return []

def get_embeddings(texts):
    try:
        response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = [e.embedding for e in response.data]
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return []

def retrieve_relevant_documents(query, texts):
    query_embedding = get_embeddings([query])[0]
    doc_embeddings = get_embeddings(texts)
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:3]  # Get top 3 documents
    top_texts = [texts[i] for i in top_indices]
    return top_texts

def perform_rag_search(query):
    search_results = google_search(query)
    texts = [f"{item['title']}\n{item['snippet']}" for item in search_results]
    top_texts = retrieve_relevant_documents(query, texts)
    
    # Format results with sources
    formatted_results = "RAG Search Results:\n"
    for i, text in enumerate(top_texts, 1):
        original_item = next(item for item in search_results if f"{item['title']}\n{item['snippet']}" == text)
        formatted_results += f"{i}. {text}\nSource: [{original_item['title']}]({original_item['link']})\n\n"
    
    context = "\n\n".join(top_texts)
    prompt = f"""
    Use the following information to answer the question:
    
    Context: {context}
    
    Question: {query}
    
    Answer:
    """
    answer = generate_openai_text(prompt)
    
    # Format answer with sources
    formatted_answer = answer + "\n\nSources:\n"
    for item in search_results:
        formatted_answer += f"- [{item['title']}]({item['link']})\n"
    
    return formatted_answer, formatted_results


def chatgpt_rag_analysis(question, twelve_labs_result, game_info):
    try:
        # Perform RAG search
        rag_answer, rag_results = perform_rag_search(question)
        
        # Get the chat history
        chat_history = st.session_state.get('chat_history', [])
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        
        prompt = f"""
        You are an AI assistant specializing in video game knowledge. You've been given a question, an initial answer about a game, and additional context from RAG and chat history. Your task is to:

        1. Analyze the initial answer from the video-based AI.
        2. Identify any missing information or gaps in the explanation.
        3. Fill in these gaps using the RAG results and your knowledge of the game, but do not alter the original description.
        4. Consider the chat history for context.
        5. Provide a comprehensive response that combines the original answer with your additional insights.

        Game Information: {game_info}
        Question: {question}
        Initial Answer from Video: {twelve_labs_result}
        
        RAG Answer: {rag_answer}
        RAG Results:
        {rag_results}
        
        Recent Chat History:
        {history_context}

        Please provide an enhanced response:
        """

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable gaming assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error in ChatGPT RAG analysis: {str(e)}")
        return twelve_labs_result  # Return the original result if there's an error

def generate_openai_text(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
            n=1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating text with OpenAI: {str(e)}")
        return ""


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def summarize_guide_with_links(chat_history):
    prompt = """
    Based on the following chat history, create a concise guide summary including:
    1. Main objective
    2. Key points or steps
    3. Any important tips or warnings
    4. Incorporate relevant links from the internet searches

    Include the most relevant links directly in the summary text.

    Chat History:
    """
    for message in chat_history:
        prompt += f"\n{message['role'].capitalize()}: {message['content']}"

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes gaming guides and scenarios, incorporating relevant internet search results and links."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def search_video(client, index_id, prompt, video_embeddings):
    try:
        # Generate embedding for the query
        query_embedding = client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            text=prompt,
            text_truncate="end"  # Specify truncation behavior
        )

        if query_embedding.text_embedding is None:
            st.error("Failed to generate query embedding")
            return []

        # Use the text embedding
        query_vector = np.array(query_embedding.text_embedding.float)

        # Perform semantic search
        results = []
        for video_embedding in video_embeddings:
            clip_vector = np.array(video_embedding.embedding.float)
            # Calculate cosine similarity
            similarity = np.dot(query_vector, clip_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(clip_vector))
            results.append({
                'start': video_embedding.start_offset_sec,
                'end': video_embedding.end_offset_sec,
                'score': float(similarity),
                'embedding_scope': video_embedding.embedding_scope
            })

        # Sort results by score (highest to lowest)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Filter out overlapping clips
        filtered_results = []
        video_duration = max(result['end'] for result in results)
        min_clip_duration = 10  # Minimum duration of a clip in seconds
        max_clips = max(5, int(video_duration / 30))  # At least 5 clips, or more for longer videos

        for result in results:
            overlap = False
            for filtered_result in filtered_results:
                if (result['start'] < filtered_result['end'] and result['end'] > filtered_result['start']) or \
                   (abs(result['start'] - filtered_result['start']) < min_clip_duration):
                    overlap = True
                    break
            if not overlap and len(filtered_results) < max_clips:
                filtered_results.append(result)

            # Break if we have covered the entire video duration
            if len(filtered_results) >= max_clips and filtered_results[-1]['end'] >= video_duration * 0.9:
                break

        # Sort the filtered results chronologically
        filtered_results.sort(key=lambda x: x['start'])

        return filtered_results
    except Exception as e:
        st.error(f"Failed to search video: {str(e)}")
        st.error(f"Error details: {type(e).__name__}, {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []



def generate_open_ended_text(video_id, prompt, is_first_answer=False):
    try:
        if is_first_answer:
            full_prompt = f"Identify the game being played and the specific level or guide shown in this video only in the first response to the user, then answer the following question: {prompt}"
        else:
            full_prompt = prompt

        res = client.generate.text(
            video_id=video_id,
            prompt=full_prompt,
            temperature=0.1,  # Set to a very low temperature for factual responses
            stream=False
        )
        return res.data
    except Exception as e:
        st.error(f"Error generating open-ended text: {str(e)}")
        return ""

def summarize_chat_history(chat_history):
    prompt = """
    Summarize the following chat history into a concise search prompt for a video game scenario page. 
    The summary should:
    - Capture key steps and important elements mentioned in the conversation
    - Be relevant to the video content and user's intentions
    - Be structured for effective video clip search
    - Not exceed 76 tokens
    - Frame the content for a wiki-style or scenario-based gaming page
    - Highlight crucial user inputs for accurate video clip retrieval
    - Emphasize the importance of maintaining chronological order in the video clips
    - Include language that encourages sequential progression through the game or scenario

    Ensure the summary search prompt promotes a chronological flow of events based on the chat history info and the video content, using phrases like:
    - "From the beginning of the game..."
    - "As the gameplay progresses..."
    - "Moving on to the next stage..."
    - "In the final part of the scenario..."

    Chat History:
    """
    
    for message in chat_history:
        prompt += f"\n{message['role'].capitalize()}: {message['content']}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes chat histories into search prompts for video game scenario pages, emphasizing chronological order and sequential progression."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=76
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error summarizing chat history: {str(e)}")
        return ""

def extract_video_clip(video_file_path, start_time, end_time, output_file_path):
    try:
        ffmpeg_extract_subclip(video_file_path, start_time, end_time, targetname=output_file_path)
    except Exception as e:
        st.error(f"Error extracting video clip: {str(e)}")




def calculate_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def generate_scenario_page(prompt, search_results, descriptions):
    st.header(f"Scenario Page: {prompt}")
    guide_summary = summarize_guide_with_links(st.session_state['chat_history'])
    st.subheader("Guide Summary")
    st.markdown(guide_summary, unsafe_allow_html=True)

    displayed_descriptions = []
    similarity_threshold = 0.8  # Adjust this value to change sensitivity

    for i, (clip, description) in enumerate(zip(search_results, descriptions), 1):
        is_unique = True
        for displayed_description in displayed_descriptions:
            if calculate_similarity(description, displayed_description) > similarity_threshold:
                is_unique = False
                break
        
        if is_unique:
            displayed_descriptions.append(description)

            st.subheader(f"Clip {i}")
            # Use Streamlit's video playback
            start_time = clip['start']
            video_file = open(st.session_state['video_file_path'], 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, start_time=int(start_time))
            st.markdown(f"<p class='relevance-score'><strong>Relevance Score:</strong> {clip['score']:.2f}</p>", unsafe_allow_html=True)
            st.markdown("<div class='description'>", unsafe_allow_html=True)
            st.markdown("<strong>AI Description:</strong>", unsafe_allow_html=True)
            
            # Add edit functionality for description
            description_key = f"description_{i}"
            if description_key not in st.session_state:
                st.session_state[description_key] = description
            
            edited_description = st.text_area("Edit Description", st.session_state[description_key], key=f"edit_{i}")
            if st.button("Save Changes", key=f"save_{i}"):
                st.session_state[description_key] = edited_description
                st.success("Changes saved successfully!")
            
            st.markdown(st.session_state[description_key], unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='comments'>", unsafe_allow_html=True)
            st.markdown("<h4>Comments:</h4>", unsafe_allow_html=True)
            st.markdown("<p><strong>User1:</strong> This part was really challenging!</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True) # Add a separator between clips

    # Add Q&A section for the entire video
    st.subheader("Ask a question about this video")
    with st.form(key='qa_form'):
        user_question = st.text_input("Your question about the video")
        submit_button = st.form_submit_button(label='Get Answer')

    if submit_button and user_question:
        with st.spinner("Generating answer..."):
            if 'video_embeddings' in st.session_state:
                video_search_results = search_video(
                    client,
                    st.session_state['video_index_id'],
                    user_question,
                    st.session_state['video_embeddings']
                )
                if video_search_results:
                    video_search_results.sort(key=lambda x: x['start'])
                    context = ""
                    for result in video_search_results[:5]:
                        context += f"From {result['start']} to {result['end']} seconds (score: {result['score']:.2f})\n"
                    
                    prompt = f"""Based on the following video segments:

{context}

Respond to this user message in detail (about 200-250 words):
{user_question}

Include specific references to the video content where relevant, and provide a comprehensive answer that covers the main points while still being concise."""

                    twelve_labs_response = generate_open_ended_text(
                        video_id=st.session_state['video_id'],
                        prompt=prompt,
                        is_first_answer=False
                    )
                    
                    answer = chatgpt_rag_analysis(user_question, twelve_labs_response, "")
                else:
                    answer = "I apologize, but I couldn't find any relevant content in the video to answer your question. Could you please rephrase your query or ask about a different aspect of the video?"
            else:
                answer = "I'm sorry, but I couldn't find the video embeddings. Please make sure the video has been properly processed."

            st.markdown(f"<strong>Question:</strong> {user_question}", unsafe_allow_html=True)
            st.markdown(f"<strong>Answer:</strong> {answer}", unsafe_allow_html=True)

    # Add floating search bar
    st.markdown("""
    <div class="floating-search">
        <input type="text" id="floating-search-input" placeholder="Ask about the video...">
        <button onclick="searchVideo()">Ask</button>
    </div>
    """, unsafe_allow_html=True)

    # Add custom CSS to maintain styling and add floating search bar
    st.markdown("""
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
        .guide-summary { background-color: #f0f0f1; padding: 15px; margin-bottom: 20px; border-left: 5px solid #333; }
        .clip { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }
        .clip h3 { color: #333; }
        .relevance-score { font-style: italic; color: #666; }
        .description { margin-top: 10px; }
        .comments { margin-top: 15px; border-top: 1px solid #eee; padding-top: 10px; }
        video { width: 100%; max-width: 600px; }
        .floating-search {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        #floating-search-input {
            width: 300px;
            padding: 5px;
            margin-right: 10px;
        }
    </style>
    <script>
    function searchVideo() {
        var question = document.getElementById('floating-search-input').value;
        // You'll need to implement a way to trigger the Streamlit app to process this question
        // This might involve using Streamlit's component communication or other methods
        alert('Asking: ' + question);
    }
    </script>
    """, unsafe_allow_html=True)

    # No need to return HTML content as we're directly rendering with Streamlit

def main():
    st.title("Gaming Wiki Page Generator")

    # Step 1: Video Upload
    st.header("Upload Your Gameplay Video")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    # Create index if not already created
    if st.session_state['video_index_id'] is None and video_file is not None:
        with st.spinner("Creating index..."):
            create_index()

    # Upload video and get video_id
    if video_file is not None and st.session_state['video_id'] is None:
        with st.spinner("Uploading and indexing video..."):
            video_bytes = video_file.read()
            st.session_state['video_id'] = upload_video(st.session_state['video_index_id'], video_bytes)
            st.session_state['video_file_path'] = "uploaded_video.mp4"

    # Create embeddings after video upload
    if st.session_state['video_id'] is not None and 'video_embeddings' not in st.session_state:
        with st.spinner("Creating video embeddings..."):
            embedding_task_id = create_video_embeddings(st.session_state['video_file_path'], None)
            if embedding_task_id:
                video_embeddings = retrieve_embeddings(embedding_task_id)
                if video_embeddings:
                    st.session_state['video_embeddings'] = video_embeddings

    # Step 2: Generate Gist
    if st.session_state['video_id'] is not None:
        st.header("Video Gist")
        if st.button("Generate Gist"):
            with st.spinner("Generating gist..."):
                gist = generate_gist(st.session_state['video_id'])
                st.text_area("Video Gist", gist, height=150)

    # Step 3: Chatbot Interaction
    st.header("Chat with the Assistant")
    st.write("You can discuss your gaming guide info, ask about the video, or request internet searches.")

    # Display chat history
    for msg in st.session_state['chat_history']:
        if msg['role'] == 'assistant':
            st.markdown(f"**Assistant**: {msg['content']}")
        else:
            st.markdown(f"**You**: {msg['content']}")

    user_input = st.text_input("Your message:")
    search_type = st.radio("Search type:", ["Video", "Internet"])

    if st.button("Send") and user_input:
        with st.spinner("Assistant is typing..."):
            if search_type == "Video" and st.session_state['video_id'] is not None:
                assistant_reply = chat_with_bot(user_input)
            else:
                assistant_reply = perform_rag_search(user_input)
            st.markdown(f"**Assistant**: {assistant_reply}")
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['chat_history'].append({"role": "assistant", "content": assistant_reply})

    # Step 4: Generate Scenario Page
    if 'generate_page' not in st.session_state:
        st.session_state['generate_page'] = False

    if st.button("Generate Page") or st.session_state['generate_page']:
        st.session_state['generate_page'] = True
        if len(st.session_state['chat_history']) > 0:
            with st.spinner("Processing..."):
                # Summarize the chat history for search
                search_prompt = summarize_chat_history(st.session_state['chat_history'])
                st.write("Search Prompt:", search_prompt)  # Display the search prompt (you can remove this line later)

                # Search for relevant clips using the summarized search prompt and embeddings
                if 'video_embeddings' in st.session_state:
                    search_results = search_video(
                        client,
                        st.session_state['video_index_id'], 
                        search_prompt, 
                        st.session_state['video_embeddings']
                    )
                else:
                    st.error("Video embeddings not found. Please ensure the video is properly processed.")
                    return

                if not search_results:
                    st.warning("No relevant clips found.")
                    return

                # Generate detailed descriptions for each clip
                descriptions = []
                for clip in search_results:
                    prompt = f"""Analyze the video clip from {clip['start']} to {clip['end']} seconds and create a structured summary:

    1. Start with a bold title explaining the main focus.
    2. Provide a concise description of key events (2-3 sentences).
    3. Use 2-3 labeled sub-topics (e.g., "Strategy:", "Key Items:") for organization.
    4. Highlight crucial controls in [brackets] if applicable.
    5. Mention any unique player insights or tips.
    6. Note 1-2 notable game mechanics or abilities if relevant.
    7. End with a brief "Key Takeaway:".
    8. Make sure to label the timestamp start and end time in second and minutes for each relevant clip in the description so the user knows what duration of the whole clip is relevant.

    Be clear for newcomers and informative for experienced players. Limit response to 1400 characters.

    Context: {search_prompt}
    """
                    description = generate_open_ended_text(
                        st.session_state['video_id'],
                        prompt=prompt
                    )
                    descriptions.append(description)

                # Generate and display the scenario page using Streamlit components
                generate_scenario_page(search_prompt, search_results, descriptions)

                st.success("Wiki page generated successfully!")

if __name__ == "__main__":
    main()

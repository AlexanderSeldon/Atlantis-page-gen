import streamlit as st
import time
import os
import numpy as np
import cv2
from twelvelabs import TwelveLabs
from dotenv import load_dotenv
import base64
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from twelvelabs import APIStatusError, AuthenticationError
from twelvelabs.models.embed import EmbeddingsTask
from openai import OpenAI
from googleapiclient.discovery import build
from PIL import Image
import re
import difflib
import sqlite3
import json
import uuid
import io

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

def init_db():
    conn = sqlite3.connect('wiki_pages.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pages
                 (id TEXT PRIMARY KEY, content TEXT)''')
    conn.commit()
    conn.close()

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

        # Use the open-ended text generation directly
        prompt = f"""
        {'Identify the game being played and the specific level or guide shown in this video, then ' if is_first_answer else ''}
        Respond to this user message in detail (about 200-250 words):
        {user_input}

        Include specific references to the video content where relevant, and provide a comprehensive answer that covers the main points while still being concise.
        """

        twelve_labs_response = generate_open_ended_text(
            video_id=st.session_state['video_id'],
            prompt=prompt,
            is_first_answer=is_first_answer
        )

        st.text("Twelve Labs Response (Debug):")
        st.text(twelve_labs_response)  # Debug output

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

        # Get the gist from session state or generate it if not available
        gist = st.session_state.get('gist', '')
        if not gist and 'video_id' in st.session_state:
            gist = generate_gist(st.session_state['video_id'])
            st.session_state['gist'] = gist

        st.text("Game Info (Debug):")
        st.text(game_info)  # Debug output

        st.text("Gist (Debug):")
        st.text(gist)  # Debug output

        # Use the twelve_labs_response as the primary content, and enhance it with RAG results
        enhanced_response = chatgpt_rag_analysis(user_input, twelve_labs_response, game_info, messages, gist)

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

def perform_rag_search(query, game_info=None):
    try:
        if game_info:
            # Combine game information with the user's query for a more relevant search
            search_query = f"{game_info} {query}"
        else:
            search_query = query

        # Perform Google Search with the combined query
        search_results = google_service.cse().list(
            q=search_query,
            cx=GOOGLE_CSE_ID,
            num=10  # Increased from 5 to 10 to get more diverse results
        ).execute()
        items = search_results.get('items', [])

        # Extract titles and snippets from the search results
        texts = [f"{item['title']}\n{item['snippet']}" for item in items]

        # Retrieve relevant documents based on the search query
        top_texts = retrieve_relevant_documents(search_query, texts)

        # Format results with sources
        formatted_results = "RAG Search Results:\n"
        for i, text in enumerate(top_texts, 1):
            original_item = next(
                (item for item in items if f"{item['title']}\n{item['snippet']}" == text),
                None
            )
            if original_item:
                formatted_results += f"{i}. {text}\nSource: [{original_item['title']}]({original_item['link']})\n\n"

        # Create context for the OpenAI prompt
        context = "\n\n".join(top_texts)
        prompt = f"""
You are a gaming expert tasked with providing accurate and relevant information about video games. Use the following information to answer the question:

Context: {context}

Question: {query}

Instructions:
1. Provide a concise and accurate answer based on the given context.
2. Use gaming terminology appropriate for the specific game or genre mentioned.
3. If the context doesn't fully answer the question, clearly state what information is missing or uncertain.
4. Include relevant examples or comparisons if they help clarify the answer.
5. Cite specific sources when providing key information.

Answer:
"""
        # Generate the answer using OpenAI
        answer = generate_openai_text(prompt)

        # Format the final answer with sources
        formatted_answer = answer + "\n\nSources:\n"
        for item in items:
            formatted_answer += f"- [{item['title']}]({item['link']})\n"

        return formatted_answer, formatted_results

    except Exception as e:
        st.error(f"Error performing RAG search: {str(e)}")
        return "", ""

def chatgpt_rag_analysis(question, twelve_labs_result, game_info, chat_history, gist):
    try:
        # Perform RAG search with game_info included
        rag_answer, rag_results = perform_rag_search(question, game_info)

        # Extract recent chat history
        recent_chat = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

        # Combine all context
        context = f"""
Game Gist: {gist}
Recent Chat History:
{recent_chat}
Game Information: {game_info}
RAG Search Results: {rag_results}
"""

        enhancement_prompt = f"""
You are an expert gaming guide writer with deep knowledge of video games, their mechanics, strategies, and terminology. Your task is to enhance a detailed gameplay description with precise gaming knowledge while maintaining its original structure and timestamps.

Context:
{context}

User Question: {question}

Original Gameplay Description:
{twelve_labs_result}

Your task:
1. Identify the specific game and level/mission being played based on the description and context.
2. Carefully analyze the original description, noting its structure, timestamps, and key points.
3. Enhance the description by:
   a) Integrating relevant gaming terminology and jargon from the RAG results where appropriate.
   b) Explaining game mechanics or systems mentioned in the RAG results that are relevant to the observed gameplay.
   c) Clarifying specific objectives, items, or strategies using information from the RAG results.
   d) Adding any missing context or background information that would help users understand the gameplay better.
4. Maintain the original structure, including all timestamps and the sequence of events.
5. Ensure your response directly answers the user's question for step-by-step directions.
6. If narration is mentioned, incorporate and expand on the narrator's instructions using gaming knowledge from the RAG results.
7. Format the response clearly, using bullet points or numbered steps if appropriate.
8. If there are any discrepancies between the original description and the RAG results, note them and provide clarification.

Important: Only add information that is directly supported by either the original description or the RAG search results. Do not invent or assume additional details.

Provide your enhanced guide below, closely following the structure of the original description and incorporating relevant information from the RAG results:
"""

        # Generate the enhanced response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert gaming guide writer who enhances gameplay descriptions with precise gaming knowledge from RAG search results while maintaining the original structure and timestamps."
                },
                {"role": "user", "content": enhancement_prompt}
            ],
            max_tokens=1500,  # Increased to allow for more detailed enhancements
            temperature=0.3,  # Slightly increased for a balance between consistency and creativity
            n=1
        )

        enhanced_response = response.choices[0].message.content.strip()
        return enhanced_response

    except Exception as e:
        st.error(f"Error in ChatGPT RAG analysis: {str(e)}")
        return twelve_labs_result  # Return the original result if there's an error

def generate_open_ended_text(video_id, prompt, is_first_answer=False):
    try:
        if is_first_answer:
            full_prompt = f"""Identify the game being played and the specific level or guide shown in this video. Then, answer the following question:

{prompt}

Your response should:
1. Start with a clear identification of the game and level/area.
2. Provide a detailed answer to the question, referencing specific visual elements and events in the video.
3. Use precise gaming terminology relevant to the identified game.
4. Include timestamps for key moments or actions mentioned in your response.
5. Be informative for both newcomers and experienced players of the game.

Limit your response to 250-300 words."""
        else:
            full_prompt = f"""Analyze the video content and answer the following question:

{prompt}

Your response should:
1. Directly address the question using information from the video.
2. Use specific examples and timestamps from the video to support your answer.
3. Employ relevant gaming terminology and explain any game-specific concepts.
4. Provide context for your observations, relating them to broader game mechanics or strategies.
5. Be clear and informative for users of various skill levels.

Limit your response to 200-250 words."""

        res = client.generate.text(
            video_id=video_id,
            prompt=full_prompt,
            temperature=0.2,  # Slightly increased for a balance between accuracy and detail
            stream=False
        )
        return res.data
    except Exception as e:
        st.error(f"Error generating open-ended text: {str(e)}")
        return ""


def extract_frames(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            frames.append(img)
    cap.release()
    return frames

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_video_clip(video_file_path, start_time, end_time, output_file_path):
    try:
        cap = cv2.VideoCapture(video_file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * 1000:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
    except Exception as e:
        st.error(f"Error extracting video clip: {str(e)}")

def analyze_frames_with_gpt4(frames):
    analyses = []
    for i, frame in enumerate(frames):
        base64_image = encode_image(frame)
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in analyzing video game screenshots. Provide concise, relevant observations that could add context to understanding the game's progression or key moments. Focus only on significant elements that might be crucial for understanding the gameplay or story."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Briefly describe the most important elements in this game screenshot. Focus on:
1. Key gameplay moments or story progression indicators
2. Significant UI elements that show important game state
3. Notable character actions or environmental changes
Limit your response to 2-3 sentences, mentioning only the most relevant details."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ],
                }
            ],
            max_tokens=100,
        )
        analyses.append(f"Frame {i+1}: {response.choices[0].message.content}")
    return "\n\n".join(analyses)

def chatgpt_rag_analysis(question, twelve_labs_result, game_info, chat_history, gist):
    try:
        # Perform RAG search with game_info included
        rag_answer, rag_results = perform_rag_search(question, game_info)
        
        # Extract recent chat history
        recent_chat = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

        enhancement_prompt = f"""
You are an expert in video game terminology, particularly for the game and level described in the following information:

Game Information: {game_info}

Recent Chat Context:
{recent_chat}

Your task is to minimally enhance the given video analysis by adding only the game title (if missing) and essential gaming terminology FROM THE PROVIDED RAG SEARCH RESULTS.

Video Analysis:
{twelve_labs_result}

RAG Search Results (source of game terminology):
{rag_results}

Your task:
1. If the game title is not mentioned in the video analysis, add it at the beginning using information from the RAG search results.
2. Identify any generic terms in the video analysis that could be replaced with specific gaming terminology found in the RAG search results.
3. Only add or replace terms if they are present in the RAG search results AND essential for understanding the content.
4. Do not change the structure or main content of the original analysis.
5. If you make any changes, enclose the added or modified terms in [square brackets].
6. If no relevant terminology is found in the RAG search results, or if no changes are needed, return the original analysis as is.

Provide the minimally enhanced analysis below, using ONLY terminology from the RAG search results:
"""

        # Generate the enhanced response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in video game terminology who makes minimal, necessary enhancements to video analyses using only information from provided RAG search results."
                },
                {"role": "user", "content": enhancement_prompt}
            ],
            max_tokens=1200,
            temperature=0.1,  # Lower temperature for more conservative output
            n=1
        )
        
        enhanced_response = response.choices[0].message.content.strip()
        return enhanced_response
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
        min_clip_duration = max(30, video_duration * 0.05)  # Minimum of 30 seconds or 5% of video length
        max_clips = max(5, min(10, int(video_duration / 60)))  # At least 5 clips, at most 10, or 1 per minute

        for result in results:
            overlap = False
            for filtered_result in filtered_results:
                if (result['start'] < filtered_result['end'] and result['end'] > filtered_result['start']):
                    # If there's overlap, extend the existing clip if the new one has a higher score
                    if result['score'] > filtered_result['score']:
                        filtered_result['start'] = min(filtered_result['start'], result['start'])
                        filtered_result['end'] = max(filtered_result['end'], result['end'])
                        filtered_result['score'] = max(filtered_result['score'], result['score'])
                    overlap = True
                    break
            if not overlap and len(filtered_results) < max_clips:
                # Ensure minimum clip duration
                if result['end'] - result['start'] < min_clip_duration:
                    result['end'] = min(result['start'] + min_clip_duration, video_duration)
                filtered_results.append(result)

            # Break if we have covered a significant portion of the video
            if len(filtered_results) >= max_clips and filtered_results[-1]['end'] >= video_duration * 0.8:
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
    st.header("Wiki Page")
    guide_summary = summarize_guide_with_links(st.session_state['chat_history'])
    st.subheader("Guide Summary")
    st.markdown(guide_summary, unsafe_allow_html=True)

    # Generate highlights for the entire video (keep this as it's used for search)
    try:
        highlights_response = client.generate.summarize(
            video_id=st.session_state['video_id'],
            type="highlight",
            prompt="Identify the most significant moments or elements in this entire video, focusing on key gameplay events, objectives, and strategies."
        )
        # We're not displaying the highlights, but keeping them for search functionality
    except Exception as e:
        st.error(f"Error generating highlights: {str(e)}")
        return

    # Use highlights as search prompts and for clip selection
    search_results = []
    descriptions = []
    for highlight in highlights_response.highlights:
        # Use the highlight text as the search prompt
        highlight_results = search_video(
            client,
            st.session_state['video_index_id'],
            highlight.highlight,
            st.session_state['video_embeddings']
        )
        
        if highlight_results:
            # Take the top result for each highlight
            top_result = highlight_results[0]
            search_results.append(top_result)
            
            # Generate description for the clip (keeping this for consistency)
            description_prompt = f"""Analyze the video clip from {top_result['start']} to {top_result['end']} seconds and create a structured summary."""
            description = generate_open_ended_text(
                st.session_state['video_id'],
                prompt=description_prompt
            )
            descriptions.append(description)

    # Initialize the list of displayed clips in session state if not already present
    if 'displayed_clips' not in st.session_state:
        st.session_state['displayed_clips'] = list(enumerate(zip(search_results, descriptions), 1))

    clip_number = 1  # Initialize clip number

    # Create a copy of the displayed clips to iterate over
    displayed_clips_copy = st.session_state['displayed_clips'].copy()

    for i, (clip, description) in displayed_clips_copy:
        st.subheader(f"Clip {clip_number}")
        clip_number += 1  # Increment clip number
        
        # Use Streamlit's video playback
        start_time = clip['start']
        end_time = clip['end']
        video_file = open(st.session_state['video_file_path'], 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time=int(start_time))
        st.markdown(
            f"<p class='relevance-score'><strong>Relevance Score:</strong> {clip['score']:.2f}</p>",
            unsafe_allow_html=True
        )

        # Generate summary for the clip using Twelve Labs API
        try:
            summary_response = client.generate.summarize(
                video_id=st.session_state['video_id'],
                type="summary",
                prompt=f"Summarize the video segment from {start_time} to {end_time} seconds in a concise, informative manner."
            )

            # Display the generated summary
            st.markdown("### Clip Summary")
            st.markdown(f"**Timestamp: {start_time} - {end_time} seconds**")
            st.markdown(summary_response.summary, unsafe_allow_html=True)

            # Update the description_key with the new summary
            description_key = f"description_{i}"
            st.session_state[description_key] = summary_response.summary

        except Exception as e:
            st.error(f"Error generating clip summary: {str(e)}")
            description_key = f"description_{i}"
            st.session_state[description_key] = description  # Fallback to original description

        # Rest of the code for comments, time range adjustment, etc. remains the same
        

    # Rest of the function remains unchanged
    

        # Add comment section here, before the "Adjust Relevant Time Range" feature
        st.subheader("Comments")
        comment_key = f"comments_{i}"
        if comment_key not in st.session_state:
            st.session_state[comment_key] = []

        # Display existing comments
        for comment in st.session_state[comment_key]:
            st.markdown(f"**{comment['author']}**: {comment['text']}")
            st.markdown(f"<small>{comment['timestamp']}</small>", unsafe_allow_html=True)

        # Add new comment
        new_comment = st.text_area("Add a comment", key=f"new_comment_{i}")
        if st.button("Post Comment", key=f"post_comment_{i}"):
            if new_comment:
                comment = {
                    "author": "User",  # You can implement user authentication to get the actual username
                    "text": new_comment,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state[comment_key].append(comment)
                st.success("Comment posted successfully!")
                st.experimental_rerun()

        # Adjust Relevant Start and Stop Times
        st.markdown("<strong>Adjust Relevant Time Range:</strong>", unsafe_allow_html=True)
        relevant_start_key = f"relevant_start_{i}"
        relevant_end_key = f"relevant_end_{i}"

        # Set session state values within the clip's time range if not already set
        if relevant_start_key not in st.session_state:
            st.session_state[relevant_start_key] = start_time
        if relevant_end_key not in st.session_state:
            st.session_state[relevant_end_key] = end_time

        col1, col2 = st.columns(2)
        with col1:
            new_start = st.number_input(
                "Relevant Start Time (seconds)",
                min_value=float(start_time),
                max_value=float(end_time),
                value=max(float(start_time), min(float(end_time), float(st.session_state[relevant_start_key]))),
                key=f"relevant_start_input_{i}"
            )
        with col2:
            new_end = st.number_input(
                "Relevant End Time (seconds)",
                min_value=float(start_time),
                max_value=float(end_time),
                value=min(float(end_time), max(float(start_time), float(st.session_state[relevant_end_key]))),
                key=f"relevant_end_input_{i}"
            )
        if st.button("Save Relevant Time Range", key=f"save_time_{i}"):
            if new_start >= start_time and new_end <= end_time and new_start < new_end:
                st.session_state[relevant_start_key] = new_start
                st.session_state[relevant_end_key] = new_end
                st.success("Relevant time range updated successfully!")
            else:
                st.error(
                    "Invalid time range. Please ensure the start time is before the end time and within the clip duration."
                )

        # Display the adjusted relevant time range
        st.markdown(
            f"**Relevant Time Range:** {st.session_state[relevant_start_key]}s to {st.session_state[relevant_end_key]}s"
        )

        st.markdown("<div class='description'>", unsafe_allow_html=True)
        st.markdown("<strong>AI Description:</strong>", unsafe_allow_html=True)
        # Add edit functionality for description
        description_key = f"description_{i}"
        if description_key not in st.session_state:
            st.session_state[description_key] = description
        edited_description = st.text_area(
            "Edit Description", st.session_state[description_key], key=f"edit_{i}"
        )
        if st.button("Save AI Description Changes", key=f"save_ai_{i}"):
            st.session_state[description_key] = edited_description
            st.success("AI Description changes saved successfully!")
        st.markdown(st.session_state[description_key], unsafe_allow_html=True)

        # Prompt Input for Generating New AI Description
        st.markdown("<strong>Generate New AI Description:</strong>", unsafe_allow_html=True)
        ai_prompt = st.text_area(
            "Enter a prompt to generate a new AI description for this clip:",
            key=f"ai_prompt_{i}"
        )
        if st.button("Generate New AI Description", key=f"generate_ai_{i}"):
            with st.spinner("Generating new AI description..."):
                # Use the adjusted relevant times
                relevant_start = st.session_state[relevant_start_key]
                relevant_end = st.session_state[relevant_end_key]
                clip_context = f"Analyze the video clip from {relevant_start} to {relevant_end} seconds."
                
                # Prepare chat history context
                chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['chat_history']])
                
                full_prompt = f"""
You have access to the entire video context and chat history. Focus your description on the specific clip from {relevant_start} to {relevant_end} seconds, but use the broader context to inform your analysis.

Video Context: {st.session_state.get('gist', 'No gist available')}

Chat History:
{chat_context}

{clip_context}

Provide a detailed description of the gameplay in this clip, focusing on:
1. Specific game mechanics and systems visible in the footage
2. Key actions or decisions made by the player
3. Any notable events or changes in the game state
4. Relevant strategic insights or tips for players

Your response should:
1. Start with a bold title summarizing the main focus or key event in this clip.
2. Use precise, game-specific terminology where appropriate.
3. Include timestamps for key moments or actions mentioned in your response.
4. Be informative for both newcomers and experienced players of the game.

Limit your response to 250-300 words.

User Prompt: {ai_prompt}
"""
                # Generate initial response using the open-ended function
                initial_response = generate_open_ended_text(
                    video_id=st.session_state['video_id'],
                    prompt=full_prompt,
                    is_first_answer=False
                )
                # Enhance the response using chatgpt_rag_analysis
                enhanced_response = chatgpt_rag_analysis(ai_prompt, initial_response, "")
                # Update the AI description
                st.session_state[description_key] = enhanced_response
                st.success("New AI Description generated successfully!")
                # Display the new description
                st.markdown(enhanced_response, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Add user content section with save functionality
        st.subheader("Add Your Content")
        # User text
        user_text_key = f"user_text_{i}"
        if user_text_key not in st.session_state:
            st.session_state[user_text_key] = ""
        user_text = st.text_area(
            "Add your text", value=st.session_state[user_text_key], key=f"input_{user_text_key}"
        )
        # User link
        user_link_key = f"user_link_{i}"
        if user_link_key not in st.session_state:
            st.session_state[user_link_key] = ""
        user_link = st.text_input(
            "Paste a link", value=st.session_state[user_link_key], key=f"input_{user_link_key}"
        )
        # User image
        user_image_key = f"user_image_{i}"
        user_image = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"], key=f"upload_{user_image_key}"
        )
        # Save user content button
        if st.button("Save User Content", key=f"save_user_{i}"):
            st.session_state[user_text_key] = user_text
            st.session_state[user_link_key] = user_link
            if user_image is not None:
                # Read the file and save it as bytes in the session state
                st.session_state[user_image_key] = user_image.read()
            st.success("User content saved successfully!")

        # Display saved user content
        if st.session_state[user_text_key]:
            st.markdown(
                f"<div class='user-content'><p>{st.session_state[user_text_key]}</p></div>",
                unsafe_allow_html=True
            )
        if st.session_state[user_link_key]:
            st.markdown(
                f"<div class='user-content'><a href='{st.session_state[user_link_key]}' target='_blank'>{st.session_state[user_link_key]}</a></div>",
                unsafe_allow_html=True
            )
        if user_image_key in st.session_state and st.session_state[user_image_key] is not None:
            # Display the image using st.image
            st.image(st.session_state[user_image_key], caption="User uploaded image")

        # Add delete button
        if st.button(f"Delete Clip {i}", key=f"delete_clip_{i}"):
            st.session_state['displayed_clips'] = [
                (idx, c) for idx, c in st.session_state['displayed_clips'] if idx != i
            ]
            st.experimental_rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

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

    # Add publish button
    if st.button("Publish Page"):
        page_id = str(uuid.uuid4())
        page_content = {
            "prompt": prompt,
            "guide_summary": guide_summary,
            "clips": [
                {"clip": clip, "description": st.session_state[f"description_{i}"]}
                for i, (clip, _) in st.session_state['displayed_clips']
            ]
        }
        save_page(page_id, page_content)
        st.success(f"Page published! Share this link: https://your-app-url.com/?page_id={page_id}")

    # Add floating search bar
    # Add floating search bar
    st.markdown("""
    <div class="floating-search">
    <input type="text" id="floating-search-input" placeholder="Ask about the video...">
    <button onclick="searchVideo()">Ask</button>
    </div>
    """, unsafe_allow_html=True)

  # This closes the generate_scenario_page function

def save_page(page_id, content):
    conn = sqlite3.connect('wiki_pages.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO pages (id, content) VALUES (?, ?)",
              (page_id, json.dumps(content)))
    conn.commit()
    conn.close()

def get_page(page_id):
    conn = sqlite3.connect('wiki_pages.db')
    c = conn.cursor()
    c.execute("SELECT content FROM pages WHERE id = ?", (page_id,))
    result = c.fetchone()
    conn.close()
    return json.loads(result[0]) if result else None

def main():
    init_db()  # Initialize the database

    # Check if a page_id is provided in the URL
    params = st.experimental_get_query_params()
    if 'page_id' in params:
        page_id = params['page_id'][0]
        page_content = get_page(page_id)
        if page_content:
            generate_scenario_page(
                prompt=page_content['prompt'],
                search_results=[clip_info['clip'] for clip_info in page_content['clips']],
                descriptions=[clip_info['description'] for clip_info in page_content['clips']]
            )
        else:
            st.error("Page not found.")
        return

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
        if st.session_state['video_id'] is not None and 'video_embeddings' in st.session_state:
            with st.spinner("Processing..."):
                # Generate highlights for the entire video
                try:
                    highlights_response = client.generate.summarize(
                        video_id=st.session_state['video_id'],
                        type="highlight",
                        prompt="Identify the most significant moments or elements in this entire video, focusing on key gameplay events, objectives, and strategies."
                    )
                    st.write("Video Highlights:")
                    for highlight in highlights_response.highlights:
                        st.write(f"* {highlight.highlight} ({highlight.start} - {highlight.end} seconds)")
                except Exception as e:
                    st.error(f"Error generating highlights: {str(e)}")
                    return

                # Use highlights as search prompts
                search_results = []
                descriptions = []
                for highlight in highlights_response.highlights:
                    highlight_results = search_video(
                        client,
                        st.session_state['video_index_id'],
                        highlight.highlight,
                        st.session_state['video_embeddings']
                    )
                    
                    if highlight_results:
                        # Take the top result for each highlight
                        top_result = highlight_results[0]
                        search_results.append(top_result)
                        
                        start_time = top_result['start']
                        end_time = top_result['end']
                        # Generate description for the clip
                        description_prompt = f"""Analyze the video clip from {start_time} to {end_time} seconds and create a concise guide-style summary:

1. Start with a bold title describing the main focus of this specific clip segment.
2. Describe the key actions, events, or information presented in this clip segment only.
3. Use bullet points for any specific instructions or important details.
4. Mention any relevant game mechanics or items that appear in this clip.
5. If applicable, note any challenges or strategies specific to this part of the game.
6. Avoid phrases like "players are informed" or "the guide emphasizes". Instead, state information directly.
7. Do not recap the overall game or task goal in every summary; focus only on what's happening in this clip.

Your summary should:
- Be clear and concise, written as if it's a section of a game guide.
- Use precise terminology relevant to the game.
- Include the exact timestamp range for this clip.
- Be informative for both newcomers and experienced players.

Limit your response to 150-200 words.

Context: This clip is part of a larger gameplay video.
"""
                        description = generate_open_ended_text(
                            st.session_state['video_id'],
                            prompt=description_prompt
                        )
                        descriptions.append(description)

                # Generate and display the scenario page using Streamlit components
                generate_scenario_page("", search_results, descriptions)

                st.success("Wiki page generated successfully!")
        else:
            st.error("Please ensure the video is uploaded and processed.")

if __name__ == "__main__":
    main()

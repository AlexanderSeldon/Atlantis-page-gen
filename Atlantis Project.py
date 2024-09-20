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

# Load environment variables
load_dotenv("Atlantis.env")

# API Keys
TWELVE_LABS_API_KEY = os.getenv('TWELVE_LABS_API_KEY')

# Initialize Twelve Labs client
client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

        # Search for relevant content in the video
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

        # Generate a summary based on the search results
        if search_results:
            response_text = generate_open_ended_text(
                video_id=st.session_state['video_id'],
                prompt=f"Based on the video content, respond to this user message: {user_input}"
            )
        else:
            response_text = "No relevant content found in the video for your question."

        messages.append({"role": "assistant", "content": response_text})
        st.session_state['chat_history'] = messages
        return response_text
    except Exception as e:
        st.error(f"Error during chatbot conversation: {str(e)}")
        return "I'm sorry, but an error occurred while processing your request."

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



def search_video(client, index_id, prompt, video_embeddings):
    try:
        # Generate embedding for the query
        query_embedding = client.embed.create(
            engine_name="Marengo-retrieval-2.6",
            text=prompt,
            text_truncate="end"  # Add this line to specify truncation behavior
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

        # Sort results by similarity score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top 10 results
        return results[:10]
    except Exception as e:
        st.error(f"Failed to search video: {str(e)}")
        st.error(f"Error details: {type(e).__name__}, {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []



def generate_open_ended_text(video_id, prompt):
    try:
        res = client.generate.text(
            video_id=video_id,
            prompt=prompt,
            temperature=0.7,  # Optional, defaults to 0.7 if not specified
            stream=False  # Optional, defaults to false if not specified
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

    Chat History:
    """
    
    for message in chat_history:
        prompt += f"\n{message['role'].capitalize()}: {message['content']}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes chat histories into search prompts for video game scenario pages."},
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

def generate_scenario_page(prompt, search_results, descriptions):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Scenario Page: {prompt}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0 auto;
                padding: 20px;
                max-width: 800px;
            }}
            .clip {{
                margin-bottom: 30px;
            }}
            .clip video {{
                width: 100%;
                height: auto;
            }}
            .comments {{
                margin-top: 20px;
            }}
            .comment {{
                border-top: 1px solid #ccc;
                padding-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Scenario Page</h1>
        <h2>{prompt}</h2>
    """

    for i, (clip, description) in enumerate(zip(search_results, descriptions), 1):
        start_time = clip['start']
        end_time = clip['end']

        html_content += f"""
        <div class="clip" id="clip-{i}">
            <h3>Clip {i}</h3>
            <video controls id="clip-{i}-video" data-start-time="{start_time}" data-end-time="{end_time}">
                <source src="{st.session_state['video_file_path']}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <script>
                (function() {{
                    var video = document.getElementById('clip-{i}-video');
                    var startTime = {start_time};
                    var endTime = {end_time};

                    video.addEventListener('play', function() {{
                        this.currentTime = startTime;
                    }});

                    video.addEventListener('timeupdate', function() {{
                        if (this.currentTime >= endTime) {{
                            this.pause();
                        }}
                    }});
                }})();
            </script>
            <p><strong>Relevance Score:</strong> {clip['score']:.2f}</p>
            <p><strong>AI Description:</strong> {description}</p>
            <div class="comments">
                <h4>Comments</h4>
                <!-- Comments would be dynamically added here -->
                <div class="comment">
                    <p><strong>User1:</strong> This part was really challenging!</p>
                </div>
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """
    return html_content


def main():
    st.title("Gaming Scenario Generator")

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
    if st.session_state['video_id'] is not None:
        st.header("Chat with the Assistant to Create Your Scenario")
        st.write("You can discuss with the assistant to refine your scenario based on the video content.")

        # Display chat history
        for msg in st.session_state['chat_history']:
            if msg['role'] == 'assistant':
                st.markdown(f"**Assistant**: {msg['content']}")
            else:
                st.markdown(f"**You**: {msg['content']}")

        user_input = st.text_input("Your message:")
        if st.button("Send") and user_input:
            with st.spinner("Assistant is typing..."):
                assistant_reply = chat_with_bot(user_input)
                st.markdown(f"**Assistant**: {assistant_reply}")

    # Step 4: Generate Scenario Page
    if st.button("Generate Scenario") and len(st.session_state['chat_history']) > 0:
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
                description = generate_open_ended_text(
                    st.session_state['video_id'],
                    prompt=f"Provide a detailed description of this clip from {clip['start']} to {clip['end']} seconds in the context of: {search_prompt}. Include relevant gameplay elements, strategies, or key events."
                )
                descriptions.append(description)

            # Generate HTML content for the scenario page
            html_content = generate_scenario_page(search_prompt, search_results, descriptions)

            # Display the scenario page
            st.subheader("Generated Scenario Page")
            st.components.v1.html(html_content, height=800, scrolling=True)

            # Provide an option to download the scenario page
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="scenario_page.html">Download Scenario Page</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

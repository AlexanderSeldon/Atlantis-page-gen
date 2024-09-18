import streamlit as st
import time
import os
from twelvelabs import TwelveLabs
from dotenv import load_dotenv
import base64
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from twelvelabs import APIStatusError, AuthenticationError

# Load environment variables
load_dotenv("Atlantis.env")

# API Keys
TWELVE_LABS_API_KEY = os.getenv('TWELVE_LABS_API_KEY')

# Initialize Twelve Labs client
client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)

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
        search_results = search_video(st.session_state['video_index_id'], user_input)

        # Generate a summary based on the search results
        if search_results:
            response_text = generate_open_ended_text(
                video_id=st.session_state['video_id'],
                prompt=f"Based on the video content, respond to this user message: {user_input}"
            )
        else:
            response_text = "No relevant content found."

        messages.append({"role": "assistant", "content": response_text})
        st.session_state['chat_history'] = messages
        return response_text
    except Exception as e:
        st.error(f"Error during chatbot conversation: {str(e)}")
        return ""

def search_video(index_id, prompt):
    try:
        search_results = client.search.query(
            index_id,
            ["visual", "conversation", "text_in_video", "logo"],  # Options as a positional argument
            query_text=prompt,
            threshold="low",
            page_limit=10
        )
        return search_results.data
    except Exception as e:
        st.error(f"Failed to search video: {str(e)}")
        return []


def generate_summary(video_id, summary_type="summary", prompt=""):
    try:
        res = client.generate.summarize(
            video_id=video_id, 
            type=summary_type, 
            prompt=prompt
        )
        return res.summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

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

def extract_video_clip(video_file_path, start_time, end_time, output_file_path):
    try:
        ffmpeg_extract_subclip(video_file_path, start_time, end_time, targetname=output_file_path)
    except Exception as e:
        st.error(f"Error extracting video clip: {str(e)}")

def generate_scenario_page(prompt, search_results, summaries):
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

    for i, (clip, summary) in enumerate(zip(search_results, summaries), 1):
        start_time = int(clip.start) if hasattr(clip, 'start') else 0
        end_time = int(clip.end) if hasattr(clip, 'end') else 0

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
            <p><strong>Relevance Score:</strong> {getattr(clip, 'score', 'N/A')}</p>
            <p><strong>AI Summary:</strong> {summary}</p>
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
            # Combine the conversation history into a single prompt
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['chat_history']])

            # Search for relevant clips using the conversation text
            search_results = search_video(st.session_state['video_index_id'], conversation_text)
            if not search_results:
                st.warning("No relevant clips found.")
                return

            # Generate summaries for each clip
            summaries = []
            for clip in search_results:
                start_time = int(clip.start) if hasattr(clip, 'start') else 0
                end_time = int(clip.end) if hasattr(clip, 'end') else 0
                summary = generate_summary(
                    st.session_state['video_id'],
                    summary_type="highlight",
                    prompt=f"Generate a highlight for this clip in the context of: {conversation_text}"
                )
                summaries.append(summary)

            # Generate HTML content for the scenario page
            html_content = generate_scenario_page(conversation_text, search_results, summaries)

            # Display the scenario page
            st.subheader("Generated Scenario Page")
            st.components.v1.html(html_content, height=800, scrolling=True)

            # Provide an option to download the scenario page
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="scenario_page.html">Download Scenario Page</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

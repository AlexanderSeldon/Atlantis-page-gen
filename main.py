import os
import streamlit as st
from twelvelabs import TwelveLabs
from Atlantis_Project import (
    chat_with_bot, create_index, create_video_embeddings, 
    generate_gist, generate_open_ended_text, generate_scenario_page, 
    get_page, perform_rag_search, retrieve_embeddings, 
    search_video, upload_video
)
from db import authenticate_user, init_db, is_username_taken, store_user

# API Keys
TWELVE_LABS_API_KEY = os.getenv('TWELVE_LABS_API_KEY')

# Initialize Twelve Labs client
client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)

def initialize_session_state():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'video_index_id' not in st.session_state:
        st.session_state.video_index_id = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def show_auth_page():
    """Display authentication page."""
    st.title("Welcome to Gaming Wiki Generator")
    
    # Create tabs for Login and Signup
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.header("Sign Up")
        new_username = st.text_input("Username", key="signup_username")
        new_password = st.text_input("Password", type="password", key="signup_password")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["viewer", "creator"])
        
        if st.button("Sign Up"):
            if is_username_taken(new_username):
                st.error("Username already taken")
            else:
                if store_user(new_username, new_password, email, role):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Error creating account")

def show_wiki_generator():
    """Display wiki generator interface for creators."""
    st.title("Gaming Wiki Generator")
    st.write(f"Welcome, {st.session_state.username}! (Role: {st.session_state.role})")
    
    if st.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    # Step 1: Video Upload
    st.header("Upload Your Gameplay Video")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    # Create index if not already created
    if st.session_state.video_index_id is None and video_file is not None:
        with st.spinner("Creating index..."):
            create_index()

    if video_file is not None and st.session_state.video_id is None:
        with st.spinner("Uploading and indexing video..."):
            video_bytes = video_file.read()
            st.session_state.video_id = upload_video(st.session_state.video_index_id, video_bytes)
            st.session_state.video_file_path = "uploaded_video.mp4"

    # Create embeddings after video upload
    if st.session_state.video_id is not None and 'video_embeddings' not in st.session_state:
        with st.spinner("Creating video embeddings..."):
            embedding_task_id = create_video_embeddings(st.session_state.video_file_path, None)
            if embedding_task_id:
                video_embeddings = retrieve_embeddings(embedding_task_id)
                if video_embeddings:
                    st.session_state.video_embeddings = video_embeddings

    # Rest of your existing wiki generator code...
    # (Generate gist, chatbot interaction, and generate scenario page sections remain the same)

def show_viewer_page():
    """Display viewer interface."""
    st.title("Gaming Wiki Viewer")
    st.write(f"Welcome, {st.session_state.username}! (Role: {st.session_state.role})")
    
    if st.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
        
    st.info("As a viewer, you can browse existing wiki pages. The generation feature is only available to creators.")
    # Add viewer-specific functionality here

def main():
    init_db()  # Initialize the database
    initialize_session_state()

    # Check for page_id in URL parameters
    params = dict(st.query_params)
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

    if not st.session_state.logged_in:
        show_auth_page()
    else:
        if st.session_state.role == "creator":
            show_wiki_generator()
        elif st.session_state.role == "viewer":
            show_viewer_page()
        else:
            st.error("Invalid user role")

if __name__ == "__main__":
    main()
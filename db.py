import sqlite3
import bcrypt
import streamlit as st

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect('wiki_pages.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password BLOB NOT NULL,
        email TEXT NOT NULL,
        role TEXT NOT NULL
    )
    ''')
    
    # Create pages table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        author TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def is_username_taken(username):
    """Check if username already exists in database."""
    conn = sqlite3.connect('wiki_pages.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return True
    conn.close()
    return False

def store_user(username, password, email, role):
    """Store a new user in the database."""
    try:
        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        conn = sqlite3.connect('wiki_pages.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)", 
                 (username, hashed_password, email, role))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing user: {e}")
        return False

def authenticate_user(username, password):
    """Authenticate user and set their role in session state."""
    try:
        conn = sqlite3.connect('wiki_pages.db')
        c = conn.cursor()
        c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        
        if result and bcrypt.checkpw(password.encode("utf-8"), result[0]):
            st.session_state.role = result[1]
            conn.close()
            return True
        conn.close()
        return False
    except Exception as e:
        print(f"Authentication error: {e}")
        return False

def get_user_role(username):
    """Get the role of a specific user."""
    conn = sqlite3.connect('wiki_pages.db')
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None
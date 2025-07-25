import streamlit as st
import os
import json


def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #8B7355;
        --primary-light: #A68B6F;
        --primary-dark: #6B5640;
        --background: #FAFAF8;
        --surface: #FFFFFF;
        --text-primary: #2C2416;
        --text-secondary: #6B5D54;
        --border: #E8E2DC;
        --hover: #F5F2EF;
        --success: #7A9A65;
        --error: #C67B5C;
    }

    /* Streamlit customization */
    .stApp {
        background-color: var(--background);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: var(--surface);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(139, 115, 85, 0.08);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        font-weight: 600;
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: var(--surface);
        border: 2px dashed var(--border);
        border-radius: 12px;
    }

    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(122, 154, 101, 0.1);
        color: var(--success);
    }

    .stError {
        background-color: rgba(198, 123, 92, 0.1);
        color: var(--error);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: var(--surface);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text-primary);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary);
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'searches_today' not in st.session_state:
        st.session_state['searches_today'] = 0

    if 'total_docs' not in st.session_state:
        # Count existing documents if database exists
        if os.path.exists('data/vector_db.json'):
            try:
                with open('data/vector_db.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sources = set()
                    for doc in data.get('documents', []):
                        sources.add(doc.get('source', ''))
                    st.session_state['total_docs'] = len(sources)
            except:
                st.session_state['total_docs'] = 0
        else:
            st.session_state['total_docs'] = 0

    if 'show_answer' not in st.session_state:
        st.session_state['show_answer'] = False

    if 'current_answer' not in st.session_state:
        st.session_state['current_answer'] = None

    if 'feedback_submitted' not in st.session_state:
        st.session_state['feedback_submitted'] = False


def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
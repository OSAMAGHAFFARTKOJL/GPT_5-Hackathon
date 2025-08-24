import streamlit as st

def render_sidebar():
    """Render sidebar and return configuration"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/username/repository")
        github_token = st.text_input("GitHub Token (Optional)", type="password", help="For higher rate limits")
        aiml_api_key = st.text_input("AIML API Key", type="password", help="For GPT-5 powered analysis and Q&A")
        google_api_key = st.text_input("Google API Key", type="password", help="For embeddings (required for RAG)")
        
        st.header("🎨 Visualization Options")
        layout_type = st.selectbox("Graph Layout", ["spring", "kamada_kawai", "circular", "shell", "random"])
        show_function_calls = st.checkbox("Show Function Calls", value=True)
        show_imports = st.checkbox("Show Imports/Requires", value=True)
        show_file_links = st.checkbox("Show File Links (CSS/JS/Images)", value=True)
        show_folder_structure = st.checkbox("Show Folder Relationships", value=True)
        
        min_connections = st.slider("Minimum Connections to Show", 0, 10, 1)
        
        analyze_button = st.button("Analyze Repository")
    
    return {
        'repo_url': repo_url,
        'github_token': github_token,
        'aiml_api_key': aiml_api_key,
        'google_api_key': google_api_key,
        'layout_type': layout_type,
        'show_function_calls': show_function_calls,
        'show_imports': show_imports,
        'show_file_links': show_file_links,
        'show_folder_structure': show_folder_structure,
        'min_connections': min_connections,
        'analyze_button': analyze_button
    }
import streamlit as st
import os
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from components.qa_section import render_qa_section
from utils.analyzer import AdvancedDependencyAnalyzer
from utils.session_state import initialize_session_state

# Set page configuration
st.set_page_config(page_title="Code Compass", layout="wide")

# App title and description
st.title("🔗 Code Compass")
st.markdown("Open-Source Contribution Helper")
st.markdown("Understand, visualize, and contribute to GitHub repositories easily. Analyze dependencies, get AI explanations, find contribution opportunities, and ask questions.")

# Initialize session state
initialize_session_state()

# Render sidebar and get configuration
config = render_sidebar()

# Main logic
if config['analyze_button'] and config['repo_url']:
    # Reset analysis state
    st.session_state.repo_analyzed = False
    
    analyzer = AdvancedDependencyAnalyzer(config['aiml_api_key'], config['google_api_key'])
    st.session_state.analyzer = analyzer
    
    username, repo_name = analyzer.extract_repo_info(config['repo_url'])
    
    if not username or not repo_name:
        st.error("❌ Invalid GitHub repository URL")
    else:
        with st.spinner("📦 Cloning and processing repository..."):
            st.session_state.repo_path = analyzer.clone_repo(config['repo_url'])
            vectorstore_built = analyzer.build_vectorstore(st.session_state.repo_path)
            if vectorstore_built:
                st.session_state.vectorstore = True
        
        if st.session_state.vectorstore:
            with st.spinner("🔍 Generating contribution opportunities..."):
                st.session_state.contribution_report = analyzer.generate_contribution_report()
                st.session_state.repo_summary = analyzer.summarize_repo(st.session_state.contribution_report)
            
            with st.spinner("📦 Fetching repository files for graph..."):
                files, repo_structure = analyzer.get_repo_contents(username, repo_name, config['github_token'])
                st.session_state.files_data = files
                st.session_state.repo_structure_data = repo_structure
            
            if files:
                options = {
                    "show_function_calls": config['show_function_calls'],
                    "show_imports": config['show_imports'],
                    "show_file_links": config['show_file_links'],
                    "show_folder_structure": config['show_folder_structure'],
                    "min_connections": config['min_connections']
                }
                
                with st.spinner("🔍 Analyzing dependencies..."):
                    graph, file_dependencies = analyzer.create_dependency_graph(files, repo_structure, options)
                    st.session_state.graph_data = graph
                    st.session_state.file_dependencies_data = file_dependencies
                
                st.session_state.repo_analyzed = True
                st.success("✅ Repository analysis completed!")

# Render main dashboard
render_dashboard(config)

# Render Q&A section
render_qa_section()

# Show welcome message if not analyzed
if not st.session_state.repo_analyzed:
    st.info("👆 Enter a GitHub repository URL and click 'Analyze Repository' to start!")
    
    # Show example usage
    st.subheader("🔥 Features")
    st.markdown("""
    - **Smart Repository Analysis**: Automatically analyze code structure and dependencies
    - **Interactive Visualization**: Explore file relationships with an interactive graph
    - **AI-Powered Q&A**: Ask questions about the code and get intelligent answers
    - **Contribution Guidance**: Get personalized suggestions for how to contribute
    - **Error Solving**: Ask about errors and get step-by-step solutions with code examples
    - **Real-time Filters**: Adjust visualization filters without re-analyzing
    
    **Example Questions You Can Ask:**
    - "How does the main authentication system work?"
    - "Fix this error: ImportError: No module named 'requests'"
    - "What are the main entry points of this application?"
    - "How do I set up the development environment?"
    - "Explain the database connection logic"
    """)

elif not st.session_state.get('vectorstore'):
    st.warning("⚠️ Repository analysis incomplete. Please check your API keys and try again.")
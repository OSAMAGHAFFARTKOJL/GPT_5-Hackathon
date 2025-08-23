import streamlit as st
import requests
# heavy visualization responsibilities moved to core.visualization
import base64
import os
import json
import re
from collections import defaultdict
from urllib.parse import urlparse, urljoin
from typing import List
import tempfile, subprocess, glob, shutil

from core.analyzer import AdvancedDependencyAnalyzer
from core.visualization import create_enhanced_visualization, create_statistics_dashboard

# Set page configuration
st.set_page_config(page_title="Code Compass", layout="wide")

# App title and description
st.title("🔗 Code Compass")
st.markdown("Open-Source Contribution Helper")
st.markdown("Understand, visualize, and contribute to GitHub repositories easily. Analyze dependencies, get AI explanations, find contribution opportunities, and ask questions.")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'repo_path' not in st.session_state:
    st.session_state.repo_path = None
if 'contribution_report' not in st.session_state:
    st.session_state.contribution_report = None
if 'file_summaries' not in st.session_state:
    st.session_state.file_summaries = {}
if 'repo_summary' not in st.session_state:
    st.session_state.repo_summary = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'files_data' not in st.session_state:
    st.session_state.files_data = None
if 'repo_structure_data' not in st.session_state:
    st.session_state.repo_structure_data = None
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'file_dependencies_data' not in st.session_state:
    st.session_state.file_dependencies_data = None
if 'repo_analyzed' not in st.session_state:
    st.session_state.repo_analyzed = False

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/username/repository")
    github_token = st.text_input("GitHub Token (Optional)", type="password", help="For higher rate limits")
    groq_api_key = st.text_input("Groq API Key", type="password", help="For AI-powered analysis and Q&A")
    google_api_key = st.text_input("Google API Key", type="password", help="For embeddings (required for RAG)")
    
    st.header("🎨 Visualization Options")
    layout_type = st.selectbox("Graph Layout", ["spring", "kamada_kawai", "circular", "shell", "random"])
    show_function_calls = st.checkbox("Show Function Calls", value=True)
    show_imports = st.checkbox("Show Imports/Requires", value=True)
    show_file_links = st.checkbox("Show File Links (CSS/JS/Images)", value=True)
    show_folder_structure = st.checkbox("Show Folder Relationships", value=True)
    
    min_connections = st.slider("Minimum Connections to Show", 0, 10, 1)
    
    analyze_button = st.button("Analyze Repository")

# visualization helpers now live in core.visualization

def update_visualization():
    """Update visualization when filters change without re-analyzing the repository"""
    if st.session_state.repo_analyzed and st.session_state.files_data is not None:
        options = {
            "show_function_calls": show_function_calls,
            "show_imports": show_imports,
            "show_file_links": show_file_links,
            "show_folder_structure": show_folder_structure,
            "min_connections": min_connections
        }
        
        with st.spinner("🎨 Updating visualization..."):
            graph, file_dependencies = st.session_state.analyzer.create_dependency_graph(
                st.session_state.files_data, 
                st.session_state.repo_structure_data, 
                options
            )
            st.session_state.graph_data = graph
            st.session_state.file_dependencies_data = file_dependencies
            
            html_content = create_enhanced_visualization(graph, layout_type)
            return html_content, graph, file_dependencies
    return None, None, None

# Main logic
if analyze_button and repo_url:
    # Reset analysis state
    st.session_state.repo_analyzed = False
    
    analyzer = AdvancedDependencyAnalyzer(groq_api_key, google_api_key)
    st.session_state.analyzer = analyzer
    
    username, repo_name = analyzer.extract_repo_info(repo_url)
    
    if not username or not repo_name:
        st.error("❌ Invalid GitHub repository URL")
    else:
        with st.spinner("📦 Cloning and processing repository..."):
            st.session_state.repo_path = analyzer.clone_repo(repo_url)
            vectorstore_built = analyzer.build_vectorstore(st.session_state.repo_path)
            if vectorstore_built:
                st.session_state.vectorstore = True
        
        if st.session_state.vectorstore:
            with st.spinner("🔍 Generating contribution opportunities..."):
                st.session_state.contribution_report = analyzer.generate_contribution_report()
                st.session_state.repo_summary = analyzer.summarize_repo(st.session_state.contribution_report)
            
            with st.spinner("📦 Fetching repository files for graph..."):
                files, repo_structure = analyzer.get_repo_contents(username, repo_name, github_token)
                st.session_state.files_data = files
                st.session_state.repo_structure_data = repo_structure
            
            if files:
                options = {
                    "show_function_calls": show_function_calls,
                    "show_imports": show_imports,
                    "show_file_links": show_file_links,
                    "show_folder_structure": show_folder_structure,
                    "min_connections": min_connections
                }
                
                with st.spinner("🔍 Analyzing dependencies..."):
                    graph, file_dependencies = analyzer.create_dependency_graph(files, repo_structure, options)
                    st.session_state.graph_data = graph
                    st.session_state.file_dependencies_data = file_dependencies
                
                st.session_state.repo_analyzed = True
                st.success("✅ Repository analysis completed!")

# Display results if repository has been analyzed
if st.session_state.repo_analyzed and st.session_state.graph_data is not None:
    
    # Check if visualization needs to be updated due to filter changes
    current_options = {
        "show_function_calls": show_function_calls,
        "show_imports": show_imports, 
        "show_file_links": show_file_links,
        "show_folder_structure": show_folder_structure,
        "min_connections": min_connections
    }
    
    # Update visualization if filters changed
    html_content, graph, file_dependencies = update_visualization()
    if html_content:
        st.subheader("🌐 Interactive Repository Graph")
        st.components.v1.html(html_content, height=850)
        
        create_statistics_dashboard(graph, file_dependencies)
    else:
        # Use cached data
        with st.spinner("🎨 Creating visualization..."):
            html_content = create_enhanced_visualization(st.session_state.graph_data, layout_type)
        
        st.subheader("🌐 Interactive Repository Graph")
        st.components.v1.html(html_content, height=850)
        
        create_statistics_dashboard(st.session_state.graph_data, st.session_state.file_dependencies_data)
    
    # Repository Summary and Contribution Report
    if st.session_state.repo_summary:
        st.subheader("📋 Repository Summary")
        st.write(st.session_state.repo_summary)
    
    if st.session_state.contribution_report:
        st.subheader("🚀 Contribution Opportunities")
        st.write(st.session_state.contribution_report)
    
    # File Summaries (only show first few to avoid clutter)
    if st.session_state.files_data and st.session_state.analyzer:
        with st.expander("📄 File Summaries (Click to expand)"):
            for i, file in enumerate(st.session_state.files_data[:20]):  # Limit to first 20 files
                if file["content"] and file["path"] not in st.session_state.file_summaries:
                    summary = st.session_state.analyzer.summarize_file(file["path"], file["content"])
                    st.session_state.file_summaries[file["path"]] = summary
                
                if file["path"] in st.session_state.file_summaries:
                    st.write(f"**{file['path']}**: {st.session_state.file_summaries[file['path']]}")
            
            if len(st.session_state.files_data) > 20:
                st.info(f"Showing first 20 files. Total files: {len(st.session_state.files_data)}")

# Q&A Section - Always available if repository is analyzed
if st.session_state.repo_analyzed and st.session_state.analyzer:
    st.subheader("❓ Ask Questions About the Repository")
    st.markdown("You can ask questions about the code, request explanations, or ask for help with errors/issues.")
    
    # Question input
    question = st.text_input(
        "Enter your question:", 
        placeholder="e.g., 'How does the authentication work?', 'Fix this error: ModuleNotFoundError', 'Explain the main function'"
    )
    
    # Submit button for questions
    if st.button("💡 Get Answer") and question:
        with st.spinner("🔍 Searching through code and generating answer..."):
            answer = st.session_state.analyzer.answer_question(question)
            
            st.subheader("🤖 Answer")
            st.write(answer)
            
            # Add to chat history if you want to implement that
            if 'qa_history' not in st.session_state:
                st.session_state.qa_history = []
            
            st.session_state.qa_history.append({
                'question': question,
                'answer': answer
            })
    
    # Display recent Q&A history
    if 'qa_history' in st.session_state and st.session_state.qa_history:
        with st.expander("📜 Recent Questions & Answers"):
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
                st.write(f"**Q{len(st.session_state.qa_history)-i}:** {qa['question']}")
                st.write(f"**A:** {qa['answer'][:200]}{'...' if len(qa['answer']) > 200 else ''}")
                st.write("---")

elif not st.session_state.repo_analyzed:
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

else:
    st.warning("⚠️ Repository analysis incomplete. Please check your API keys and try again.")

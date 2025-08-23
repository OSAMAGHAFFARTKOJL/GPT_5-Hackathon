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

from core.analyzer import AdvancedDependencyAnalyzer, create_analyzer  # Added create_analyzer import
from core.visualization import create_enhanced_visualization, create_statistics_dashboard

# page configuration
st.set_page_config(
    page_title="Code Compass", 
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="expanded"
)

# Enhanced app title and description
st.title("🧭 Code Compass")
st.markdown("**Next-Generation Open-Source Contribution Helper**")
st.markdown("""
🚀 Understand, visualize, and contribute to GitHub repositories with AI assistance. 
Analyze dependencies, get intelligent explanations, find contribution opportunities, and solve coding problems with **GPT-5**.
""")

# Initialize session state with new GPT-5 related states
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

# New GPT-5 specific session states
if 'code_insights' not in st.session_state:
    st.session_state.code_insights = None
if 'ai_capabilities_enabled' not in st.session_state:
    st.session_state.ai_capabilities_enabled = False
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'enhanced_analysis' not in st.session_state:
    st.session_state.enhanced_analysis = None

# Enhanced sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Repository input
    repo_url = st.text_input(
        "GitHub Repository URL", 
        placeholder="https://github.com/username/repository",
        help="Enter any public GitHub repository URL for analysis"
    )
    
    # API Keys section with improved UI
    st.subheader("🔑 API Keys")
    
    # GitHub token (unchanged)
    github_token = st.text_input(
        "GitHub Token (Optional)", 
        type="password", 
        help="For higher rate limits and private repositories"
    )
    
    # Updated for OpenAI GPT-5
    openai_api_key = st.text_input(
        "OpenAI API Key (GPT-5)", 
        type="password", 
        help="Required for AI-powered analysis, Q&A, and code insights"
    )
    
    # Google API key (unchanged but with enhanced help)
    google_api_key = st.text_input(
        "Google API Key", 
        type="password", 
        help="Required for embeddings and semantic search (RAG functionality)"
    )
    
    # API status indicators
    if openai_api_key:
        st.success("🤖 GPT-5 Enabled")
    else:
        st.warning("⚠️ GPT-5 Disabled - API key required")
    
    if google_api_key:
        st.success("🔍 Vector Search Enabled")
    else:
        st.warning("⚠️ Vector Search Limited - API key required")
    
    # Analysis options
    st.subheader("🧠 Analysis Options")
    enable_ai_insights = st.checkbox(
        "Enable Advanced AI Insights", 
        value=True, 
        help="Generate detailed code insights and recommendations using GPT-5"
    )
    
    enable_comprehensive_qa = st.checkbox(
        "Enable Enhanced Q&A", 
        value=True, 
        help="Advanced question answering with context awareness"
    )
    
    st.header("🎨 Visualization Options")
    layout_type = st.selectbox(
        "Graph Layout", 
        ["spring", "kamada_kawai", "circular", "shell", "random"],
        help="Choose the layout algorithm for the dependency graph"
    )
    
    # Visualization filters
    show_function_calls = st.checkbox("Show Function Calls", value=True)
    show_imports = st.checkbox("Show Imports/Requires", value=True)
    show_file_links = st.checkbox("Show File Links (CSS/JS/Images)", value=True)
    show_folder_structure = st.checkbox("Show Folder Relationships", value=True)
    
    min_connections = st.slider("Minimum Connections to Show", 0, 10, 1)
    
    # Enhanced analyze button
    analyze_button = st.button("🚀 Analyze Repository", type="primary", use_container_width=True)

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

def display_api_status():
    """Display current API status and capabilities"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.analyzer and st.session_state.analyzer.openai_client:
            st.success("🤖 GPT-5: Connected")
        else:
            st.error("❌ GPT-5: Not Available")
    
    with col2:
        if st.session_state.analyzer and st.session_state.analyzer.embedding_model:
            st.success("🔍 Embeddings: Connected")
        else:
            st.error("❌ Embeddings: Not Available")
    
    with col3:
        if st.session_state.repo_analyzed:
            st.success("📊 Analysis: Complete")
        else:
            st.info("⏳ Analysis: Pending")

# Main analysis logic with enhanced GPT-5 integration
if analyze_button and repo_url:
    # Reset analysis state
    st.session_state.repo_analyzed = False
    st.session_state.ai_capabilities_enabled = False
    
    # Initialize analyzer with updated API keys
    with st.spinner("🔧 Initializing GPT-5 powered analyzer..."):
        analyzer = create_analyzer(openai_api_key, google_api_key)  # Use the new create_analyzer function
        st.session_state.analyzer = analyzer
    
    if not analyzer:
        st.error("❌ Failed to initialize analyzer. Please check your API keys.")
        st.stop()
    
    # Display current API status
    st.subheader("🔌 API Connection Status")
    display_api_status()
    
    # Extract repository information
    username, repo_name = analyzer.extract_repo_info(repo_url)
    
    if not username or not repo_name:
        st.error("❌ Invalid GitHub repository URL")
        st.stop()
    
    # Repository processing
    try:
        with st.spinner("📦 Cloning and processing repository..."):
            st.session_state.repo_path = analyzer.clone_repo(repo_url)
            vectorstore_built = analyzer.build_vectorstore(st.session_state.repo_path)
            if vectorstore_built:
                st.session_state.vectorstore = True
                st.success("✅ Vector store built successfully")
        
        if not st.session_state.vectorstore:
            st.error("❌ Failed to build vector store. Some AI features may be limited.")
        
        # Enhanced AI-powered analysis
        if st.session_state.vectorstore and analyzer.gpt5_handler:
            with st.spinner("🧠 Generating AI-powered insights..."):
                # Generate contribution report
                st.session_state.contribution_report = analyzer.generate_contribution_report()
                
                # Generate repository summary
                if st.session_state.contribution_report:
                    st.session_state.repo_summary = analyzer.summarize_repo(st.session_state.contribution_report)
                
                # Generate advanced code insights if enabled
                if enable_ai_insights:
                    repo_context = f"Contribution Report:\n{st.session_state.contribution_report}\n\nRepository Summary:\n{st.session_state.repo_summary}"
                    st.session_state.code_insights = analyzer.generate_code_insights(repo_context)
                
                st.session_state.ai_capabilities_enabled = True
                st.success("✅ AI analysis completed")
        
        # Repository content analysis
        with st.spinner("📦 Fetching repository files for dependency graph..."):
            files, repo_structure = analyzer.get_repo_contents(username, repo_name, github_token)
            st.session_state.files_data = files
            st.session_state.repo_structure_data = repo_structure
        
        if not files:
            st.warning("⚠️ No files found in repository or access denied")
            st.stop()
        
        # Dependency analysis
        options = {
            "show_function_calls": show_function_calls,
            "show_imports": show_imports,
            "show_file_links": show_file_links,
            "show_folder_structure": show_folder_structure,
            "min_connections": min_connections
        }
        
        with st.spinner("🔍 Analyzing dependencies and creating visualization..."):
            graph, file_dependencies = analyzer.create_dependency_graph(files, repo_structure, options)
            st.session_state.graph_data = graph
            st.session_state.file_dependencies_data = file_dependencies
        
        st.session_state.repo_analyzed = True
        
        # Success message with stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Analyzed", len(files))
        with col2:
            st.metric("Dependencies Found", len(file_dependencies) if file_dependencies else 0)
        with col3:
            st.metric("Graph Nodes", graph.number_of_nodes() if graph else 0)
        with col4:
            st.metric("Graph Edges", graph.number_of_edges() if graph else 0)
        
        st.success("🎉 Repository analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("Please check the repository URL and your API keys, then try again.")

# Enhanced results display
if st.session_state.repo_analyzed and st.session_state.graph_data is not None:
    
    # Tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🧠 AI Insights", "📋 Analysis Results", "❓ Q&A Assistant"])
    
    with tab1:
        st.subheader("🌐 Interactive Repository Dependency Graph")
        
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
            st.components.v1.html(html_content, height=850)
            create_statistics_dashboard(graph, file_dependencies)
        else:
            # Use cached data
            with st.spinner("🎨 Creating visualization..."):
                html_content = create_enhanced_visualization(st.session_state.graph_data, layout_type)
            
            st.components.v1.html(html_content, height=850)
            create_statistics_dashboard(st.session_state.graph_data, st.session_state.file_dependencies_data)
    
    with tab2:
        st.subheader("🧠 GPT-5 Powered Code Insights")
        
        if st.session_state.ai_capabilities_enabled:
            # Repository Summary
            if st.session_state.repo_summary:
                st.subheader("📋 Repository Overview")
                st.markdown(st.session_state.repo_summary)
            
            # Contribution Opportunities
            if st.session_state.contribution_report:
                st.subheader("🚀 Contribution Opportunities")
                st.markdown(st.session_state.contribution_report)
            
            # Advanced Code Insights
            if st.session_state.code_insights:
                st.subheader("🔍 Advanced Code Analysis")
                st.markdown(st.session_state.code_insights)
            
        else:
            st.warning("🔑 AI insights require valid OpenAI API key. Please add your API key and re-analyze.")
            st.info("💡 With GPT-5 insights, you'll get:")
            st.markdown("""
            - **Architecture Analysis** - Deep dive into code structure and patterns
            - **Contribution Opportunities** - Personalized suggestions for contributing
            - **Code Quality Assessment** - Professional code review insights
            - **Performance Recommendations** - Optimization suggestions
            - **Security Analysis** - Potential security improvements
            """)
    
    with tab3:
        st.subheader("📊 Analysis Results")
        
        # File Summaries with enhanced presentation
        if st.session_state.files_data and st.session_state.analyzer:
            with st.expander("📄 File Summaries (AI-Generated)", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("📝 Files are automatically summarized using GPT-5 for better understanding")
                with col2:
                    if st.button("🔄 Regenerate All Summaries"):
                        st.session_state.file_summaries = {}
                        st.rerun()
                
                # Generate summaries for files
                files_to_summarize = st.session_state.files_data[:25]  # Limit to first 25 files
                progress_bar = st.progress(0)
                
                for i, file in enumerate(files_to_summarize):
                    if file["content"] and len(file["content"].strip()) > 50:  # Skip very small files
                        if file["path"] not in st.session_state.file_summaries:
                            if st.session_state.analyzer and st.session_state.ai_capabilities_enabled:
                                try:
                                    summary = st.session_state.analyzer.summarize_file(file["path"], file["content"])
                                    st.session_state.file_summaries[file["path"]] = summary
                                except Exception as e:
                                    st.session_state.file_summaries[file["path"]] = f"Error generating summary: {str(e)}"
                            else:
                                st.session_state.file_summaries[file["path"]] = "AI summary requires OpenAI API key"
                        
                        # Display summary
                        if file["path"] in st.session_state.file_summaries:
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**📁 {file['path']}**")
                                    st.write(st.session_state.file_summaries[file["path"]])
                                with col2:
                                    st.caption(f"Lines: {len(file['content'].splitlines())}")
                                    st.caption(f"Size: {len(file['content'])} chars")
                                st.divider()
                    
                    progress_bar.progress((i + 1) / len(files_to_summarize))
                
                if len(st.session_state.files_data) > 25:
                    st.info(f"📊 Showing first 25 files. Total files in repository: {len(st.session_state.files_data)}")
        
        # Additional statistics and insights
        if st.session_state.graph_data:
            st.subheader("📈 Repository Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Code Complexity", 
                    st.session_state.graph_data.number_of_edges(),
                    delta="connections",
                    help="Number of dependencies and relationships"
                )
            
            with col2:
                if st.session_state.file_dependencies_data:
                    avg_deps = sum(len(deps) for deps in st.session_state.file_dependencies_data.values()) / len(st.session_state.file_dependencies_data)
                    st.metric(
                        "Avg Dependencies", 
                        f"{avg_deps:.1f}",
                        help="Average dependencies per file"
                    )
            
            with col3:
                if st.session_state.graph_data.number_of_nodes() > 0:
                    density = st.session_state.graph_data.number_of_edges() / st.session_state.graph_data.number_of_nodes()
                    st.metric(
                        "Graph Density", 
                        f"{density:.2f}",
                        help="Connectivity ratio of the dependency graph"
                    )
    
    with tab4:
        st.subheader("🤖 GPT-5 Powered Q&A Assistant")
        
        if st.session_state.ai_capabilities_enabled:
            st.markdown("""
            **💡 Ask anything about this repository:**
            - How does specific functionality work?
            - Help with error messages or bugs
            - Setup and configuration questions
            - Code explanation and architecture
            - Best practices and improvements
            """)
            
            # Enhanced question input with examples
            question_examples = [
                "How does the authentication system work?",
                "Fix this error: ModuleNotFoundError: No module named 'requests'",
                "What is the main entry point of this application?",
                "How do I set up the development environment?",
                "Explain the database connection logic",
                "What are the main components and how do they interact?",
                "How can I contribute to this project?",
                "What testing frameworks are used here?"
            ]
            
            # Question input with improved UI
            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_area(
                    "Enter your question:", 
                    height=100,
                    placeholder="Type your question here... (e.g., 'How does the authentication work?' or 'Fix this error: ImportError')"
                )
            
            with col2:
                st.markdown("**💡 Example Questions:**")
                for i, example in enumerate(question_examples[:4]):
                    if st.button(f"📝 {example[:30]}...", key=f"example_{i}", help=example):
                        question = example
                        st.rerun()
            
            # Enhanced submit section
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                ask_button = st.button("🧠 Ask GPT-5", type="primary", use_container_width=True)
            with col2:
                if st.button("🔄 Clear History"):
                    st.session_state.qa_history = []
                    st.rerun()
            with col3:
                enhanced_qa = st.checkbox("🔍 Enhanced Mode", value=enable_comprehensive_qa, 
                                        help="More detailed analysis with additional context")
            
            # Process question
            if ask_button and question.strip():
                with st.spinner("🤔 GPT-5 is analyzing the repository and generating answer..."):
                    try:
                        if enhanced_qa:
                            # Use enhanced Q&A with additional context
                            answer = st.session_state.analyzer.answer_question(question)
                        else:
                            # Standard Q&A
                            answer = st.session_state.analyzer.answer_question(question)
                        
                        # Display answer with enhanced formatting
                        st.subheader("🤖 GPT-5 Answer")
                        st.markdown(answer)
                        
                        # Add to history
                        st.session_state.qa_history.append({
                            'question': question,
                            'answer': answer,
                            'timestamp': st.time()
                        })
                        
                        # Auto-scroll to answer (visual feedback)
                        st.success("✅ Answer generated! Check above for the response.")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
                        st.info("Please check your API keys and try again.")
            
            # Enhanced Q&A history
            if st.session_state.qa_history:
                st.subheader("📜 Q&A History")
                
                # Show recent questions in an organized way
                for i, qa in enumerate(reversed(st.session_state.qa_history[-10:])):  # Show last 10
                    with st.expander(f"❓ Q{len(st.session_state.qa_history)-i}: {qa['question'][:60]}..."):
                        st.markdown("**Question:**")
                        st.write(qa['question'])
                        st.markdown("**Answer:**")
                        st.markdown(qa['answer'])
                        st.caption(f"⏰ Asked at: {qa.get('timestamp', 'Unknown time')}")
        
        else:
            st.warning("🔑 Q&A Assistant requires valid OpenAI API key")
            st.info("💡 With GPT-5 Q&A, you can:")
            st.markdown("""
            - **Get instant answers** about any part of the codebase
            - **Solve errors and bugs** with step-by-step guidance  
            - **Understand complex code** with detailed explanations
            - **Learn best practices** for the specific technology stack
            - **Get setup help** for development environment
            """)

elif not st.session_state.repo_analyzed:
    # Enhanced welcome screen
    st.info("🎯 **Ready to analyze your first repository?** Enter a GitHub URL above and click 'Analyze Repository'!")
    
    # Feature showcase with better presentation
    st.subheader("🚀 Code Compass Features")
    
    # Features in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 **GPT-5 Powered Analysis**
        - **Smart Repository Analysis**: Deep understanding of code structure
        - **AI-Powered Q&A**: Ask questions and get intelligent answers
        - **Code Insights**: Professional-level code review and recommendations
        - **Error Solving**: Step-by-step debugging assistance
        """)
        
        st.markdown("""
        ### 📊 **Advanced Visualization**
        - **Interactive Dependency Graphs**: Explore file relationships
        - **Real-time Filters**: Customize views without re-analysis
        - **Multiple Layouts**: Choose the best visualization style
        - **Detailed Statistics**: Comprehensive repository metrics
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 **Contribution Guidance**
        - **Personalized Suggestions**: Find ways to contribute
        - **Quick Wins**: Identify beginner-friendly tasks
        - **Architecture Understanding**: Learn the codebase structure
        - **Best Practices**: Get recommendations for improvements
        """)
        
        st.markdown("""
        ### 🔍 **Smart Search & Analysis**
        - **Semantic Search**: Find relevant code using natural language
        - **File Summaries**: AI-generated explanations of each file
        - **Dependency Tracking**: Understand code relationships
        - **Multi-language Support**: Works with various programming languages
        """)
    
    # Example repositories
    st.subheader("💡 Try These Popular Repositories")
    
    examples = [
        ("React", "https://github.com/facebook/react", "Popular JavaScript library for building user interfaces"),
        ("FastAPI", "https://github.com/tiangolo/fastapi", "Modern, fast web framework for building APIs with Python"),
        ("Vue.js", "https://github.com/vuejs/vue", "Progressive JavaScript framework for building UIs"),
        ("Django", "https://github.com/django/django", "High-level Python web framework"),
    ]
    
    cols = st.columns(2)
    for i, (name, url, description) in enumerate(examples):
        with cols[i % 2]:
            st.markdown(f"""
            **{name}**  
            {description}  
            `{url}`
            """)
    
    # Requirements and tips
    st.subheader("📋 Getting Started")
    
    req_col1, req_col2 = st.columns(2)
    
    with req_col1:
        st.markdown("""
        **🔑 Required API Keys:**
        - **OpenAI API Key**: For GPT-5 powered features
        - **Google API Key**: For vector search and embeddings
        - **GitHub Token**: Optional, for higher rate limits
        """)
    
    with req_col2:
        st.markdown("""
        **💡 Pro Tips:**
        - Start with smaller repositories for faster analysis
        - Enable all AI features for comprehensive insights
        - Use the Q&A feature to understand complex code
        - Check the contribution report for ways to help
        """)

else:
    st.warning("⚠️ Repository analysis incomplete. Please check your API keys and try again.")
    
    # Troubleshooting help
    with st.expander("🔧 Troubleshooting Help"):
        st.markdown("""
        **Common Issues:**
        
        1. **API Key Problems:**
           - Ensure your OpenAI API key is valid and has GPT-5 access
           - Check that your Google API key has the required permissions
           - Verify API keys don't have leading/trailing spaces
        
        2. **Repository Issues:**
           - Make sure the GitHub URL is correct and public
           - Some repositories may be too large for analysis
           - Private repositories require a GitHub token
        
        3. **Network Issues:**
           - Check your internet connection
           - Some corporate firewalls may block API calls
           - Try again in a few minutes if services are busy
        
        **Need Help?**
        - Check the status indicators in the sidebar
        - Try with a smaller, simpler repository first
        - Ensure all required API keys are provided
        """)

# Footer with version info and credits
st.markdown("---")
st.markdown(
    "🧭 **Code Compass**  "
    "Built with ❤️ for the open source community by Deep Minds Team"
)
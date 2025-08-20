import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Code Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("🧭 Code Compass")
    st.markdown("### AI-Powered Open Source Navigation")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a feature:",
            ["🗺️ Map Repository", "🔍 Query Codebase", "📊 Analyze Code", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        
        # GitHub token input
        github_token = st.text_input(
            "GitHub Token (Optional)",
            type="password",
            help="For private repositories or higher rate limits"
        )
        
        st.session_state["github_token"] = github_token
    
    # Main content based on selected page
    if page == "🗺️ Map Repository":
        map_repository_page()
    elif page == "🔍 Query Codebase":
        query_codebase_page()
    elif page == "📊 Analyze Code":
        analyze_code_page()
    elif page == "ℹ️ About":
        about_page()

def map_repository_page():
    st.header("🗺️ Map Repository")
    st.markdown("Generate an interactive knowledge graph of a GitHub repository")
    
    # Repository input
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        help="Enter a public GitHub repository URL"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        map_button = st.button("🗺️ Map Repository", type="primary")
    
    if map_button and repo_url:
        with st.spinner("Mapping repository... This may take a few minutes."):
            try:
                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/map-repo",
                    json={
                        "repo_url": repo_url,
                        "github_token": st.session_state.get("github_token")
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()["data"]
                    
                    # Display results
                    display_mapping_results(data)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

def display_mapping_results(data: Dict[str, Any]):
    st.success("✅ Repository mapped successfully!")
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    kg_summary = data.get("knowledge_graph_summary", {})
    code_summary = data.get("code_structure_summary", {})
    
    with col1:
        st.metric("Files", code_summary.get("files", 0))
    with col2:
        st.metric("Functions", code_summary.get("functions", 0))
    with col3:
        st.metric("Classes", code_summary.get("classes", 0))
    with col4:
        st.metric("Graph Nodes", kg_summary.get("nodes", 0))
    
    # Display visualization
    st.subheader("📊 Knowledge Graph Visualization")
    
    visualization_data = data.get("visualization_data", {})
    if visualization_data and visualization_data.get("nodes"):
        display_knowledge_graph(visualization_data)
    else:
        st.info("No visualization data available")
    
    # Display sample files
    sample_files = code_summary.get("sample_files", [])
    if sample_files:
        st.subheader("📁 Sample Files")
        for file_path in sample_files:
            st.code(file_path, language="text")

def display_knowledge_graph(viz_data: Dict[str, Any]):
    """Display an interactive knowledge graph using Plotly"""
    nodes = viz_data.get("nodes", [])
    edges = viz_data.get("edges", [])
    
    if not nodes:
        st.info("No graph data to display")
        return
    
    # Create network graph using Plotly
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge.get("from_x", 0), edge.get("to_x", 1)],
            y=[edge.get("from_y", 0), edge.get("to_y", 1)],
            mode='lines',
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes
    node_x = [i % 10 for i in range(len(nodes))]  # Simple layout
    node_y = [i // 10 for i in range(len(nodes))]
    node_text = [node.get("label", node.get("id", "")) for node in nodes]
    node_colors = [get_node_color(node.get("type", "unknown")) for node in nodes]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=12,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=node_text,
        textposition="middle center",
        hoverinfo='text',
        hovertext=[f"{node.get('type', 'unknown')}: {node.get('label', '')}" for node in nodes],
        showlegend=False
    ))
    
    fig.update_layout(
        title="Repository Knowledge Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Interactive graph - hover over nodes for details",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            font=dict(size=12, color="gray")
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_node_color(node_type: str) -> str:
    """Get color for node based on type"""
    color_map = {
        "file": "#FF6B6B",
        "class": "#4ECDC4", 
        "function": "#45B7D1",
        "unknown": "#96CEB4"
    }
    return color_map.get(node_type, "#96CEB4")

def query_codebase_page():
    st.header("🔍 Query Codebase")
    st.markdown("Ask natural language questions about a repository")
    
    # Repository input
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        key="query_repo_url"
    )
    
    # Query input
    query = st.text_area(
        "Your Question",
        placeholder="e.g., How does authentication work in this project?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        query_button = st.button("🔍 Ask Question", type="primary")
    
    if query_button and repo_url and query:
        with st.spinner("Analyzing codebase and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={
                        "repo_url": repo_url,
                        "query": query
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()["data"]
                    display_query_results(data)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

def display_query_results(data: Dict[str, Any]):
    st.success("✅ Analysis complete!")
    
    # Display summary
    summary = data.get("summary", "")
    if summary:
        st.subheader("📝 Summary")
        st.info(summary)
    
    # Display code insights
    insights = data.get("code_insights", [])
    if insights:
        st.subheader("🔍 Code Insights")
        for insight in insights:
            with st.expander(f"{insight.get('type', 'Insight')}: {insight.get('description', '')[:50]}..."):
                st.write(f"**Type:** {insight.get('type', 'Unknown')}")
                st.write(f"**Description:** {insight.get('description', 'No description')}")
                st.write(f"**Location:** {insight.get('location', 'Unknown')}")
                st.write(f"**Relevance:** {insight.get('relevance_score', 0):.3f}")
    
    # Display suggested issues
    issues = data.get("suggested_issues", [])
    if issues:
        st.subheader("🎯 Beginner-Friendly Issues")
        for issue in issues:
            st.markdown(f"- [{issue.get('title', 'Issue')}]({issue.get('url', '#')})")

def analyze_code_page():
    st.header("📊 Analyze Code")
    st.markdown("Detect code smells and get contribution suggestions")
    
    # Repository input
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        key="analyze_repo_url"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("📊 Analyze", type="primary")
    
    if analyze_button and repo_url:
        with st.spinner("Analyzing code quality and detecting issues..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={"repo_url": repo_url},
                    timeout=300
                )
                
                if response.status_code == 200:
                    data = response.json()["data"]
                    display_analysis_results(data)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

def display_analysis_results(data: Dict[str, Any]):
    st.success("✅ Code analysis complete!")
    
    # Display quality score
    quality_score = data.get("quality_score", 0)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.metric("Code Quality Score", f"{quality_score:.1f}/100")
        
        # Quality score gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Quality Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': get_quality_color(quality_score)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display contribution suggestions
    suggestions = data.get("contribution_suggestions", [])
    if suggestions:
        st.subheader("💡 Contribution Suggestions")
        
        for suggestion in suggestions:
            difficulty = suggestion.get("difficulty", "unknown")
            color = get_difficulty_color(difficulty)
            
            with st.expander(f"{suggestion.get('type', 'Suggestion')}: {suggestion.get('description', '')[:60]}..."):
                st.markdown(f"**Difficulty:** <span style='color:{color}'>{difficulty.title()}</span>", unsafe_allow_html=True)
                st.write(f"**Description:** {suggestion.get('description', 'No description')}")
                st.write(f"**Location:** {suggestion.get('location', 'Unknown')}")
                st.write(f"**Estimated Effort:** {suggestion.get('estimated_effort', 'Unknown')}")
                st.write(f"**Impact:** {suggestion.get('impact', 'Unknown')}")
    
    # Display code smells
    code_smells = data.get("code_smells", [])
    if code_smells:
        st.subheader("🦨 Code Smells Detected")
        
        # Create DataFrame for better display
        df_smells = pd.DataFrame(code_smells)
        if not df_smells.empty:
            st.dataframe(df_smells, use_container_width=True)

def get_quality_color(score: float) -> str:
    """Get color based on quality score"""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "yellow"
    else:
        return "red"

def get_difficulty_color(difficulty: str) -> str:
    """Get color based on difficulty level"""
    color_map = {
        "beginner": "green",
        "intermediate": "orange", 
        "advanced": "red"
    }
    return color_map.get(difficulty.lower(), "gray")

def about_page():
    st.header("ℹ️ About Code Compass")
    
    st.markdown("""
    ## 🧭 What is Code Compass?
    
    Code Compass is an AI-powered tool designed to help students explore and contribute to open-source GitHub repositories. It uses advanced multi-agent systems to:
    
    - **🗺️ Map Codebases**: Generate interactive knowledge graphs of repository structure
    - **🔍 Answer Questions**: Use natural language to query code and get intelligent answers  
    - **📊 Analyze Quality**: Detect code smells and suggest improvement opportunities
    - **💡 Guide Contributions**: Find beginner-friendly issues and contribution opportunities
    
    ## 🏗️ Architecture
    
    The system consists of three specialized AI agents:
    
    ### 📍 Mapper Agent
    - Clones and parses repository code
    - Builds knowledge graphs using NetworkX
    - Creates interactive visualizations
    
    ### 🧭 Navigator Agent  
    - Processes natural language queries
    - Searches code using semantic embeddings
    - Finds relevant GitHub issues
    
    ### 📊 Analyst Agent
    - Runs static code analysis
    - Detects code smells and quality issues
    - Predicts bug-prone areas using ML
    
    ## 🛠️ Technology Stack
    
    - **Backend**: FastAPI, Python
    - **AI/ML**: LangGraph, Sentence Transformers, scikit-learn
    - **Graph Processing**: NetworkX, PyVis
    - **Frontend**: Streamlit
    - **Code Analysis**: Tree-sitter, Pylint
    - **Version Control**: GitHub API integration
    
    ## 🚀 Getting Started
    
    1. Enter a GitHub repository URL
    2. Optionally provide a GitHub token for private repos
    3. Choose your desired action:
       - Map the repository structure
       - Ask questions about the code
       - Analyze code quality and get contribution suggestions
    
    ## 🎯 Perfect for Students
    
    Code Compass is specifically designed to help students:
    - Understand complex open-source projects
    - Find good first issues to work on
    - Learn best practices through code analysis
    - Navigate large codebases with confidence
    
    ---
    
    **Built with ❤️ to make open-source accessible for students!**
    """)

if __name__ == "__main__":
    main()
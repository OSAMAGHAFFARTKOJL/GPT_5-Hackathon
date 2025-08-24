import streamlit as st
import os
from utils.visualization import create_enhanced_visualization, create_statistics_dashboard

def update_visualization(config):
    """Update visualization when filters change without re-analyzing the repository"""
    if st.session_state.repo_analyzed and st.session_state.files_data is not None:
        options = {
            "show_function_calls": config['show_function_calls'],
            "show_imports": config['show_imports'],
            "show_file_links": config['show_file_links'],
            "show_folder_structure": config['show_folder_structure'],
            "min_connections": config['min_connections']
        }
        
        with st.spinner("🎨 Updating visualization..."):
            graph, file_dependencies = st.session_state.analyzer.create_dependency_graph(
                st.session_state.files_data, 
                st.session_state.repo_structure_data, 
                options
            )
            st.session_state.graph_data = graph
            st.session_state.file_dependencies_data = file_dependencies
            
            html_content = create_enhanced_visualization(graph, config['layout_type'])
            return html_content, graph, file_dependencies
    return None, None, None

def render_dashboard(config):
    """Render the main dashboard with visualization and reports"""
    # Display results if repository has been analyzed
    if st.session_state.repo_analyzed and st.session_state.graph_data is not None:
        
        # Check if visualization needs to be updated due to filter changes
        html_content, graph, file_dependencies = update_visualization(config)
        if html_content:
            st.subheader("🌐 Interactive Repository Graph")
            st.components.v1.html(html_content, height=850)
            
            create_statistics_dashboard(graph, file_dependencies)
        else:
            # Use cached data
            with st.spinner("🎨 Creating visualization..."):
                html_content = create_enhanced_visualization(st.session_state.graph_data, config['layout_type'])
            
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
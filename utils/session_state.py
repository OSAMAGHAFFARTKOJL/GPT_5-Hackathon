import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'vectorstore': None,
        'repo_path': None,
        'contribution_report': None,
        'file_summaries': {},
        'repo_summary': None,
        'analyzer': None,
        'files_data': None,
        'repo_structure_data': None,
        'graph_data': None,
        'file_dependencies_data': None,
        'repo_analyzed': False,
        'qa_history': []
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
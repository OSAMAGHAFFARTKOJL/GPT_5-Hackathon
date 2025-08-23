import base64
import os
import re
import ast
import tempfile
import subprocess
import glob
import shutil
from typing import List
from collections import defaultdict

import streamlit as st
from github import Github
from groq import Groq
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# import moved method implementations
from .analyzer_repo import extract_repo_info, get_repo_contents, clone_repo
from .analyzer_text import chunk_text, load_code_files, build_vectorstore
from .analyzer_ai import generate_contribution_report, summarize_repo, summarize_file, answer_question, analyze_dependencies_with_ai
from .analyzer_parse import (
    extract_python_dependencies, extract_javascript_dependencies,
    extract_html_dependencies, extract_css_dependencies, extract_with_regex,
    create_dependency_graph
)

INDEX_DIR = "faiss_index"


class AdvancedDependencyAnalyzer:
    def __init__(self, groq_api_key=None, google_api_key=None):
        self.groq_client = None
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
            except:
                st.warning("Invalid Groq API key")
        
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        else:
            self.embedding_model = None

# attach moved methods to the class (preserve original names and behavior)
AdvancedDependencyAnalyzer.extract_repo_info = extract_repo_info
AdvancedDependencyAnalyzer.get_repo_contents = get_repo_contents
AdvancedDependencyAnalyzer.clone_repo = clone_repo
AdvancedDependencyAnalyzer.chunk_text = chunk_text
AdvancedDependencyAnalyzer.load_code_files = load_code_files
AdvancedDependencyAnalyzer.build_vectorstore = build_vectorstore
AdvancedDependencyAnalyzer.generate_contribution_report = generate_contribution_report
AdvancedDependencyAnalyzer.summarize_repo = summarize_repo
AdvancedDependencyAnalyzer.summarize_file = summarize_file
AdvancedDependencyAnalyzer.answer_question = answer_question
AdvancedDependencyAnalyzer.analyze_dependencies_with_ai = analyze_dependencies_with_ai
AdvancedDependencyAnalyzer.extract_python_dependencies = extract_python_dependencies
AdvancedDependencyAnalyzer.extract_javascript_dependencies = extract_javascript_dependencies
AdvancedDependencyAnalyzer.extract_html_dependencies = extract_html_dependencies
AdvancedDependencyAnalyzer.extract_css_dependencies = extract_css_dependencies
AdvancedDependencyAnalyzer.extract_with_regex = extract_with_regex
AdvancedDependencyAnalyzer.create_dependency_graph = create_dependency_graph


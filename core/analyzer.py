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
import openai  # Changed from groq import
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Import moved method implementations
from .analyzer_repo import extract_repo_info, get_repo_contents, clone_repo
from .analyzer_text import chunk_text, load_code_files, build_vectorstore
from .analyzer_ai_gpt import GPT5AnalyticsHandler  # Updated import
from .analyzer_parse import (
    extract_python_dependencies, extract_javascript_dependencies,
    extract_html_dependencies, extract_css_dependencies, extract_with_regex,
    create_dependency_graph
)

INDEX_DIR = "faiss_index"


class AdvancedDependencyAnalyzer:
    """
    Advanced Repository Dependency Analyzer with GPT-5 Integration
    
    This class provides comprehensive repository analysis capabilities including:
    - Dependency extraction and analysis
    - AI-powered insights and Q&A
    - Code summarization and contribution reports
    - Vector-based semantic search
    """
    
    def __init__(self, openai_api_key=None, google_api_key=None):
        """
        Initialize the Advanced Dependency Analyzer
        
        Args:
            openai_api_key (str): OpenAI API key for GPT-5 access
            google_api_key (str): Google API key for embeddings
        """
        # Initialize GPT-5 client (replaced Groq)
        self.openai_client = None
        self.gpt5_handler = None
        
        if openai_api_key:
            try:
                openai.api_key = openai_api_key
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                # Test the connection
                self.openai_client.models.list()
                st.success("✅ OpenAI GPT-5 connection established successfully!")
            except Exception as e:
                st.error(f"❌ Failed to connect to OpenAI API: {str(e)}")
                st.info("Please check your OpenAI API key and ensure you have access to GPT-5")
                self.openai_client = None
        else:
            st.warning("⚠️ OpenAI API key not provided. AI-powered features will be disabled.")
        
        # Initialize Google embeddings (unchanged)
        self.embedding_model = None
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=google_api_key
                )
                st.success("✅ Google embeddings initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize Google embeddings: {str(e)}")
                self.embedding_model = None
        else:
            st.warning("⚠️ Google API key not provided. Vector search capabilities will be limited.")
        
        # Initialize GPT-5 handler with both API keys
        if openai_api_key or google_api_key:
            try:
                self.gpt5_handler = GPT5AnalyticsHandler(openai_api_key, self.embedding_model)
                if self.gpt5_handler.openai_client:
                    st.success("🧠 GPT-5 analytics handler initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize GPT-5 handler: {str(e)}")
                self.gpt5_handler = None

    # AI-powered methods (updated to use GPT-5 handler)
    def generate_contribution_report(self):
        """Generate AI-powered contribution opportunities report"""
        if not self.gpt5_handler:
            return "GPT-5 handler not initialized. Please provide valid API keys."
        return self.gpt5_handler.generate_contribution_report()

    def summarize_repo(self, contribution_report):
        """Generate comprehensive repository summary"""
        if not self.gpt5_handler:
            return "GPT-5 handler not initialized. Please provide valid API keys."
        return self.gpt5_handler.summarize_repo(contribution_report)

    def summarize_file(self, file_path, content):
        """Generate concise file summary"""
        if not self.gpt5_handler:
            return "GPT-5 handler not initialized. Please provide valid API keys."
        return self.gpt5_handler.summarize_file(file_path, content)

    def answer_question(self, question):
        """Enhanced Q&A system using GPT-5 with RAG"""
        if not self.gpt5_handler:
            return "GPT-5 handler not initialized. Please provide valid API keys."
        return self.gpt5_handler.answer_question(question)

    def analyze_dependencies_with_ai(self, content, file_path):
        """AI-powered dependency analysis"""
        if not self.gpt5_handler:
            return {}
        return self.gpt5_handler.analyze_dependencies_with_ai(content, file_path)

    def generate_code_insights(self, repo_context):
        """Generate advanced code insights and recommendations (NEW)"""
        if not self.gpt5_handler:
            return "GPT-5 handler not initialized. Please provide valid API keys."
        return self.gpt5_handler.generate_code_insights(repo_context)

    # Legacy compatibility methods (for backward compatibility)
    def _make_gpt5_call(self, prompt, temperature=0.1, max_tokens=2000):
        """Direct GPT-5 API call with error handling"""
        if not self.openai_client:
            return "OpenAI API key required for GPT-5 functionality."
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # Update to actual model name when available
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling GPT-5: {str(e)}"

    # Enhanced analysis method with GPT-5 insights
    def analyze_repository_comprehensive(self, repo_url_or_path, include_ai_insights=True):
        """
        Comprehensive repository analysis with optional AI insights
        
        Args:
            repo_url_or_path (str): Repository URL or local path
            include_ai_insights (bool): Whether to include AI-powered analysis
            
        Returns:
            dict: Comprehensive analysis results
        """
        results = {
            'basic_analysis': {},
            'ai_insights': {},
            'contribution_report': '',
            'repo_summary': '',
            'error': None
        }
        
        try:
            # Basic analysis (existing functionality)
            st.info("🔍 Performing basic repository analysis...")
            
            # Repository extraction and processing
            if repo_url_or_path.startswith(('http://', 'https://')):
                repo_info = self.extract_repo_info(repo_url_or_path)
                contents = self.get_repo_contents(repo_url_or_path)
            else:
                # Local path analysis
                contents = self.load_code_files(repo_url_or_path)
            
            # Build vector store for semantic search
            if self.embedding_model and contents:
                st.info("🧮 Building vector store for semantic search...")
                vectorstore_result = self.build_vectorstore(contents)
                results['basic_analysis']['vectorstore'] = vectorstore_result
            
            # AI-powered analysis if enabled and available
            if include_ai_insights and self.gpt5_handler:
                st.info("🧠 Generating AI-powered insights...")
                
                # Generate contribution report
                contribution_report = self.generate_contribution_report()
                results['contribution_report'] = contribution_report
                
                # Generate repository summary
                if contribution_report and not contribution_report.startswith("GPT-5 handler"):
                    repo_summary = self.summarize_repo(contribution_report)
                    results['repo_summary'] = repo_summary
                
                # Generate advanced code insights
                repo_context = f"Contribution Report:\n{contribution_report}\n\nRepository Summary:\n{repo_summary}"
                code_insights = self.generate_code_insights(repo_context)
                results['ai_insights']['code_insights'] = code_insights
                
                # Analyze dependencies with AI for key files
                if contents:
                    st.info("🔗 Analyzing dependencies with AI...")
                    ai_dependencies = {}
                    
                    # Analyze up to 10 key files to avoid API limits
                    key_files = list(contents.items())[:10]
                    for file_path, content in key_files:
                        if len(content.strip()) > 100:  # Skip very small files
                            ai_deps = self.analyze_dependencies_with_ai(content, file_path)
                            if ai_deps:  # Only store non-empty results
                                ai_dependencies[file_path] = ai_deps
                    
                    results['ai_insights']['ai_dependencies'] = ai_dependencies
                
                st.success("✅ AI-powered analysis completed!")
            
            results['basic_analysis']['status'] = 'success'
            results['basic_analysis']['files_analyzed'] = len(contents) if contents else 0
            
        except Exception as e:
            error_msg = f"Error during comprehensive analysis: {str(e)}"
            results['error'] = error_msg
            st.error(f"❌ {error_msg}")
        
        return results

    # Enhanced Q&A with context awareness
    def enhanced_qa_session(self, questions_list):
        """
        Process multiple questions with context awareness
        
        Args:
            questions_list (List[str]): List of questions to process
            
        Returns:
            dict: Q&A results with context
        """
        if not self.gpt5_handler:
            return {"error": "GPT-5 handler not available"}
        
        results = {}
        context_summary = ""
        
        for i, question in enumerate(questions_list):
            try:
                # Enhanced question with previous context
                if context_summary and i > 0:
                    enhanced_question = f"Previous context: {context_summary[:500]}...\n\nCurrent question: {question}"
                else:
                    enhanced_question = question
                
                answer = self.answer_question(enhanced_question)
                results[f"q{i+1}"] = {
                    "question": question,
                    "answer": answer,
                    "timestamp": st.timestamp()
                }
                
                # Build context for next questions
                context_summary += f" Q: {question} A: {answer[:200]}..."
                
            except Exception as e:
                results[f"q{i+1}"] = {
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "error": True
                }
        
        return results


# Attach moved methods to the class (preserve original names and behavior)
# Repository methods
AdvancedDependencyAnalyzer.extract_repo_info = extract_repo_info
AdvancedDependencyAnalyzer.get_repo_contents = get_repo_contents
AdvancedDependencyAnalyzer.clone_repo = clone_repo

# Text processing methods
AdvancedDependencyAnalyzer.chunk_text = chunk_text
AdvancedDependencyAnalyzer.load_code_files = load_code_files
AdvancedDependencyAnalyzer.build_vectorstore = build_vectorstore

# Parsing methods
AdvancedDependencyAnalyzer.extract_python_dependencies = extract_python_dependencies
AdvancedDependencyAnalyzer.extract_javascript_dependencies = extract_javascript_dependencies
AdvancedDependencyAnalyzer.extract_html_dependencies = extract_html_dependencies
AdvancedDependencyAnalyzer.extract_css_dependencies = extract_css_dependencies
AdvancedDependencyAnalyzer.extract_with_regex = extract_with_regex
AdvancedDependencyAnalyzer.create_dependency_graph = create_dependency_graph


# Utility function for easy initialization
def create_analyzer(openai_api_key=None, google_api_key=None):
    """
    Convenience function to create and initialize the analyzer
    
    Args:
        openai_api_key (str): OpenAI API key for GPT-5
        google_api_key (str): Google API key for embeddings
        
    Returns:
        AdvancedDependencyAnalyzer: Initialized analyzer instance
    """
    try:
        analyzer = AdvancedDependencyAnalyzer(openai_api_key, google_api_key)
        
        # Validation and status reporting
        if analyzer.openai_client:
            st.info("🟢 GPT-5 capabilities: ENABLED")
        else:
            st.warning("🟡 GPT-5 capabilities: DISABLED (API key required)")
        
        if analyzer.embedding_model:
            st.info("🟢 Vector search capabilities: ENABLED")
        else:
            st.warning("🟡 Vector search capabilities: DISABLED (Google API key required)")
        
        return analyzer
        
    except Exception as e:
        st.error(f"❌ Failed to create analyzer: {str(e)}")
        return None


# Enhanced usage example
def demo_usage():
    """
    Demonstration of the enhanced analyzer capabilities
    """
    st.markdown("### 🚀 GPT-5 Enhanced Analyzer Demo")
    
    # Initialize with API keys from Streamlit secrets or environment
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    google_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    analyzer = create_analyzer(openai_key, google_key)
    
    if analyzer:
        repo_url = st.text_input("Repository URL", "https://github.com/microsoft/vscode")
        
        if st.button("🔍 Analyze Repository"):
            with st.spinner("Analyzing repository..."):
                results = analyzer.analyze_repository_comprehensive(repo_url, include_ai_insights=True)
                
                if results.get('error'):
                    st.error(results['error'])
                else:
                    st.success("✅ Analysis completed!")
                    
                    # Display results
                    if results.get('contribution_report'):
                        st.markdown("## 🎯 Contribution Report")
                        st.markdown(results['contribution_report'])
                    
                    if results.get('repo_summary'):
                        st.markdown("## 📋 Repository Summary")
                        st.markdown(results['repo_summary'])
                    
                    if results.get('ai_insights', {}).get('code_insights'):
                        st.markdown("## 🧠 Code Insights")
                        st.markdown(results['ai_insights']['code_insights'])


if __name__ == "__main__":
    demo_usage()
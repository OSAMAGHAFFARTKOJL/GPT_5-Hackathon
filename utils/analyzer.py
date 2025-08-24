import streamlit as st
import requests
import networkx as nx
import base64
import os
import json
import re
import ast
import tempfile
import subprocess
import glob
import shutil
from github import Github
from collections import defaultdict
from urllib.parse import urlparse, urljoin
from openai import OpenAI
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from typing import List

INDEX_DIR = "faiss_index"

class AdvancedDependencyAnalyzer:
    def __init__(self, aiml_api_key=None, google_api_key=None):
        self.openai_client = None
        if aiml_api_key:
            try:
                self.openai_client = OpenAI(
                    base_url='https://api.aimlapi.com/v1',
                    api_key=aiml_api_key
                )
            except:
                st.warning("Invalid AIML API key")
        
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        else:
            self.embedding_model = None
    
    def extract_repo_info(self, url):
        pattern = r"github\.com\/([\w.-]+)\/([\w.-]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def get_repo_contents(self, username, repo_name, github_token=None):
        try:
            g = Github(github_token) if github_token else Github()
            repo = g.get_repo(f"{username}/{repo_name}")
            
            contents = []
            dirs_to_process = [""]
            
            # Get repository structure
            repo_structure = {"dirs": set(), "files": []}
            
            while dirs_to_process:
                current_dir = dirs_to_process.pop(0)
                try:
                    items = repo.get_contents(current_dir)
                    for item in items:
                        if item.type == "dir":
                            dirs_to_process.append(item.path)
                            repo_structure["dirs"].add(item.path)
                        elif item.type == "file":
                            try:
                                content = ""
                                if item.size < 1000000:  # Only decode files smaller than 1MB
                                    if item.encoding == "base64":
                                        content = base64.b64decode(item.content).decode('utf-8', errors='ignore')
                                
                                file_info = {
                                    "name": item.name,
                                    "path": item.path,
                                    "content": content,
                                    "size": item.size,
                                    "download_url": item.download_url,
                                    "directory": os.path.dirname(item.path)
                                }
                                contents.append(file_info)
                                repo_structure["files"].append(file_info)
                            except Exception as e:
                                st.warning(f"Error reading {item.path}: {str(e)}")
                except Exception as e:
                    st.warning(f"Error accessing {current_dir}: {str(e)}")
            
            return contents, repo_structure
        except Exception as e:
            st.error(f"Error accessing repository: {str(e)}")
            return [], {"dirs": set(), "files": []}
    
    def clone_repo(self, repo_url):
        print("⬇ Cloning repository...")
        repo_dir = tempfile.mkdtemp(prefix="repo_")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
        return repo_dir
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        lines = text.splitlines()
        chunks = []
        i = 0
        while i < len(lines):
            j = min(i + chunk_size, len(lines))
            chunks.append("\n".join(lines[i:j]))
            if j == len(lines):
                break
            i = max(j - overlap, i + 1)
        return [c for c in chunks if c.strip()]
    
    def load_code_files(self, repo_path: str, extensions=None) -> List[str]:
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".tsx", ".java", ".go", ".md", ".yaml", ".yml"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(f"{repo_path}/**/*{ext}", recursive=True))
        return [f for f in files if os.path.isfile(f) and os.path.getsize(f) <= 2_000_000]
    
    def build_vectorstore(self, repo_path):
        if not self.embedding_model:
            st.error("Google API key required for embeddings.")
            return False
        
        print("🧱 Building vector store...")
        docs = []
        for fp in self.load_code_files(repo_path):
            try:
                with open(fp, "r", errors="ignore") as f:
                    txt = f.read()
                for chunk in self.chunk_text(txt):
                    docs.append(Document(page_content=chunk, metadata={"source": fp}))
            except:
                continue
        vs = FAISS.from_documents(docs, self.embedding_model)
        vs.save_local(INDEX_DIR)
        return True
    
    def generate_contribution_report(self):
        if not self.openai_client:
            return "AIML API key required for contribution report."
        
        if not os.path.exists(INDEX_DIR):
            return "Vector store not found. Please analyze the repository first."
        
        vs = FAISS.load_local(INDEX_DIR, self.embedding_model, allow_dangerous_deserialization=True)
        queries = [
            "README and documentation",
            "tests and coverage",
            "TODO and FIXME",
            "main entrypoints and core modules",
            "CI configuration and developer experience"
        ]
        gathered = []
        seen = set()
        for q in queries:
            for d in vs.similarity_search(q, k=2):
                key = (d.metadata.get("source"), d.page_content[:100])
                if key not in seen:
                    seen.add(key)
                    gathered.append(d)
        context = "\n\n".join([f"[SNIPPET {i}] {d.page_content}" for i, d in enumerate(gathered, 1)])
        prompt = f"""
You are an Open-Source Contribution Advisor.
Analyze the repository and suggest:
1. Opportunities
2. Possible Ways to Contribute
3. Quick Wins
4. Next Steps
Context:
{context}
"""
        response = self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-5-2025-08-07",
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def summarize_repo(self, contribution_report):
        if not self.openai_client:
            return "AIML API key required for repo summary."
        
        prompt = f"""
Summarize the overall working of the repository in a few paragraphs. Focus on purpose, main components, and how it works.
Based on this contribution report:
{contribution_report}
"""
        response = self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-5-2025-08-07",
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def summarize_file(self, file_path, content):
        if not self.openai_client:
            return "AIML API key required for file summary."
        
        prompt = f"""
Summarize the working of this file in a few words (1-2 sentences max). Focus on its purpose and key functions.
File: {file_path}
Content (snippet): {content[:1000]}...
"""
        response = self.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-5-2025-08-07",
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def answer_question(self, question):
        if not self.openai_client:
            return "AIML API key required for Q&A."
        
        if not self.embedding_model:
            return "Google API key required for embeddings in Q&A."
        
        if not os.path.exists(INDEX_DIR):
            return "Vector store not found. Please analyze the repository first."
        
        try:
            # Load vector store and get relevant documents
            vs = FAISS.load_local(INDEX_DIR, self.embedding_model, allow_dangerous_deserialization=True)
            docs = vs.similarity_search(question, k=10)
            context = "\n\n".join([f"[SNIPPET {i}] File: {d.metadata.get('source', 'Unknown')}\n{d.page_content}" for i, d in enumerate(docs, 1)])
            
            # Determine if this is an error/issue question
            is_error_question = any(keyword in question.lower() for keyword in 
                                  ['error', 'fix', 'bug', 'issue', 'problem', 'troubleshoot', 'debug', 'not working', 'broken'])
            
            if is_error_question:
                prompt = f"""You are a helpful coding assistant. Based on the provided code context, help solve the user's problem.

CONTEXT FROM CODEBASE:
{context}

USER QUESTION: {question}

Please provide:
1. Analysis of the problem based on the code context
2. Possible causes
3. Step-by-step solution with code examples if applicable
4. Alternative approaches if relevant

If you need to provide code solutions, format them properly with syntax highlighting.
"""
            else:
                prompt = f"""You are a helpful coding assistant. Use the provided code context to answer the user's question comprehensively.

CONTEXT FROM CODEBASE:
{context}

USER QUESTION: {question}

Instructions:
- Answer based on the code context provided
- If the question involves explaining how something works, provide clear explanations
- If relevant code examples would help, include them
- Be specific and reference the actual code when possible
- If the context doesn't fully answer the question, mention what's missing
"""
            
            response = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-5-2025-08-07",
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def analyze_dependencies_with_ai(self, content, file_path):
        if not self.openai_client:
            return {}
        
        try:
            prompt = f"""
            Analyze the following code file and extract dependencies, imports, and relationships.
            
            File path: {file_path}
            
            Code:
            {content[:2000]}...
            
            Respond with ONLY a valid JSON object in this exact format (no additional text):
            {{
                "imports": ["module1", "module2"],
                "functions": ["function1", "function2"],
                "function_calls": ["call1", "call2"],
                "file_references": ["file1.py", "file2.js"],
                "external_apis": ["api1", "api2"]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-5-2025-08-07",
                temperature=0.1
            )
            
            content_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON if it's wrapped in other text
            try:
                result = json.loads(content_response)
            except json.JSONDecodeError:
                # Try to find JSON within the response
                import re
                json_match = re.search(r'\{.*\}', content_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # If still can't parse, return empty dict and continue with regex analysis
                    return {}
            
            # Validate the structure
            expected_keys = ["imports", "functions", "function_calls", "file_references", "external_apis"]
            for key in expected_keys:
                if key not in result:
                    result[key] = []
                elif not isinstance(result[key], list):
                    result[key] = []
            
            return result
            
        except Exception as e:
            # Don't show warning for every file, just continue with regex analysis
            return {}
    
    def create_dependency_graph(self, files, repo_structure, options):
        from utils.dependency_extractors import (
            extract_python_dependencies,
            extract_javascript_dependencies,
            extract_html_dependencies,
            extract_css_dependencies
        )
        
        G = nx.DiGraph()
        file_dependencies = {}
        ai_analysis_stats = {"success": 0, "failed": 0}
        
        filtered_files = [f for f in files if f["size"] < 1000000 and not any(
            skip in f["path"] for skip in [".git", "node_modules", "__pycache__", ".pytest_cache"]
        )]
        
        if options.get("show_folder_structure", False):
            for directory in repo_structure["dirs"]:
                if directory:
                    G.add_node(f"📁 {directory}", node_type="directory", color="#FFD700")
        
        for file in filtered_files:
            file_path = file["path"]
            content = file["content"]
            file_ext = os.path.splitext(file_path)[1].lower()
            
            G.add_node(file_path, 
                      node_type="file", 
                      file_type=file_ext,
                      size=file["size"],
                      directory=file["directory"])
            
            if options.get("show_folder_structure", False) and file["directory"]:
                G.add_edge(f"📁 {file['directory']}", file_path, 
                          edge_type="contains", color="#CCCCCC")
            
            if file_ext == ".py" and content:
                deps = extract_python_dependencies(content, file_path)
                if self.openai_client:
                    ai_deps = self.analyze_dependencies_with_ai(content, file_path)
                    if ai_deps:  # Only merge if AI analysis succeeded
                        ai_analysis_stats["success"] += 1
                        for key in ai_deps:
                            if key in deps:
                                # Combine and deduplicate
                                deps[key] = list(set(deps[key] + ai_deps[key]))
                    else:
                        ai_analysis_stats["failed"] += 1
                file_dependencies[file_path] = deps
            
            elif file_ext in [".js", ".ts", ".jsx", ".tsx"] and content:
                deps = extract_javascript_dependencies(content, file_path)
                if self.openai_client:
                    ai_deps = self.analyze_dependencies_with_ai(content, file_path)
                    if ai_deps:
                        ai_analysis_stats["success"] += 1
                        for key in ai_deps:
                            if key in deps:
                                deps[key] = list(set(deps[key] + ai_deps[key]))
                    else:
                        ai_analysis_stats["failed"] += 1
                file_dependencies[file_path] = deps
            
            elif file_ext == ".html" and content:
                deps = extract_html_dependencies(content, file_path)
                file_dependencies[file_path] = deps
            
            elif file_ext == ".css" and content:
                deps = extract_css_dependencies(content, file_path)
                file_dependencies[file_path] = deps
        
        # Show AI analysis summary instead of individual warnings
        if self.openai_client and (ai_analysis_stats["success"] + ai_analysis_stats["failed"]) > 0:
            total = ai_analysis_stats["success"] + ai_analysis_stats["failed"]
            success_rate = (ai_analysis_stats["success"] / total) * 100
            
            if ai_analysis_stats["failed"] > 0:
                st.info(f"🤖 AI Analysis: {ai_analysis_stats['success']}/{total} files analyzed successfully ({success_rate:.0f}% success rate)")
            else:
                st.success(f"🤖 AI Analysis: All {ai_analysis_stats['success']} files analyzed successfully!")
        
        for file_path, deps in file_dependencies.items():
            for other_file in filtered_files:
                other_path = other_file["path"]
                if file_path == other_path:
                    continue
                
                if options.get("show_function_calls", True):
                    other_deps = file_dependencies.get(other_path, {})
                    for func in other_deps.get("functions", []):
                        if func in deps.get("function_calls", []):
                            G.add_edge(file_path, other_path, 
                                     edge_type=f"calls_function_{func}",
                                     color="#FF4444", weight=2)
                
                if options.get("show_imports", True):
                    file_basename = os.path.splitext(os.path.basename(other_path))[0]
                    relative_path = os.path.relpath(other_path, os.path.dirname(file_path))
                    
                    for imp in deps.get("imports", []):
                        if (file_basename in imp or 
                            relative_path.replace("\\", "/") in imp or
                            other_path.replace("\\", "/") in imp):
                            G.add_edge(file_path, other_path, 
                                     edge_type="imports",
                                     color="#4444FF", weight=3)
                
                if options.get("show_file_links", True):
                    for link_type in ["css_links", "js_links", "image_links", "other_links"]:
                        for link in deps.get(link_type, []):
                            if os.path.basename(other_path) in link:
                                G.add_edge(file_path, other_path, 
                                         edge_type=link_type.replace("_links", "_link"),
                                         color="#44FF44", weight=1)
        
        if options.get("min_connections", 0) > 0:
            nodes_to_remove = [node for node in G.nodes() 
                             if G.degree(node) < options["min_connections"]]
            G.remove_nodes_from(nodes_to_remove)
        
        return G, file_dependencies
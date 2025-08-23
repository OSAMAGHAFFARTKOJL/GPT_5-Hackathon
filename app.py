import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import base64
import os
import json
from github import Github
from pyvis.network import Network
import re
from collections import defaultdict
from urllib.parse import urlparse, urljoin
import ast
from groq import Groq
import pandas as pd
import tempfile, subprocess, glob, shutil
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from typing import List

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
    groq_api_key = st.text_input("GPT-5 API Key", type="password", help="For AI-powered analysis and Q&A")
    google_api_key = st.text_input("Google API Key", type="password", help="For embeddings (required for RAG)")
    
    st.header("🎨 Visualization Options")
    layout_type = st.selectbox("Graph Layout", ["spring", "kamada_kawai", "circular", "shell", "random"])
    show_function_calls = st.checkbox("Show Function Calls", value=True)
    show_imports = st.checkbox("Show Imports/Requires", value=True)
    show_file_links = st.checkbox("Show File Links (CSS/JS/Images)", value=True)
    show_folder_structure = st.checkbox("Show Folder Relationships", value=True)
    
    min_connections = st.slider("Minimum Connections to Show", 0, 10, 1)
    
    analyze_button = st.button("Analyze Repository")

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
        if not self.groq_client:
            return "Groq API key required for contribution report."
        
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
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def summarize_repo(self, contribution_report):
        if not self.groq_client:
            return "Groq API key required for repo summary."
        
        prompt = f"""
Summarize the overall working of the repository in a few paragraphs. Focus on purpose, main components, and how it works.
Based on this contribution report:
{contribution_report}
"""
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def summarize_file(self, file_path, content):
        if not self.groq_client:
            return "Groq API key required for file summary."
        
        prompt = f"""
Summarize the working of this file in a few words (1-2 sentences max). Focus on its purpose and key functions.
File: {file_path}
Content (snippet): {content[:1000]}...
"""
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def answer_question(self, question):
        if not self.groq_client:
            return "Groq API key required for Q&A."
        
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
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing question: {str(e)}"
    
    def extract_python_dependencies(self, content, file_path):
        dependencies = {
            "imports": [],
            "functions": [],
            "classes": [],
            "function_calls": [],
            "file_references": []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies["imports"].append(node.module)
                
                elif isinstance(node, ast.FunctionDef):
                    dependencies["functions"].append(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    dependencies["classes"].append(node.name)
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        dependencies["function_calls"].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        dependencies["function_calls"].append(node.func.attr)
        
        except Exception as e:
            dependencies.update(self.extract_with_regex(content, "python"))
        
        return dependencies
    
    def extract_javascript_dependencies(self, content, file_path):
        dependencies = {
            "imports": [],
            "functions": [],
            "classes": [],
            "function_calls": [],
            "file_references": []
        }
        
        import_patterns = [
            r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
            r"import\s+['\"]([^'\"]+)['\"]",
            r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            r"import\(\s*['\"]([^'\"]+)['\"]\s*\)"
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            dependencies["imports"].extend(matches)
        
        func_patterns = [
            r"function\s+(\w+)\s*\(",
            r"const\s+(\w+)\s*=\s*function\s*\(",
            r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>",
            r"(\w+)\s*:\s*function\s*\(",
            r"(\w+)\s*:\s*\([^)]*\)\s*=>"
        ]
        
        for pattern in func_patterns:
            matches = re.findall(pattern, content)
            dependencies["functions"].extend(matches)
        
        class_matches = re.findall(r"class\s+(\w+)", content)
        dependencies["classes"].extend(class_matches)
        
        call_matches = re.findall(r"(\w+)\s*\(", content)
        dependencies["function_calls"].extend(call_matches)
        
        return dependencies
    
    def extract_html_dependencies(self, content, file_path):
        dependencies = {
            "css_links": [],
            "js_links": [],
            "image_links": [],
            "other_links": [],
            "imports": [],
            "file_references": []
        }
        
        css_matches = re.findall(r'<link[^>]*href=["\']([^"\']+\.css)["\']', content, re.IGNORECASE)
        dependencies["css_links"].extend(css_matches)
        
        js_matches = re.findall(r'<script[^>]*src=["\']([^"\']+\.js[^"\']*)["\']', content, re.IGNORECASE)
        dependencies["js_links"].extend(js_matches)
        
        img_matches = re.findall(r'<img[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE)
        dependencies["image_links"].extend(img_matches)
        
        asset_matches = re.findall(r'href=["\']([^"\']+\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot))["\']', content, re.IGNORECASE)
        dependencies["other_links"].extend([match[0] for match in asset_matches])
        
        return dependencies
    
    def extract_css_dependencies(self, content, file_path):
        dependencies = {
            "imports": [],
            "url_references": [],
            "file_references": []
        }
        
        import_matches = re.findall(r'@import\s+["\']([^"\']+)["\']', content)
        dependencies["imports"].extend(import_matches)
        
        url_matches = re.findall(r'url\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)', content)
        dependencies["url_references"].extend(url_matches)
        
        return dependencies
    
    def extract_with_regex(self, content, file_type):
        dependencies = {"imports": [], "functions": [], "function_calls": []}
        
        if file_type == "python":
            import_matches = re.findall(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
            for match in import_matches:
                dependencies["imports"].extend([m for m in match if m])
            
            func_matches = re.findall(r'def\s+(\w+)\s*\(', content)
            dependencies["functions"].extend(func_matches)
        
        return dependencies
    
    def analyze_dependencies_with_ai(self, content, file_path):
        if not self.groq_client:
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
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
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
                deps = self.extract_python_dependencies(content, file_path)
                if self.groq_client:
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
                deps = self.extract_javascript_dependencies(content, file_path)
                if self.groq_client:
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
                deps = self.extract_html_dependencies(content, file_path)
                file_dependencies[file_path] = deps
            
            elif file_ext == ".css" and content:
                deps = self.extract_css_dependencies(content, file_path)
                file_dependencies[file_path] = deps
        
        # Show AI analysis summary instead of individual warnings
        if self.groq_client and (ai_analysis_stats["success"] + ai_analysis_stats["failed"]) > 0:
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

def create_enhanced_visualization(graph, layout_type="spring"):
    if len(graph.nodes()) == 0:
        return "<div>No connections found with current filters</div>"
    
    if layout_type == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout_type == "circular":
        pos = nx.circular_layout(graph)
    elif layout_type == "shell":
        pos = nx.shell_layout(graph)
    else:
        pos = nx.random_layout(graph)
    
    net = Network(height="800px", width="100%", bgcolor="#1e1e1e", 
                  font_color="white", directed=True, notebook=True)
    
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)
    
    for node in graph.nodes(data=True):
        node_id, data = node
        
        if data.get("node_type") == "directory":
            color = "#FFD700"
            size = 25
            shape = "box"
        else:
            file_ext = data.get("file_type", "")
            color_map = {
                ".py": "#3572A5", ".js": "#F7DF1E", ".ts": "#3178C6",
                ".html": "#E34C26", ".css": "#563D7C", ".json": "#40A832",
                ".md": "#083FA1", ".txt": "#CCCCCC", ".xml": "#FF9900",
                ".yml": "#808080", ".yaml": "#808080", ".jsx": "#61DAFB",
                ".tsx": "#61DAFB", ".vue": "#4FC08D", ".php": "#777BB4",
                ".java": "#ED8B00", ".cpp": "#00599C", ".c": "#A8B9CC"
            }
            color = color_map.get(file_ext, "#888888")
            size = min(15 + graph.degree(node_id) * 2, 35)
            shape = "dot"
        
        label = os.path.basename(node_id) if not node_id.startswith("📁") else node_id
        title = f"{node_id}<br>Connections: {graph.degree(node_id)}"
        if data.get("size"):
            title += f"<br>Size: {data['size']} bytes"
        
        net.add_node(node_id, label=label, color=color, size=size, 
                    title=title, shape=shape)
    
    for edge in graph.edges(data=True):
        source, target, data = edge
        edge_type = data.get("edge_type", "connected")
        
        if "function" in edge_type:
            color = "#FF4444"
            width = 3
            label = edge_type.replace("calls_function_", "calls: ")
        elif edge_type == "imports":
            color = "#4444FF"
            width = 2
            label = "imports"
        elif "link" in edge_type:
            color = "#44FF44"
            width = 1
            label = edge_type.replace("_", " ")
        elif edge_type == "contains":
            color = "#CCCCCC"
            width = 1
            label = "contains"
        else:
            color = "#888888"
            width = 1
            label = edge_type
        
        net.add_edge(source, target, color=color, width=width, 
                    title=label, label=label if len(label) < 15 else "",
                    arrows="to")
    
    html_file = "enhanced_network.html"
    net.save_graph(html_file)
    
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return html_content

def create_statistics_dashboard(graph, file_dependencies):
    st.subheader("📊 Repository Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", len([n for n in graph.nodes() if not n.startswith("📁")]))
    with col2:
        st.metric("Total Connections", len(graph.edges()))
    with col3:
        st.metric("Connected Files", len([n for n in graph.nodes() if graph.degree(n) > 0]))
    with col4:
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes()) if graph.nodes() else 0
        st.metric("Avg Connections/File", f"{avg_degree:.1f}")
    
    file_types = defaultdict(int)
    for node in graph.nodes(data=True):
        if not node[0].startswith("📁"):
            ext = os.path.splitext(node[0])[1] or "no extension"
            file_types[ext] += 1
    
    if file_types:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 File Type Distribution")
            df_types = pd.DataFrame(list(file_types.items()), columns=["Extension", "Count"])
            fig_pie = px.pie(df_types, values="Count", names="Extension", 
                           title="File Types in Repository")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("🔗 Most Connected Files")
            degrees = [(node, degree) for node, degree in graph.degree() 
                      if not node.startswith("📁")]
            degrees.sort(key=lambda x: x[1], reverse=True)
            
            top_files = degrees[:10]
            if top_files:
                df_connected = pd.DataFrame(top_files, columns=["File", "Connections"])
                df_connected["File"] = df_connected["File"].apply(os.path.basename)
                fig_bar = px.bar(df_connected, x="Connections", y="File", 
                               orientation="h", title="Top Connected Files")
                st.plotly_chart(fig_bar, use_container_width=True)

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

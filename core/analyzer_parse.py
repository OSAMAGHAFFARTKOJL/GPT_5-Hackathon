import re
import ast
import os
import json
import streamlit as st


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
    import networkx as nx

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

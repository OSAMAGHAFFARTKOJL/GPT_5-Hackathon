import os
import tempfile
import shutil
from typing import Dict, Any, Optional
import networkx as nx
from pathlib import Path

from .base_agent import BaseAgent

try:
    import git
except ImportError:
    git = None

try:
    from data.knowledge_graph import KnowledgeGraphBuilder
    from utils.github_client import GitHubClient
except ImportError:
    from src.data.knowledge_graph import KnowledgeGraphBuilder
    from src.utils.github_client import GitHubClient

class MapperAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Mapper", config)
        self.kg_builder = KnowledgeGraphBuilder()
        self.github_client = GitHubClient()
        
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        repo_url = state.get("repo_url")
        github_token = state.get("github_token")
        
        if not repo_url:
            raise ValueError("Repository URL is required")
        
        self.log_execution("Starting repository mapping", repo_url)
        
        # Clone repository to temporary directory
        temp_dir = await self._clone_repository(repo_url, github_token)
        
        try:
            # Parse code structure
            code_structure = await self._parse_codebase(temp_dir)
            
            # Build knowledge graph
            knowledge_graph = await self.kg_builder.build_graph(code_structure)
            
            # Generate visualization data
            visualization_data = await self._generate_visualization(knowledge_graph)
            
            state.update({
                "knowledge_graph": knowledge_graph,
                "code_structure": code_structure,
                "visualization_data": visualization_data,
                "temp_dir": temp_dir
            })
            
            self.log_execution("Repository mapping completed")
            return state
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
    
    async def _clone_repository(self, repo_url: str, github_token: str = None) -> str:
        temp_dir = tempfile.mkdtemp()
        
        try:
            if git is None:
                raise ImportError("GitPython is required for repository cloning")
            
            if github_token:
                auth_url = repo_url.replace("https://", f"https://{github_token}@")
                git.Repo.clone_from(auth_url, temp_dir)
            else:
                git.Repo.clone_from(repo_url, temp_dir)
            
            return temp_dir
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
    
    async def _parse_codebase(self, repo_path: str) -> Dict[str, Any]:
        # Parse Python files using tree-sitter
        structure = {
            "files": [],
            "functions": [],
            "classes": [],
            "imports": [],
            "dependencies": []
        }
        
        for py_file in Path(repo_path).rglob("*.py"):
            if "/.git/" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_structure = self._parse_python_file(content, str(py_file))
                structure["files"].append(file_structure)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse {py_file}: {e}")
        
        return structure
    
    def _parse_python_file(self, content: str, file_path: str) -> Dict[str, Any]:
        # Simplified parsing - in production, use tree-sitter
        import ast
        
        try:
            tree = ast.parse(content)
            
            file_info = {
                "path": file_path,
                "functions": [],
                "classes": [],
                "imports": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    file_info["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    file_info["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_info["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            file_info["imports"].append(f"{module}.{alias.name}")
            
            return file_info
            
        except SyntaxError:
            return {"path": file_path, "functions": [], "classes": [], "imports": []}
    
    async def _generate_visualization(self, graph: nx.Graph) -> Dict[str, Any]:
        # Generate data for PyVis visualization
        nodes = []
        edges = []
        
        for node, data in graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": data.get("label", node),
                "type": data.get("type", "unknown"),
                "size": data.get("size", 10)
            })
        
        for source, target, data in graph.edges(data=True):
            edges.append({
                "from": source,
                "to": target,
                "label": data.get("relationship", ""),
                "weight": data.get("weight", 1)
            })
        
        return {"nodes": nodes, "edges": edges}
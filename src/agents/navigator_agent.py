from typing import Dict, Any, List, Optional
import numpy as np
import networkx as nx

from .base_agent import BaseAgent

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from utils.github_client import GitHubClient
    from data.vector_store import VectorStore
except ImportError:
    from src.utils.github_client import GitHubClient
    from src.data.vector_store import VectorStore

class NavigatorAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Navigator", config)
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for NavigatorAgent")
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.github_client = GitHubClient()
        self.vector_store = VectorStore()
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query")
        repo_url = state.get("repo_url")
        knowledge_graph = state.get("knowledge_graph")
        
        if not query:
            raise ValueError("Query is required")
        
        self.log_execution("Processing query", query)
        
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])[0]
        
        # Search knowledge graph
        graph_results = await self._search_knowledge_graph(
            knowledge_graph, query, query_embedding
        )
        
        # Search GitHub issues for beginner-friendly tasks
        issue_suggestions = await self._find_beginner_issues(repo_url)
        
        # Generate comprehensive response
        response = await self._generate_response(
            query, graph_results, issue_suggestions
        )
        
        state.update({
            "query_response": response,
            "graph_search_results": graph_results,
            "issue_suggestions": issue_suggestions
        })
        
        self.log_execution("Query processing completed")
        return state
    
    async def _search_knowledge_graph(
        self, 
        graph: nx.Graph, 
        query: str, 
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        
        if not graph:
            return []
        
        results = []
        
        # Search nodes by similarity
        for node, data in graph.nodes(data=True):
            node_text = data.get("description", "") or data.get("label", str(node))
            if node_text:
                node_embedding = self.embeddings_model.encode([node_text])[0]
                similarity = np.dot(query_embedding, node_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
                )
                
                if similarity > 0.3:  # Threshold for relevance
                    results.append({
                        "node": node,
                        "type": data.get("type", "unknown"),
                        "similarity": float(similarity),  # Convert numpy float to Python float
                        "description": node_text,
                        "file_path": data.get("file_path", ""),
                        "line_number": data.get("line_number", 0)
                    })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:10]  # Return top 10 results
    
    async def _find_beginner_issues(self, repo_url: str) -> List[Dict[str, Any]]:
        try:
            # Extract owner and repo from URL
            parts = repo_url.split("/")
            if len(parts) >= 2:
                owner = parts[-2]
                repo = parts[-1].replace(".git", "")
                
                issues = await self.github_client.get_beginner_issues(owner, repo)
                return issues[:5]  # Return top 5 beginner issues
        except Exception as e:
            self.logger.warning(f"Failed to fetch GitHub issues: {e}")
        
        return []
    
    async def _generate_response(
        self, 
        query: str, 
        graph_results: List[Dict[str, Any]], 
        issue_suggestions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        response = {
            "query": query,
            "code_insights": [],
            "suggested_issues": issue_suggestions,
            "relevant_files": [],
            "summary": ""
        }
        
        # Process graph results
        for result in graph_results:
            insight = {
                "type": result["type"],
                "description": result["description"],
                "location": f"{result['file_path']}:{result['line_number']}",
                "relevance_score": float(result["similarity"])  # Ensure it's a Python float
            }
            response["code_insights"].append(insight)
            
            if result["file_path"] and result["file_path"] not in response["relevant_files"]:
                response["relevant_files"].append(result["file_path"])
        
        # Generate summary
        if graph_results:
            response["summary"] = f"Found {len(graph_results)} relevant code elements. "
            
            if graph_results[0]["type"] == "function":
                response["summary"] += f"Most relevant function: {graph_results[0]['description']}"
            elif graph_results[0]["type"] == "class":
                response["summary"] += f"Most relevant class: {graph_results[0]['description']}"
        
        if issue_suggestions:
            response["summary"] += f" Also found {len(issue_suggestions)} beginner-friendly issues."
        
        return response
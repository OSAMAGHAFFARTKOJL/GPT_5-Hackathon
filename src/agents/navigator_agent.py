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
    from data.pg_vector_store import PostgreSQLCodeVectorStore
except ImportError:
    from src.utils.github_client import GitHubClient
    from src.data.pg_vector_store import PostgreSQLCodeVectorStore

class NavigatorAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Navigator", config)
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for NavigatorAgent")
        
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.github_client = GitHubClient()
        
        # Store config for lazy initialization
        self.config = config
        self._vector_store = None
    
    @property
    def vector_store(self):
        """Lazy initialization of vector store"""
        if self._vector_store is None:
            connection_string = self.config.get("postgres_connection_string") if self.config else None
            if not connection_string:
                raise ValueError("PostgreSQL connection string is required in config")
            
            self._vector_store = PostgreSQLCodeVectorStore(
                connection_string=connection_string,
                model_name='all-MiniLM-L6-v2'
            )
        return self._vector_store

    def search_similar_code(self, query: str, top_k: int = 10, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar code snippets using the vector store"""
        try:
            # Vector store will be initialized here when accessed
            results = self.vector_store.search_similar_code(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            return results
        except Exception as e:
            self.logger.error(f"Error searching similar code: {e}")
            return []

    def analyze_repository_structure(self, repo_url: str) -> Dict[str, Any]:
        """Analyze repository structure and create navigation graph"""
        try:
            # Get repository information
            repo_info = self.github_client.get_repository_info(repo_url)
            
            # Create navigation graph
            graph = nx.DiGraph()
            
            # Add nodes for different components
            for file_info in repo_info.get('files', []):
                graph.add_node(file_info['path'], **file_info)
            
            # Add edges based on dependencies (this would need more sophisticated analysis)
            # For now, just create a basic structure
            
            return {
                'repository': repo_info,
                'navigation_graph': graph,
                'entry_points': self._identify_entry_points(repo_info),
                'key_modules': self._identify_key_modules(repo_info)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing repository structure: {e}")
            return {}

    def get_code_path_suggestions(self, user_query: str) -> List[Dict[str, Any]]:
        """Get suggested code paths based on user query"""
        try:
            # Search for similar code
            similar_code = self.search_similar_code(
                query=user_query,
                top_k=self.config.get("search_params", {}).get("default_top_k", 10),
                similarity_threshold=self.config.get("search_params", {}).get("similarity_threshold", 0.3)
            )
            
            # Generate navigation suggestions
            suggestions = []
            for code_snippet in similar_code:
                suggestion = {
                    'file_path': code_snippet.get('file_path'),
                    'function_name': code_snippet.get('function_name'),
                    'description': code_snippet.get('description'),
                    'similarity_score': code_snippet.get('similarity_score'),
                    'context': code_snippet.get('context', {}),
                    'suggested_action': self._generate_action_suggestion(code_snippet, user_query)
                }
                suggestions.append(suggestion)
            
            return suggestions
        except Exception as e:
            self.logger.error(f"Error getting code path suggestions: {e}")
            return []

    def create_exploration_plan(self, repository_url: str, user_goal: str) -> Dict[str, Any]:
        """Create a structured plan for exploring a codebase"""
        try:
            # Analyze repository structure
            repo_analysis = self.analyze_repository_structure(repository_url)
            
            # Get relevant code suggestions
            code_suggestions = self.get_code_path_suggestions(user_goal)
            
            # Create exploration plan
            plan = {
                'goal': user_goal,
                'repository': repository_url,
                'recommended_starting_points': [],
                'exploration_sequence': [],
                'key_areas_to_focus': [],
                'potential_challenges': []
            }
            
            # Populate plan based on analysis
            if repo_analysis.get('entry_points'):
                plan['recommended_starting_points'] = repo_analysis['entry_points'][:3]
            
            if code_suggestions:
                plan['key_areas_to_focus'] = [
                    {
                        'area': suggestion['file_path'],
                        'reason': suggestion['suggested_action'],
                        'priority': suggestion['similarity_score']
                    }
                    for suggestion in code_suggestions[:5]
                ]
            
            return plan
        except Exception as e:
            self.logger.error(f"Error creating exploration plan: {e}")
            return {}

    def _identify_entry_points(self, repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential entry points in the codebase"""
        entry_points = []
        
        # Look for common entry point patterns
        common_entry_files = ['main.py', 'app.py', 'index.py', '__init__.py', 'server.py']
        
        for file_info in repo_info.get('files', []):
            file_name = file_info.get('path', '').split('/')[-1]
            if file_name in common_entry_files:
                entry_points.append({
                    'path': file_info['path'],
                    'type': 'entry_point',
                    'confidence': 0.8 if file_name == 'main.py' else 0.6
                })
        
        return entry_points

    def _identify_key_modules(self, repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key modules in the codebase"""
        key_modules = []
        
        # Look for directories that typically contain important code
        important_dirs = ['src', 'lib', 'core', 'api', 'models', 'services', 'utils']
        
        for file_info in repo_info.get('files', []):
            path_parts = file_info.get('path', '').split('/')
            if len(path_parts) > 1 and path_parts[0] in important_dirs:
                key_modules.append({
                    'path': file_info['path'],
                    'module': path_parts[0],
                    'type': 'key_module'
                })
        
        return key_modules

    def _generate_action_suggestion(self, code_snippet: Dict[str, Any], user_query: str) -> str:
        """Generate action suggestions based on code snippet and user query"""
        file_path = code_snippet.get('file_path', '')
        function_name = code_snippet.get('function_name', '')
        
        if 'test' in file_path.lower():
            return f"Review test cases in {file_path} to understand expected behavior"
        elif function_name:
            return f"Examine the '{function_name}' function in {file_path} for implementation details"
        else:
            return f"Explore {file_path} to understand its role in the codebase"

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute navigation tasks (required by BaseAgent abstract method)
        This is the main entry point for the NavigatorAgent
        """
        try:
            # Log the execution start
            self.log_execution("execute", f"Processing task type: {state.get('type', 'unknown')}")
            
            # Validate input
            if not self.validate_input(state):
                return {
                    'status': 'error',
                    'type': 'error',
                    'message': 'Invalid input provided'
                }
            
            task_type = state.get('type', 'search')
            
            if task_type == 'search':
                query = state.get('query', '')
                results = self.get_code_path_suggestions(query)
                self.log_execution("search_completed", f"Found {len(results)} results for query: {query}")
                return {
                    'status': 'success',
                    'type': 'search_results',
                    'results': results,
                    'metadata': {
                        'query': query,
                        'total_results': len(results)
                    }
                }
            
            elif task_type == 'analyze_repository':
                repo_url = state.get('repository_url', '')
                analysis = self.analyze_repository_structure(repo_url)
                self.log_execution("repository_analysis_completed", f"Analyzed repository: {repo_url}")
                return {
                    'status': 'success',
                    'type': 'repository_analysis',
                    'analysis': analysis,
                    'metadata': {
                        'repository_url': repo_url
                    }
                }
            
            elif task_type == 'create_plan':
                repo_url = state.get('repository_url', '')
                goal = state.get('goal', '')
                plan = self.create_exploration_plan(repo_url, goal)
                self.log_execution("plan_creation_completed", f"Created plan for goal: {goal}")
                return {
                    'status': 'success',
                    'type': 'exploration_plan',
                    'plan': plan,
                    'metadata': {
                        'repository_url': repo_url,
                        'goal': goal
                    }
                }
            
            else:
                self.log_execution("error", f"Unknown task type: {task_type}")
                return {
                    'status': 'error',
                    'type': 'error',
                    'message': f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            self.log_execution("error", f"Exception during execution: {str(e)}")
            self.logger.error(f"Error executing task: {e}")
            return {
                'status': 'error',
                'type': 'error',
                'message': str(e)
            }

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process navigation requests (async wrapper for execute)"""
        return self.execute(request)
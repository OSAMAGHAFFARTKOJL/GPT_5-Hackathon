from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum

try:
    from agents.mapper_agent import MapperAgent
    from agents.navigator_agent import NavigatorAgent
    from agents.analyst_agent import AnalystAgent
except ImportError:
    from src.agents.mapper_agent import MapperAgent
    from src.agents.navigator_agent import NavigatorAgent
    from src.agents.analyst_agent import AnalystAgent

class WorkflowState(Enum):
    INITIALIZED = "initialized"
    MAPPING = "mapping"
    NAVIGATING = "navigating" 
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class CodeCompassState:
    repo_url: str = ""
    github_token: Optional[str] = None
    query: str = ""
    workflow_state: WorkflowState = WorkflowState.INITIALIZED
    iterations: int = 0
    max_iterations: int = 10
    
    # Agent outputs
    knowledge_graph: Optional[Any] = None
    code_structure: Optional[Dict[str, Any]] = None
    visualization_data: Optional[Dict[str, Any]] = None
    query_response: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    
    # Temporary data
    temp_dir: str = ""
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "github_token": self.github_token,
            "query": self.query,
            "workflow_state": self.workflow_state.value,
            "iterations": self.iterations,
            "knowledge_graph": self.knowledge_graph,
            "code_structure": self.code_structure,
            "visualization_data": self.visualization_data,
            "query_response": self.query_response,
            "analysis_results": self.analysis_results,
            "temp_dir": self.temp_dir,
            "errors": self.errors
        }

class CodeCompassOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mapper_agent = MapperAgent()
        self.navigator_agent = NavigatorAgent()
        self.analyst_agent = AnalystAgent()
        
    async def map_repository(self, repo_url: str, github_token: Optional[str] = None) -> Dict[str, Any]:
        """Map a repository and build knowledge graph"""
        state = CodeCompassState(
            repo_url=repo_url,
            github_token=github_token,
            workflow_state=WorkflowState.INITIALIZED
        )
        
        try:
            # Execute mapping workflow
            state = await self._execute_mapping_workflow(state)
            
            if state.workflow_state == WorkflowState.ERROR:
                raise Exception(f"Mapping failed: {'; '.join(state.errors)}")
            
            return {
                "knowledge_graph_summary": self._summarize_knowledge_graph(state.knowledge_graph),
                "visualization_data": state.visualization_data,
                "code_structure_summary": self._summarize_code_structure(state.code_structure),
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Repository mapping failed: {e}")
            raise e
        finally:
            await self._cleanup_resources(state)
    
    async def query_codebase(self, repo_url: str, query: str) -> Dict[str, Any]:
        """Query a codebase using natural language"""
        state = CodeCompassState(
            repo_url=repo_url,
            query=query,
            workflow_state=WorkflowState.INITIALIZED
        )
        
        try:
            # First map the repository if needed
            state = await self._execute_mapping_workflow(state)
            
            if state.workflow_state == WorkflowState.ERROR:
                raise Exception(f"Mapping failed: {'; '.join(state.errors)}")
            
            # Execute navigation workflow
            state = await self._execute_navigation_workflow(state)
            
            if state.workflow_state == WorkflowState.ERROR:
                raise Exception(f"Query processing failed: {'; '.join(state.errors)}")
            
            return state.query_response
            
        except Exception as e:
            self.logger.error(f"Codebase query failed: {e}")
            raise e
        finally:
            await self._cleanup_resources(state)
    
    async def analyze_codebase(self, repo_url: str) -> Dict[str, Any]:
        """Analyze codebase for code smells and contribution opportunities"""
        state = CodeCompassState(
            repo_url=repo_url,
            workflow_state=WorkflowState.INITIALIZED
        )
        
        try:
            # First map the repository
            state = await self._execute_mapping_workflow(state)
            
            if state.workflow_state == WorkflowState.ERROR:
                raise Exception(f"Mapping failed: {'; '.join(state.errors)}")
            
            # Execute analysis workflow
            state = await self._execute_analysis_workflow(state)
            
            if state.workflow_state == WorkflowState.ERROR:
                raise Exception(f"Analysis failed: {'; '.join(state.errors)}")
            
            return state.analysis_results
            
        except Exception as e:
            self.logger.error(f"Codebase analysis failed: {e}")
            raise e
        finally:
            await self._cleanup_resources(state)
    
    async def _execute_mapping_workflow(self, state: CodeCompassState) -> CodeCompassState:
        """Execute the mapping workflow"""
        state.workflow_state = WorkflowState.MAPPING
        
        try:
            state_dict = state.to_dict()
            updated_state_dict = await self.mapper_agent.execute(state_dict)
            
            # Update state object
            state.knowledge_graph = updated_state_dict.get("knowledge_graph")
            state.code_structure = updated_state_dict.get("code_structure") 
            state.visualization_data = updated_state_dict.get("visualization_data")
            state.temp_dir = updated_state_dict.get("temp_dir", "")
            
            state.workflow_state = WorkflowState.COMPLETED
            
        except Exception as e:
            state.workflow_state = WorkflowState.ERROR
            state.errors.append(f"Mapping error: {str(e)}")
            self.logger.error(f"Mapping workflow failed: {e}")
        
        return state
    
    async def _execute_navigation_workflow(self, state: CodeCompassState) -> CodeCompassState:
        """Execute the navigation workflow"""
        state.workflow_state = WorkflowState.NAVIGATING
        
        try:
            state_dict = state.to_dict()
            updated_state_dict = await self.navigator_agent.execute(state_dict)
            
            # Update state object
            state.query_response = updated_state_dict.get("query_response")
            
            state.workflow_state = WorkflowState.COMPLETED
            
        except Exception as e:
            state.workflow_state = WorkflowState.ERROR
            state.errors.append(f"Navigation error: {str(e)}")
            self.logger.error(f"Navigation workflow failed: {e}")
        
        return state
    
    async def _execute_analysis_workflow(self, state: CodeCompassState) -> CodeCompassState:
        """Execute the analysis workflow"""
        state.workflow_state = WorkflowState.ANALYZING
        
        try:
            state_dict = state.to_dict()
            updated_state_dict = await self.analyst_agent.execute(state_dict)
            
            # Update state object
            state.analysis_results = updated_state_dict.get("analysis_results")
            
            state.workflow_state = WorkflowState.COMPLETED
            
        except Exception as e:
            state.workflow_state = WorkflowState.ERROR
            state.errors.append(f"Analysis error: {str(e)}")
            self.logger.error(f"Analysis workflow failed: {e}")
        
        return state
    
    def _summarize_knowledge_graph(self, graph) -> Dict[str, Any]:
        """Create a summary of the knowledge graph"""
        if not graph:
            return {"nodes": 0, "edges": 0, "components": []}
        
        return {
            "nodes": graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
            "edges": graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0,
            "components": list(graph.nodes())[:10] if hasattr(graph, 'nodes') else []
        }
    
    def _summarize_code_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the code structure"""
        if not structure:
            return {"files": 0, "functions": 0, "classes": 0}
        
        total_functions = sum(len(f.get("functions", [])) for f in structure.get("files", []))
        total_classes = sum(len(f.get("classes", [])) for f in structure.get("files", []))
        
        return {
            "files": len(structure.get("files", [])),
            "functions": total_functions,
            "classes": total_classes,
            "sample_files": [f.get("path", "") for f in structure.get("files", [])][:5]
        }
    
    async def _cleanup_resources(self, state: CodeCompassState):
        """Clean up temporary resources"""
        if state.temp_dir:
            import shutil
            import os
            try:
                if os.path.exists(state.temp_dir):
                    shutil.rmtree(state.temp_dir)
                    self.logger.info(f"Cleaned up temporary directory: {state.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e}")
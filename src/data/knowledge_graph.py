import networkx as nx
from typing import Dict, Any, List, Tuple
import logging

class KnowledgeGraphBuilder:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def build_graph(self, code_structure: Dict[str, Any]) -> nx.DiGraph:
        """Build a knowledge graph from code structure"""
        graph = nx.DiGraph()
        
        if not code_structure or not code_structure.get("files"):
            return graph
        
        # Add nodes for files, classes, and functions
        await self._add_file_nodes(graph, code_structure["files"])
        await self._add_class_nodes(graph, code_structure["files"]) 
        await self._add_function_nodes(graph, code_structure["files"])
        
        # Add edges for relationships
        await self._add_import_edges(graph, code_structure["files"])
        await self._add_call_edges(graph, code_structure["files"])
        await self._add_inheritance_edges(graph, code_structure["files"])
        
        self.logger.info(f"Built knowledge graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    async def _add_file_nodes(self, graph: nx.DiGraph, files: List[Dict[str, Any]]):
        """Add file nodes to the graph"""
        for file_info in files:
            file_path = file_info.get("path", "")
            if file_path:
                graph.add_node(
                    f"file:{file_path}",
                    type="file",
                    label=file_path.split("/")[-1],
                    file_path=file_path,
                    description=f"File: {file_path}",
                    size=15
                )
    
    async def _add_class_nodes(self, graph: nx.DiGraph, files: List[Dict[str, Any]]):
        """Add class nodes to the graph"""
        for file_info in files:
            file_path = file_info.get("path", "")
            classes = file_info.get("classes", [])
            
            for class_info in classes:
                class_name = class_info.get("name", "")
                if class_name:
                    class_id = f"class:{file_path}:{class_name}"
                    graph.add_node(
                        class_id,
                        type="class",
                        label=class_name,
                        file_path=file_path,
                        line_number=class_info.get("line", 0),
                        description=f"Class: {class_name}",
                        methods=class_info.get("methods", []),
                        size=20
                    )
                    
                    # Add edge from file to class
                    file_id = f"file:{file_path}"
                    if graph.has_node(file_id):
                        graph.add_edge(file_id, class_id, relationship="contains")
    
    async def _add_function_nodes(self, graph: nx.DiGraph, files: List[Dict[str, Any]]):
        """Add function nodes to the graph"""
        for file_info in files:
            file_path = file_info.get("path", "")
            functions = file_info.get("functions", [])
            
            for func_info in functions:
                func_name = func_info.get("name", "")
                if func_name:
                    func_id = f"function:{file_path}:{func_name}"
                    graph.add_node(
                        func_id,
                        type="function",
                        label=func_name,
                        file_path=file_path,
                        line_number=func_info.get("line", 0),
                        description=f"Function: {func_name}",
                        args=func_info.get("args", []),
                        size=12
                    )
                    
                    # Add edge from file to function
                    file_id = f"file:{file_path}"
                    if graph.has_node(file_id):
                        graph.add_edge(file_id, func_id, relationship="contains")
    
    async def _add_import_edges(self, graph: nx.DiGraph, files: List[Dict[str, Any]]):
        """Add import relationship edges"""
        for file_info in files:
            file_path = file_info.get("path", "")
            imports = file_info.get("imports", [])
            
            file_id = f"file:{file_path}"
            
            for import_name in imports:
                # Try to find the imported module/function in our graph
                for node_id, node_data in graph.nodes(data=True):
                    node_name = node_data.get("label", "")
                    if (import_name.endswith(node_name) or 
                        node_name in import_name.split(".")):
                        
                        graph.add_edge(
                            file_id, node_id, 
                            relationship="imports",
                            weight=1
                        )
    
    async def _add_call_edges(self, graph: nx.DiGraph, files: List[Dict[str, Any]]):
        """Add function call relationship edges (simplified)"""
        # This is a simplified implementation
        # In practice, you'd parse function bodies to detect calls
        pass
    
    async def _add_inheritance_edges(self, graph: nx.DiGraph, files: List[Dict[str, Any]]):
        """Add class inheritance edges (simplified)"""
        # This would require parsing class definitions for base classes
        pass
    
    def get_node_neighbors(self, graph: nx.DiGraph, node_id: str) -> List[str]:
        """Get neighboring nodes of a given node"""
        if not graph.has_node(node_id):
            return []
        
        predecessors = list(graph.predecessors(node_id))
        successors = list(graph.successors(node_id))
        
        return predecessors + successors
    
    def find_paths(self, graph: nx.DiGraph, source: str, target: str) -> List[List[str]]:
        """Find all paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
            return paths[:10]  # Limit to first 10 paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_central_nodes(self, graph: nx.DiGraph, n: int = 10) -> List[Tuple[str, float]]:
        """Get the most central nodes in the graph"""
        if graph.number_of_nodes() == 0:
            return []
        
        try:
            centrality = nx.betweenness_centrality(graph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:n]
        except:
            # Fallback to degree centrality
            centrality = nx.degree_centrality(graph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:n]
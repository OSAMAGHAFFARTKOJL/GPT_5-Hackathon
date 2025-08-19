import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from sentence_transformers import SentenceTransformer
import logging

class VectorStore:
    """Simple in-memory vector store for code embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}  # {doc_id: embedding}
        self.documents = {}   # {doc_id: document_content}
        self.metadata = {}    # {doc_id: metadata}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the vector store"""
        try:
            # Generate embedding
            embedding = self.model.encode([content])[0]
            
            # Store everything
            self.embeddings[doc_id] = embedding
            self.documents[doc_id] = content
            self.metadata[doc_id] = metadata or {}
            
            self.logger.debug(f"Added document {doc_id} to vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents at once"""
        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if doc_id and content:
                self.add_document(doc_id, content, metadata)
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity >= score_threshold:
                    similarities.append({
                        "doc_id": doc_id,
                        "content": self.documents[doc_id],
                        "metadata": self.metadata[doc_id],
                        "similarity": float(similarity)
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        if doc_id not in self.documents:
            return None
        
        return {
            "doc_id": doc_id,
            "content": self.documents[doc_id],
            "metadata": self.metadata[doc_id],
            "embedding": self.embeddings[doc_id].tolist()
        }
    
    def delete_document(self, doc_id: str):
        """Delete a document from the store"""
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
            del self.documents[doc_id]
            del self.metadata[doc_id]
            self.logger.debug(f"Deleted document {doc_id}")
    
    def clear(self):
        """Clear all documents from the store"""
        self.embeddings.clear()
        self.documents.clear()
        self.metadata.clear()
        self.logger.info("Cleared vector store")
    
    def size(self) -> int:
        """Get the number of documents in the store"""
        return len(self.documents)
    
    def save_to_file(self, filepath: str):
        """Save the vector store to a file"""
        try:
            data = {
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
                "documents": self.documents,
                "metadata": self.metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved vector store to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load_from_file(self, filepath: str):
        """Load the vector store from a file"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"File {filepath} does not exist")
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.embeddings = {k: np.array(v) for k, v in data.get("embeddings", {}).items()}
            self.documents = data.get("documents", {})
            self.metadata = data.get("metadata", {})
            
            self.logger.info(f"Loaded vector store from {filepath} with {self.size()} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            raise

class CodeVectorStore(VectorStore):
    """Specialized vector store for code documents"""
    
    def add_code_file(self, file_path: str, content: str, language: str = "python"):
        """Add a code file with specialized metadata"""
        metadata = {
            "type": "code_file",
            "language": language,
            "file_path": file_path,
            "file_name": os.path.basename(file_path)
        }
        
        doc_id = f"file:{file_path}"
        self.add_document(doc_id, content, metadata)
    
    def add_function(self, file_path: str, func_name: str, func_content: str, 
                     line_number: int = 0, args: List[str] = None):
        """Add a function with specialized metadata"""
        metadata = {
            "type": "function",
            "file_path": file_path,
            "function_name": func_name,
            "line_number": line_number,
            "args": args or []
        }
        
        doc_id = f"function:{file_path}:{func_name}"
        self.add_document(doc_id, func_content, metadata)
    
    def add_class(self, file_path: str, class_name: str, class_content: str,
                  line_number: int = 0, methods: List[str] = None):
        """Add a class with specialized metadata"""
        metadata = {
            "type": "class", 
            "file_path": file_path,
            "class_name": class_name,
            "line_number": line_number,
            "methods": methods or []
        }
        
        doc_id = f"class:{file_path}:{class_name}"
        self.add_document(doc_id, class_content, metadata)
    
    def search_by_type(self, query: str, doc_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents of a specific type"""
        results = self.search(query, top_k * 2)  # Get more results to filter
        
        filtered_results = [
            r for r in results 
            if r["metadata"].get("type") == doc_type
        ]
        
        return filtered_results[:top_k]
    
    def get_file_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all functions from a specific file"""
        results = []
        for doc_id, metadata in self.metadata.items():
            if (metadata.get("type") == "function" and 
                metadata.get("file_path") == file_path):
                results.append(self.get_document(doc_id))
        
        return results
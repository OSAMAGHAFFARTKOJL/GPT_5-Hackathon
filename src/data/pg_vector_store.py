import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import uuid
from contextlib import contextmanager

class PostgreSQLVectorStore:
    """PostgreSQL-based vector store using pgvector extension"""
    
    def __init__(self, 
                 connection_string: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 table_name: str = "documents"):
        self.connection_string = connection_string
        self.table_name = table_name
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Verify connection and table
        self._initialize_table()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def _initialize_table(self):
        """Create table if it doesn't exist"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id VARCHAR(255) UNIQUE NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}',
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
        ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
        
        CREATE INDEX IF NOT EXISTS {self.table_name}_doc_id_idx 
        ON {self.table_name} (doc_id);
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
        
        self.logger.info(f"Initialized table {self.table_name}")
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the vector store"""
        try:
            # Generate embedding
            embedding = self.model.encode([content])[0].tolist()
            
            # Insert or update document
            upsert_sql = f"""
            INSERT INTO {self.table_name} (doc_id, content, metadata, embedding, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (doc_id) 
            DO UPDATE SET 
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(upsert_sql, (
                        doc_id, 
                        content, 
                        Json(metadata or {}), 
                        embedding
                    ))
                    conn.commit()
            
            self.logger.debug(f"Added/updated document {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add multiple documents efficiently using batch insert"""
        if not documents:
            return
        
        try:
            # Generate all embeddings at once for efficiency
            contents = [doc.get("content", "") for doc in documents if doc.get("content")]
            embeddings = self.model.encode(contents)
            
            # Prepare batch data
            batch_data = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id")
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                if doc_id and content:
                    batch_data.append((
                        doc_id,
                        content,
                        Json(metadata),
                        embeddings[i].tolist()
                    ))
            
            # Batch insert
            if batch_data:
                upsert_sql = f"""
                INSERT INTO {self.table_name} (doc_id, content, metadata, embedding, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (doc_id) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """
                
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.executemany(upsert_sql, batch_data)
                        conn.commit()
                
                self.logger.info(f"Added/updated {len(batch_data)} documents")
        
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0].tolist()
            
            # Search with cosine similarity
            search_sql = f"""
            SELECT 
                doc_id,
                content,
                metadata,
                1 - (embedding <=> %s) as similarity
            FROM {self.table_name}
            WHERE 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(search_sql, (
                        query_embedding, 
                        query_embedding, 
                        score_threshold, 
                        query_embedding, 
                        top_k
                    ))
                    results = cur.fetchall()
            
            # Convert to list of dicts
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            select_sql = f"""
            SELECT doc_id, content, metadata, embedding
            FROM {self.table_name}
            WHERE doc_id = %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(select_sql, (doc_id,))
                    result = cur.fetchone()
            
            return dict(result) if result else None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store"""
        try:
            delete_sql = f"DELETE FROM {self.table_name} WHERE doc_id = %s"
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(delete_sql, (doc_id,))
                    deleted = cur.rowcount > 0
                    conn.commit()
            
            if deleted:
                self.logger.debug(f"Deleted document {doc_id}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def clear(self):
        """Clear all documents from the store"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {self.table_name}")
                    count = cur.rowcount
                    conn.commit()
            
            self.logger.info(f"Cleared {count} documents from vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to clear vector store: {e}")
            raise
    
    def size(self) -> int:
        """Get the number of documents in the store"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    return cur.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to get store size: {e}")
            return 0


class PostgreSQLCodeVectorStore(PostgreSQLVectorStore):
    """Specialized PostgreSQL vector store for code documents"""
    
    def __init__(self, connection_string: str, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(connection_string, model_name, "code_documents")
    
    def add_code_file(self, file_path: str, content: str, language: str = "python"):
        """Add a code file with specialized metadata"""
        import os
        
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
        try:
            query_embedding = self.model.encode([query])[0].tolist()
            
            search_sql = f"""
            SELECT 
                doc_id,
                content,
                metadata,
                1 - (embedding <=> %s) as similarity
            FROM {self.table_name}
            WHERE metadata->>'type' = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(search_sql, (query_embedding, doc_type, query_embedding, top_k))
                    results = cur.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Search by type failed: {e}")
            return []
    
    def get_file_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all functions from a specific file"""
        try:
            search_sql = f"""
            SELECT doc_id, content, metadata
            FROM {self.table_name}
            WHERE metadata->>'type' = 'function' 
            AND metadata->>'file_path' = %s
            ORDER BY (metadata->>'line_number')::int
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(search_sql, (file_path,))
                    results = cur.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Failed to get file functions: {e}")
            return []


# Usage example
if __name__ == "__main__":
    # Connection string
    conn_str = "postgresql://username:password@localhost:5432/your_database"
    
    # Initialize store
    store = PostgreSQLCodeVectorStore(conn_str)
    
    # Add some code
    store.add_code_file("utils.py", "def helper_function():\n    return 'hello'", "python")
    store.add_function("main.py", "calculate", "def calculate(x, y):\n    return x + y", 10, ["x", "y"])
    
    # Search
    results = store.search("mathematical calculation", top_k=3)
    print(f"Found {len(results)} results")
    
    for result in results:
        print(f"Doc: {result['doc_id']}, Similarity: {result['similarity']:.3f}")
import os
import glob
from typing import List
import streamlit as st
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"


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

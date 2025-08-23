import os
import json
from langchain_community.vectorstores import FAISS
import streamlit as st

INDEX_DIR = "faiss_index"


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

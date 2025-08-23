import os
import json
import openai
from langchain_community.vectorstores import FAISS
import streamlit as st
from typing import Dict, List, Optional, Any

INDEX_DIR = "faiss_index"

class GPT5AnalyticsHandler:
    """
    GPT-5 powered analytics and Q&A handler for repository analysis
    """
    
    def __init__(self, openai_api_key: str = None, embedding_model = None):
        """
        Initialize GPT-5 client and embedding model
        
        Args:
            openai_api_key: OpenAI API key for GPT-5 access
            embedding_model: Embedding model for vector operations
        """
        self.openai_client = None
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client if API key is provided
        if openai_api_key:
            try:
                openai.api_key = openai_api_key
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                # Test the connection
                self.openai_client.models.list()
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.openai_client = None

    def _make_gpt5_call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """
        Make a call to GPT-5 with error handling
        
        Args:
            prompt: The prompt to send to GPT-5
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            str: GPT-5 response or error message
        """
        if not self.openai_client:
            return "OpenAI API key required for GPT-5 functionality."
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # Update to actual GPT-5 model name when available
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            return f"OpenAI API Error: {str(e)}"
        except Exception as e:
            return f"Error calling GPT-5: {str(e)}"

    def generate_contribution_report(self) -> str:
        """
        Generate AI-powered contribution opportunities report using GPT-5
        
        Returns:
            str: Detailed contribution report with opportunities and suggestions
        """
        if not self.openai_client:
            return "OpenAI API key required for contribution report."
        
        if not os.path.exists(INDEX_DIR):
            return "Vector store not found. Please analyze the repository first."
        
        try:
            # Load vector store and gather relevant context
            vs = FAISS.load_local(INDEX_DIR, self.embedding_model, allow_dangerous_deserialization=True)
            
            # Enhanced queries for better context gathering
            queries = [
                "README documentation setup instructions",
                "test files unit testing integration testing",
                "TODO FIXME issues bugs improvements",
                "main entry points core modules architecture",
                "CI configuration GitHub Actions workflow developer setup",
                "configuration files package.json requirements.txt",
                "error handling exception logging",
                "API endpoints routes controllers"
            ]
            
            gathered = []
            seen = set()
            
            for query in queries:
                try:
                    docs = vs.similarity_search(query, k=3)
                    for doc in docs:
                        # Create unique key based on source and content snippet
                        key = (doc.metadata.get("source"), doc.page_content[:100])
                        if key not in seen:
                            seen.add(key)
                            gathered.append(doc)
                except Exception as e:
                    st.warning(f"Error searching for '{query}': {str(e)}")
                    continue
            
            # Build comprehensive context
            context = "\n\n".join([
                f"[SNIPPET {i}] File: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content[:500]}..." 
                for i, doc in enumerate(gathered, 1)
            ])
            
            prompt = f"""
You are an expert Open-Source Contribution Advisor with deep knowledge of software development practices.

Analyze this repository and provide a comprehensive contribution guide:

## Repository Context:
{context}

## Please provide a detailed analysis with:

### 1. 🎯 **Contribution Opportunities**
- Identify specific areas where contributors can make meaningful impact
- Highlight gaps in functionality or documentation
- Suggest feature enhancements based on the codebase

### 2. 🚀 **Possible Ways to Contribute**
- Code contributions (bug fixes, features, optimizations)
- Documentation improvements
- Testing and quality assurance
- Community engagement opportunities

### 3. ⚡ **Quick Wins for New Contributors**
- Small, manageable tasks that provide immediate value
- Good first issues that don't require deep domain knowledge
- Documentation fixes and improvements

### 4. 📋 **Next Steps & Getting Started**
- Setup instructions and prerequisites
- Development workflow recommendations
- How to find and claim issues
- Best practices for contributing to this specific project

### 5. 🔍 **Technical Analysis**
- Code quality observations
- Architecture insights
- Potential technical debt areas

Format your response with clear headings, bullet points, and actionable recommendations.
Be specific and reference actual files/components when possible.
"""

            return self._make_gpt5_call(prompt, temperature=0.1, max_tokens=3000)
            
        except Exception as e:
            return f"Error generating contribution report: {str(e)}"

    def summarize_repo(self, contribution_report: str) -> str:
        """
        Generate comprehensive repository summary using GPT-5
        
        Args:
            contribution_report: Previously generated contribution report for context
            
        Returns:
            str: Detailed repository summary
        """
        if not self.openai_client:
            return "OpenAI API key required for repository summary."
        
        prompt = f"""
You are a Senior Software Architect tasked with creating a comprehensive repository overview.

Based on the following contribution analysis, create a detailed repository summary:

## Contribution Analysis:
{contribution_report}

## Please provide a comprehensive summary covering:

### 🎯 **Project Purpose & Vision**
- What problem does this project solve?
- Target audience and use cases
- Project goals and mission

### 🏗️ **Architecture & Components**
- High-level system architecture
- Key modules and their relationships
- Technology stack and dependencies
- Design patterns used

### ⚙️ **How It Works**
- Core functionality explanation
- Data flow and processing
- Key algorithms or methodologies
- Integration points

### 🛠️ **Development & Maintenance**
- Development practices and standards
- Testing strategy
- Deployment and CI/CD
- Code organization principles

### 📈 **Project Health & Community**
- Code quality indicators
- Community engagement
- Maintenance status
- Future roadmap insights

Write in clear, technical language suitable for developers who want to understand and potentially contribute to the project.
Focus on providing actionable insights rather than just describing what's visible.
"""

        return self._make_gpt5_call(prompt, temperature=0.2, max_tokens=2500)

    def summarize_file(self, file_path: str, content: str) -> str:
        """
        Generate concise file summary using GPT-5
        
        Args:
            file_path: Path to the file being analyzed
            content: File content (truncated if necessary)
            
        Returns:
            str: Concise file summary (1-2 sentences)
        """
        if not self.openai_client:
            return "OpenAI API key required for file summary."
        
        # Truncate content for efficiency
        truncated_content = content[:1500] if len(content) > 1500 else content
        
        prompt = f"""
You are a code documentation expert. Analyze this file and provide a concise summary.

File Path: {file_path}
Content:
```
{truncated_content}
```

Provide a brief, technical summary (1-2 sentences maximum) that explains:
- Primary purpose of this file
- Key functionality or responsibilities
- How it fits into the larger system

Be concise but informative. Focus on what the file DOES, not just what it contains.
"""

        return self._make_gpt5_call(prompt, temperature=0.1, max_tokens=150)

    def answer_question(self, question: str) -> str:
        """
        Enhanced Q&A system using GPT-5 with RAG (Retrieval-Augmented Generation)
        
        Args:
            question: User's question about the repository
            
        Returns:
            str: Comprehensive answer based on repository context
        """
        if not self.openai_client:
            return "OpenAI API key required for Q&A functionality."
        
        if not self.embedding_model:
            return "Embedding model required for semantic search in Q&A."
        
        if not os.path.exists(INDEX_DIR):
            return "Vector store not found. Please analyze the repository first."
        
        try:
            # Load vector store and perform semantic search
            vs = FAISS.load_local(INDEX_DIR, self.embedding_model, allow_dangerous_deserialization=True)
            
            # Enhanced retrieval with multiple search strategies
            primary_docs = vs.similarity_search(question, k=8)
            
            # Additional context search for better understanding
            question_keywords = question.lower().split()
            context_queries = []
            
            # Generate related queries for broader context
            if any(word in question_keywords for word in ['error', 'fix', 'bug', 'issue', 'problem']):
                context_queries.extend(['error handling', 'exception', 'try catch', 'logging'])
            if any(word in question_keywords for word in ['how', 'works', 'explain']):
                context_queries.extend(['function', 'method', 'algorithm', 'process'])
            if any(word in question_keywords for word in ['install', 'setup', 'configure']):
                context_queries.extend(['requirements', 'dependencies', 'installation', 'config'])
            
            # Gather additional context
            additional_docs = []
            for ctx_query in context_queries[:3]:  # Limit to prevent too many calls
                try:
                    ctx_docs = vs.similarity_search(ctx_query, k=2)
                    additional_docs.extend(ctx_docs)
                except:
                    continue
            
            # Combine and deduplicate documents
            all_docs = primary_docs + additional_docs
            seen_content = set()
            unique_docs = []
            
            for doc in all_docs:
                content_key = doc.page_content[:100]  # Use first 100 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_docs.append(doc)
                if len(unique_docs) >= 12:  # Limit total context
                    break
            
            # Build comprehensive context
            context = "\n\n".join([
                f"[CONTEXT {i}] File: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                for i, doc in enumerate(unique_docs, 1)
            ])
            
            # Determine question type for specialized handling
            is_error_question = any(keyword in question.lower() for keyword in 
                                  ['error', 'fix', 'bug', 'issue', 'problem', 'troubleshoot', 'debug', 
                                   'not working', 'broken', 'fails', 'exception'])
            
            is_how_question = any(keyword in question.lower() for keyword in 
                                ['how', 'explain', 'what does', 'how does', 'works', 'mechanism'])
            
            is_setup_question = any(keyword in question.lower() for keyword in 
                                  ['install', 'setup', 'configure', 'deploy', 'run', 'start'])
            
            # Create specialized prompts based on question type
            if is_error_question:
                prompt = f"""
You are an expert software debugging assistant. Help solve this coding problem using the provided repository context.

REPOSITORY CONTEXT:
{context}

USER QUESTION: {question}

Please provide a comprehensive solution including:

🔍 **Problem Analysis:**
- Identify the likely root cause based on the code context
- Reference specific files/functions where relevant

🛠️ **Step-by-Step Solution:**
- Provide detailed steps to resolve the issue
- Include code examples with proper syntax highlighting
- Show before/after comparisons if applicable

⚡ **Alternative Approaches:**
- Suggest different ways to solve the problem
- Explain trade-offs between different solutions

🚨 **Prevention:**
- How to avoid this issue in the future
- Best practices related to this problem

Format code blocks properly and reference specific files from the context when applicable.
"""
            
            elif is_how_question:
                prompt = f"""
You are a senior software engineer explaining complex systems. Use the repository context to provide a clear, educational explanation.

REPOSITORY CONTEXT:
{context}

USER QUESTION: {question}

Please provide a comprehensive explanation including:

🎯 **Core Concept:**
- What is being asked about and why it matters
- High-level overview of the mechanism

🏗️ **Technical Details:**
- How it works internally based on the code
- Key components and their interactions
- Reference specific functions/classes from the context

📊 **Flow & Process:**
- Step-by-step breakdown of the process
- Data flow and transformations
- Decision points and logic branches

💡 **Examples:**
- Practical examples from the codebase
- Usage patterns and common scenarios

🔗 **Relationships:**
- How this connects to other parts of the system
- Dependencies and interactions

Use the actual code from the context to illustrate your explanations.
"""
            
            elif is_setup_question:
                prompt = f"""
You are a DevOps and setup expert. Provide clear installation and configuration guidance based on the repository context.

REPOSITORY CONTEXT:
{context}

USER QUESTION: {question}

Please provide detailed setup instructions including:

📋 **Prerequisites:**
- System requirements and dependencies
- Required tools and versions

🔧 **Installation Steps:**
- Step-by-step installation process
- Command line instructions with proper formatting
- Configuration file examples

⚙️ **Configuration:**
- Required settings and environment variables
- Configuration options and customization
- Sample configuration files

✅ **Verification:**
- How to verify the setup is working correctly
- Common commands to test functionality
- Expected outputs

🐛 **Troubleshooting:**
- Common setup issues and solutions
- Error messages and their meanings

Reference specific files from the repository context (requirements.txt, package.json, etc.).
"""
            
            else:
                # General question prompt
                prompt = f"""
You are a knowledgeable software engineering assistant. Answer this question comprehensively using the provided repository context.

REPOSITORY CONTEXT:
{context}

USER QUESTION: {question}

Instructions:
- Provide a thorough, accurate answer based on the code context
- Reference specific files, functions, and code snippets when relevant
- If you need to show code examples, format them properly with syntax highlighting
- Explain not just WHAT but also WHY and HOW things work
- If the context doesn't fully answer the question, clearly state what information is missing
- Structure your response with clear headings and sections for better readability

Be specific, actionable, and educational in your response.
"""

            return self._make_gpt5_call(prompt, temperature=0.3, max_tokens=3500)
            
        except Exception as e:
            return f"Error processing question: {str(e)}"

    def analyze_dependencies_with_ai(self, content: str, file_path: str) -> Dict[str, List[str]]:
        """
        AI-powered dependency and relationship analysis using GPT-5
        
        Args:
            content: File content to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            Dict: Structured dependency information
        """
        if not self.openai_client:
            return {}
        
        try:
            # Truncate content for API efficiency
            truncated_content = content[:3000] if len(content) > 3000 else content
            
            prompt = f"""
You are a code analysis expert. Analyze this code file and extract all dependencies, imports, and relationships.

File Path: {file_path}

Code:
```
{truncated_content}
```

Analyze and extract:
1. **Imports/Dependencies**: All imported modules, libraries, packages
2. **Functions**: Function/method definitions in this file
3. **Function Calls**: External functions/methods being called
4. **File References**: Any references to other files or modules
5. **External APIs**: External services, databases, or APIs used
6. **Configuration Dependencies**: Config files, environment variables referenced

CRITICAL: Respond with ONLY a valid JSON object. No additional text, explanations, or formatting.

Expected JSON format:
{{
    "imports": ["module1", "module2", "package.submodule"],
    "functions": ["function_name1", "method_name2", "class_method"],
    "function_calls": ["external_func1", "api_call", "library_method"],
    "file_references": ["config.json", "data/file.csv", "templates/index.html"],
    "external_apis": ["database_connection", "rest_api", "third_party_service"],
    "config_dependencies": ["ENV_VAR", "CONFIG_FILE", "settings.py"]
}}
"""

            response_content = self._make_gpt5_call(prompt, temperature=0.1, max_tokens=1000)
            
            # Enhanced JSON parsing with error recovery
            try:
                result = json.loads(response_content.strip())
            except json.JSONDecodeError:
                # Attempt to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return self._create_empty_dependency_dict()
                else:
                    return self._create_empty_dependency_dict()
            
            # Validate and sanitize the response
            expected_keys = ["imports", "functions", "function_calls", "file_references", "external_apis", "config_dependencies"]
            validated_result = {}
            
            for key in expected_keys:
                if key in result and isinstance(result[key], list):
                    # Clean and validate list items
                    validated_result[key] = [
                        str(item).strip() for item in result[key] 
                        if item and str(item).strip()
                    ][:20]  # Limit to 20 items per category
                else:
                    validated_result[key] = []
            
            return validated_result
            
        except Exception as e:
            # Return empty dict on any error to allow fallback to regex analysis
            return self._create_empty_dependency_dict()
    
    def _create_empty_dependency_dict(self) -> Dict[str, List[str]]:
        """Create empty dependency dictionary with all expected keys"""
        return {
            "imports": [],
            "functions": [],
            "function_calls": [],
            "file_references": [],
            "external_apis": [],
            "config_dependencies": []
        }

    def generate_code_insights(self, repo_context: str) -> str:
        """
        Generate advanced code insights and recommendations using GPT-5
        
        Args:
            repo_context: Repository analysis context
            
        Returns:
            str: Detailed code insights and recommendations
        """
        if not self.openai_client:
            return "OpenAI API key required for code insights."
        
        prompt = f"""
You are a Senior Software Architect conducting a comprehensive code review.

Repository Analysis Context:
{repo_context}

Provide advanced insights covering:

### 🏗️ **Architecture Analysis**
- Overall system design patterns
- Module separation and coupling
- Scalability considerations

### 📊 **Code Quality Assessment**
- Code organization and structure
- Naming conventions and consistency
- Potential technical debt

### 🚀 **Performance Considerations**
- Potential bottlenecks
- Optimization opportunities
- Resource usage patterns

### 🔒 **Security & Best Practices**
- Security implications
- Error handling patterns
- Logging and monitoring

### 📈 **Recommendations**
- Refactoring suggestions
- Architecture improvements
- Development workflow enhancements

Be specific and actionable in your recommendations.
"""
        
        return self._make_gpt5_call(prompt, temperature=0.2, max_tokens=2500)


# Usage example and integration helper
def initialize_gpt5_handler(openai_api_key: str, embedding_model = None) -> GPT5AnalyticsHandler:
    """
    Initialize GPT-5 analytics handler with proper error handling
    
    Args:
        openai_api_key: OpenAI API key
        embedding_model: Embedding model for vector operations
        
    Returns:
        GPT5AnalyticsHandler: Initialized handler instance
    """
    try:
        handler = GPT5AnalyticsHandler(openai_api_key, embedding_model)
        if handler.openai_client:
            st.success("✅ GPT-5 analytics initialized successfully!")
        else:
            st.warning("⚠️ GPT-5 initialization failed - check API key")
        return handler
    except Exception as e:
        st.error(f"❌ Error initializing GPT-5 handler: {str(e)}")
        return GPT5AnalyticsHandler()  # Return empty handler
# 🧭 Code Compass - How to Run Guide

## Overview
Code Compass is an AI-powered tool that helps students explore and contribute to open-source GitHub repositories through intelligent code analysis, natural language queries, and contribution suggestions.

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for repository cloning)
- Internet connection (for GitHub API and ML model downloads)
- 4GB+ RAM recommended (for ML models)

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd GPT_5-Hackathon

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_structure.py
```

### 2. Running the Application

**Option A: Run Both Components Separately (Recommended)**

```bash
# Terminal 1: Start the API Server
python main.py
# ✅ API will be available at http://localhost:8000

# Terminal 2: Start the Web Interface
streamlit run src/ui/streamlit_app.py --server.port=8501
# ✅ Web UI will be available at http://localhost:8501
```

**Option B: Package Installation**

```bash
# Install as package
pip install -e .

# Run components
codecompass-api      # Start API server
codecompass-ui       # Start web interface
```

### 3. First Usage

1. **Open your browser:** Navigate to `http://localhost:8501`
2. **Enter GitHub repo:** Paste any public GitHub repository URL
3. **Choose an action:**
   - 🗺️ Map Repository (generate knowledge graph)
   - 🔍 Query Codebase (ask questions)
   - 📊 Analyze Code (get contribution suggestions)

---

## Application Components

### 🏗️ **Architecture Overview**

Code Compass uses a **multi-agent architecture** where specialized AI agents work together to analyze code repositories:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │────│   API Gateway    │────│  Multi-Agent    │
│   (Streamlit)   │    │   (FastAPI)      │    │  Orchestrator   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                         ┌───────────────────────────────┼───────────────────────────────┐
                         │                               │                               │
                ┌────────▼────────┐         ┌───────────▼───────────┐         ┌────────▼────────┐
                │  Mapper Agent   │         │  Navigator Agent      │         │  Analyst Agent  │
                │                 │         │                       │         │                 │
                │ • Code Parsing  │         │ • NLP Queries        │         │ • Code Smells   │
                │ • Knowledge     │         │ • Semantic Search    │         │ • Bug Prediction│
                │   Graph Building│         │ • Issue Discovery    │         │ • Contribution  │
                │ • Visualization │         │ • Context Retrieval  │         │   Suggestions   │
                └─────────────────┘         └─────────────────────┘         └─────────────────┘
```

---

### 🎯 **Component Details**

## 1. 🌐 **Web Interface (Streamlit UI)**

**Location:** `src/ui/streamlit_app.py`  
**Port:** `http://localhost:8501`

**Features:**
- **🗺️ Map Repository:** Interactive visualization of code structure
- **🔍 Query Codebase:** Natural language questions about code
- **📊 Analyze Code:** Quality analysis and contribution suggestions
- **⚙️ Settings:** GitHub token configuration

**Key Technologies:**
- Streamlit for web interface
- Plotly for interactive graphs
- Pandas for data display

---

## 2. 🔌 **API Gateway (FastAPI)**

**Location:** `src/api/main.py`  
**Port:** `http://localhost:8000`  

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/map-repo` | POST | Generate repository knowledge graph |
| `/query` | POST | Query codebase with natural language |
| `/analyze` | POST | Analyze code quality and suggestions |

**Request Examples:**

```json
// Map Repository
POST /map-repo
{
  "repo_url": "https://github.com/user/repo",
  "github_token": "optional_token"
}

// Query Codebase
POST /query
{
  "repo_url": "https://github.com/user/repo",
  "query": "How does authentication work?"
}

// Analyze Code
POST /analyze
{
  "repo_url": "https://github.com/user/repo"
}
```

---

## 3. 🤖 **Multi-Agent System**

### 🗺️ **Mapper Agent**
**Location:** `src/agents/mapper_agent.py`

**Responsibilities:**
- Clone GitHub repositories to temporary directories
- Parse code structure using AST (Abstract Syntax Trees)
- Build knowledge graphs with NetworkX
- Extract relationships between files, classes, and functions
- Generate visualization data for interactive displays

**Technologies:**
- GitPython for repository cloning
- Python AST parser for code analysis
- NetworkX for graph construction
- Tree-sitter for advanced parsing (optional)

### 🧭 **Navigator Agent**  
**Location:** `src/agents/navigator_agent.py`

**Responsibilities:**
- Process natural language queries about codebases
- Generate semantic embeddings using sentence-transformers
- Search knowledge graphs for relevant code elements
- Find beginner-friendly GitHub issues
- Provide contextual code explanations

**Technologies:**
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- Vector similarity search
- GitHub API integration
- Natural language processing

### 📊 **Analyst Agent**
**Location:** `src/agents/analyst_agent.py`

**Responsibilities:**
- Run static code analysis (Pylint-style)
- Detect code smells and quality issues
- Predict bug-prone areas using machine learning
- Generate contribution suggestions
- Calculate code quality scores

**Technologies:**
- scikit-learn for machine learning predictions
- Static analysis tools
- Code complexity metrics
- Quality scoring algorithms

### 🎭 **Orchestrator**
**Location:** `src/orchestrator/state_graph.py`

**Responsibilities:**
- Coordinate multi-agent workflows
- Manage shared state between agents
- Handle error recovery and retry logic
- Control iteration limits and resource cleanup
- Provide unified API interface

**Technologies:**
- LangGraph-inspired state management
- Async/await for concurrent operations
- Resource lifecycle management

---

## 4. 💾 **Data Layer**

### 📊 **Knowledge Graph**
**Location:** `src/data/knowledge_graph.py`

**Features:**
- Directed graph representation of code structure
- Nodes: files, classes, functions, imports
- Edges: relationships, dependencies, calls
- Centrality analysis and path finding
- NetworkX-based implementation

### 🔍 **Vector Store**
**Location:** `src/data/vector_store.py`

**Features:**
- Semantic embeddings storage
- Code similarity search
- Document indexing and retrieval
- Metadata management
- File persistence support

### 🌐 **GitHub Integration**
**Location:** `src/utils/github_client.py`

**Features:**
- Repository cloning and access
- Issue discovery and filtering
- API rate limit handling
- Authentication support
- Content retrieval

---

## 5. ⚙️ **Configuration & Utils**

### 🔧 **Configuration Files**
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template
- `setup.py` - Package installation configuration

### 🛠️ **Utility Modules**
- `src/utils/json_utils.py` - JSON serialization utilities
- `src/utils/github_client.py` - GitHub API wrapper

---

## 📋 **Usage Examples**

### Example 1: Exploring a New Repository

```bash
1. Start both servers (API + UI)
2. Open http://localhost:8501
3. Enter: https://github.com/fastapi/fastapi
4. Click "🗺️ Map Repository"
5. View interactive knowledge graph
6. Ask: "How does dependency injection work?"
7. Get contextual code explanations
```

### Example 2: Finding Contribution Opportunities

```bash
1. Navigate to "📊 Analyze Code"
2. Enter repository URL
3. Click "📊 Analyze"
4. Review code quality metrics
5. View beginner-friendly suggestions
6. Check recommended improvements
```

### Example 3: Code Understanding

```bash
1. Use "🔍 Query Codebase" 
2. Ask questions like:
   - "Where is user authentication handled?"
   - "How are database connections managed?"
   - "What are the main API endpoints?"
3. Get file locations and explanations
```

---

## 🚀 **Performance & Optimization**

### **First Run Considerations:**
- ML model downloads (~500MB) on first use
- Repository cloning time varies by size
- Embedding generation takes 30-60 seconds

### **Resource Usage:**
- **RAM:** 2-4GB (depending on repository size)
- **CPU:** Moderate usage during analysis
- **Disk:** Temporary storage for cloned repositories
- **Network:** GitHub API calls and model downloads

### **Optimization Tips:**
- Use GitHub tokens for higher API rate limits
- Smaller repositories process faster
- Consider local caching for repeated analyses

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**1. Import Errors**
```bash
# Fix: Install missing dependencies
pip install -r requirements.txt

# Or install specific packages
pip install sentence-transformers torch scikit-learn
```

**2. API Server Not Starting**
```bash
# Check if port 8000 is available
netstat -an | grep 8000

# Or use different port
uvicorn src.api.main:app --port 8001
```

**3. Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with specific port
streamlit run src/ui/streamlit_app.py --server.port=8502
```

**4. GitHub API Rate Limits**
```bash
# Set GitHub token in environment
export GITHUB_TOKEN=your_token_here

# Or add to .env file
echo "GITHUB_TOKEN=your_token_here" > .env
```

**5. Memory Issues**
```bash
# Monitor memory usage
top -p $(pgrep python)

# Consider smaller repositories for testing
# Clear browser cache if UI is slow
```

### **Debugging Commands:**

```bash
# Test project structure
python test_structure.py

# Test JSON serialization
python test_json_fix.py

# Check API health
curl http://localhost:8000/health

# View API documentation
# Open http://localhost:8000/docs
```

---

## 🎓 **Educational Use Cases**

### **For Students:**
- **Repository Exploration:** Understand large codebases quickly
- **Code Learning:** Ask questions about implementation patterns
- **Contribution Finding:** Discover beginner-friendly issues
- **Best Practices:** Learn from code quality analysis

### **For Educators:**
- **Code Review:** Analyze student projects
- **Teaching Aid:** Explain complex codebases interactively  
- **Assignment Creation:** Find suitable open-source projects
- **Quality Assessment:** Automated code quality metrics

### **For Open Source Maintainers:**
- **Onboarding:** Help new contributors understand projects
- **Documentation:** Generate explanations of code structure
- **Quality Monitoring:** Track code health over time
- **Issue Triage:** Identify areas needing attention

---

## 📚 **Next Steps**

After successful setup:

1. **Try the demo repositories:**
   - `https://github.com/pallets/flask` (Python web framework)
   - `https://github.com/fastapi/fastapi` (Modern API framework)
   - `https://github.com/microsoft/vscode` (Large TypeScript project)

2. **Explore advanced features:**
   - GitHub token configuration
   - Custom analysis parameters
   - Export functionality

3. **Contribute to Code Compass:**
   - Add support for new programming languages
   - Improve visualization components  
   - Enhance ML prediction models

---

**🎉 Happy exploring! Code Compass makes open-source contribution accessible and educational.**
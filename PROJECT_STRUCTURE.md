# Code Compass - Project Structure

## Overview

This document describes the complete project structure for Code Compass, an AI-powered tool to help students explore and contribute to open-source GitHub repositories.

## Directory Structure

```
E:\GitRepos\GPT_5-Hackathon\
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation configuration
├── .env.example                 # Environment variables template
├── main.py                      # Main API server entry point
├── run_streamlit.py             # Streamlit UI entry point
├── test_structure.py            # Project validation script
├── PROJECT_STRUCTURE.md         # This file
├── src/                         # Main source code
│   ├── __init__.py
│   ├── agents/                  # AI Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Base class for all agents
│   │   ├── mapper_agent.py      # Repository mapping agent
│   │   ├── navigator_agent.py   # Code query and navigation agent
│   │   └── analyst_agent.py     # Code analysis and suggestion agent
│   ├── api/                     # FastAPI REST API
│   │   ├── __init__.py
│   │   └── main.py              # API endpoints and server configuration
│   ├── orchestrator/            # Multi-agent orchestration
│   │   ├── __init__.py
│   │   └── state_graph.py       # LangGraph-inspired workflow management
│   ├── data/                    # Data layer and storage
│   │   ├── __init__.py
│   │   ├── knowledge_graph.py   # NetworkX-based knowledge graph
│   │   └── vector_store.py      # Vector embeddings storage
│   ├── ui/                      # Frontend interface
│   │   ├── __init__.py
│   │   └── streamlit_app.py     # Streamlit web application
│   ├── utils/                   # Utility modules
│   │   ├── __init__.py
│   │   └── github_client.py     # GitHub API integration
│   └── models/                  # Data models and schemas
├── tests/                       # Unit and integration tests
├── config/                      # Configuration files
└── docs/                        # Additional documentation
```

## Key Components

### 1. Multi-Agent System

- **Mapper Agent**: Clones repositories, parses code structure, builds knowledge graphs
- **Navigator Agent**: Handles natural language queries, searches code semantically
- **Analyst Agent**: Detects code smells, predicts bug-prone areas, suggests contributions
- **Orchestrator**: Manages agent workflows and shared state

### 2. API Layer

- **FastAPI Server**: RESTful endpoints (`/map-repo`, `/query`, `/analyze`)
- **Request Models**: Pydantic schemas for API validation
- **Error Handling**: Comprehensive error responses

### 3. Data Layer

- **Knowledge Graph**: NetworkX-based directed graph of code structure
- **Vector Store**: Semantic embeddings for code search using sentence-transformers
- **GitHub Integration**: Repository cloning, issue fetching, API interactions

### 4. User Interface

- **Streamlit App**: Interactive web interface with three main features:
  - Repository mapping and visualization
  - Natural language code queries
  - Code analysis and contribution suggestions

## Technology Stack

### Backend

- **FastAPI**: High-performance REST API framework
- **Python 3.8+**: Core programming language
- **NetworkX**: Graph processing and analysis
- **GitPython**: Git repository operations

### AI/ML

- **Sentence Transformers**: Semantic embeddings (all-MiniLM-L6-v2)
- **scikit-learn**: Machine learning for bug prediction
- **Tree-sitter**: Code parsing and AST generation
- **Pylint**: Static code analysis

### Frontend

- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive graph visualizations
- **Pandas**: Data manipulation and display

### External APIs

- **GitHub API**: Repository data, issues, commits
- **github3.py**: Python GitHub API client

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd GPT_5-Hackathon
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment (optional):**

   ```bash
   cp .env.example .env
   # Edit .env with your GitHub token and preferences
   ```

4. **Validate project structure:**
   ```bash
   python test_structure.py
   ```

## Running the Application

### Option 1: API Server + UI Separately

```bash
# Terminal 1: Start API server
python main.py

# Terminal 2: Start UI
python run_streamlit.py
```

### Option 2: Package Installation

```bash
pip install -e .
codecompass-api  # Start API server
codecompass-ui   # Start UI
```

## API Endpoints

- **POST /map-repo**: Generate knowledge graph from repository URL
- **POST /query**: Query codebase with natural language
- **POST /analyze**: Analyze code quality and get contribution suggestions
- **GET /health**: Health check endpoint

## Usage Examples

### 1. Map a Repository

```json
POST /map-repo
{
  "repo_url": "https://github.com/user/repo",
  "github_token": "optional_token"
}
```

### 2. Query Codebase

```json
POST /query
{
  "repo_url": "https://github.com/user/repo",
  "query": "How does authentication work in this project?"
}
```

### 3. Analyze Code

```json
POST /analyze
{
  "repo_url": "https://github.com/user/repo"
}
```

## Development Workflow

1. **Phase 1**: Basic repository mapping and graph visualization
2. **Phase 2**: Natural language queries and issue suggestions
3. **Phase 3**: Code analysis and contribution recommendations
4. **Future**: Multi-language support, voice input, PR generation

## Architecture Highlights

- **Modular Design**: Each agent is independent and can be scaled separately
- **Async/Await**: Non-blocking operations for better performance
- **Error Resilience**: Graceful handling of API failures and parsing errors
- **Extensible**: Easy to add new agents and analysis tools
- **Security**: No hardcoded tokens, environment-based configuration

## Testing & Validation

The project includes comprehensive structure validation:

- **Directory Structure**: Validates all required folders exist
- **File Structure**: Ensures all core files are present
- **Import Testing**: Tests Python module imports (without heavy dependencies)
- **Requirements Validation**: Verifies all necessary packages are listed

Run validation: `python test_structure.py`

## Deployment Considerations

- **Dependencies**: Large ML models (sentence-transformers, torch)
- **Storage**: Temporary repository storage for analysis
- **Rate Limits**: GitHub API rate limiting considerations
- **Security**: Token handling and repository access permissions

---

**Built with ❤️ to make open-source accessible for students!**

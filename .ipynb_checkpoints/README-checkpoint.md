Code Compass

Code Compass is an AI-powered tool designed to help students explore and contribute to open-source GitHub repositories. It addresses the challenge of navigating complex codebases by providing a visual knowledge graph, answering natural language queries, and suggesting beginner-friendly tasks. The tool leverages multi-agent systems, graph-based learning, NLP, and static analysis to make open-source contributions accessible.

Project Overview





Problem: Students find it daunting to contribute to open-source projects due to complex codebases and difficulty identifying starting points (e.g., bugs or features).



Solution: A modular, AI-driven system with three core agents:





Mapper Agent: Builds a knowledge graph of the codebase (files, functions, classes, dependencies).



Navigator Agent: Answers natural language queries (e.g., "How does this project handle authentication?") and suggests beginner-friendly issues.



Analyst Agent: Detects code smells and predicts bug-prone areas for meaningful contributions.



Goal: Provide an intuitive interface for students to explore repositories, understand architecture, and contribute effectively.

Architecture

The system is built as a modular, multi-agent framework inspired by LangGraph, CrewAI, and AutoGen, ensuring scalability and iterative refinement. Below is the high-level architecture:

Components





User Interface Layer:





Web-based frontend (Streamlit or React) for inputting GitHub repo URLs, querying, and visualizing code maps.



Displays interactive graphs, query responses, and contribution suggestions.



API Layer:





RESTful APIs (FastAPI) for handling user requests, GitHub authentication, and agent coordination.



Endpoints: /map-repo, /query, /analyze.



Orchestrator:





Built with LangGraph’s StateGraph to manage agent workflows and shared state (e.g., knowledge graph, query context).



Controls iterations (max 5-10 cycles) and context truncation for LLM efficiency.



Agent Layer:





Mapper Agent: Parses repos using Tree-sitter, builds a directed acyclic graph (DAG) with NetworkX/Neo4j, and visualizes via PyVis.



Navigator Agent: Uses NLP (Hugging Face embeddings, e.g., all-MiniLM-L6-v2) to query the graph and GitHub issues for tasks like "good first issue."



Analyst Agent: Runs static analysis (Pylint) and ML models (scikit-learn RandomForest) to identify code smells and bug-prone areas.



Verifier Logic: Ensures response accuracy, triggering re-runs if needed.



Data Layer:





Knowledge Graph: Stores codebase structure (nodes: files/functions/classes; edges: dependencies/calls).



Vector Store: PGVector for semantic code/doc search.



Temporary Storage: Cloned repo files, deleted post-session.



ML Models: Pre-trained for bug prediction, fine-tuned on datasets like Promise.



External Integrations:





GitHub API for repo cloning, issues, and commits.



Optional: External knowledge retrieval (e.g., Stack Overflow).

Workflow





User inputs a GitHub repo URL → Mapper builds knowledge graph.



User asks a query (e.g., "Find beginner issues") → Navigator searches graph/issues → Generates response.



Analyst scans for code smells → Suggests high-impact, low-complexity contributions.



Orchestrator manages iterations, ensuring complete and accurate outputs.

Architecture Diagram

graph TD
    UI[User Interface<br>(Streamlit/React)] --> API[API Layer<br>(FastAPI)]
    API --> ORCH[Orchestrator<br>(LangGraph StateGraph)]
    ORCH --> MAP[Mapper Agent<br>(Graph Builder)]
    MAP --> KG[Knowledge Graph<br>(NetworkX/Neo4j)]
    ORCH --> NAV[Navigator Agent<br>(Query Handler)]
    NAV --> KG
    NAV --> GH[GitHub API<br>(Issues/Commits)]
    ORCH --> ANA[Analyst Agent<br>(Smell Detector)]
    ANA --> KG
    ANA --> TOOLS[Static Tools<br>(Pylint/ML Models)]
    ORCH --> VER[Verifier Logic<br>(Iterative Check)]
    VER --> ORCH
    KG --> VS[Vector Store<br>(PGVector Embeddings)]



LangGraph: Multi-agent orchestration



Tree-sitter: Code parsing



NetworkX/PyVis: Graph construction/visualization



Hugging Face Transformers: NLP and embeddings



Pylint/scikit-learn: Static analysis and ML



FastAPI/Streamlit: Backend and UI



github3.py: GitHub API integration



PGVector: Vector storage for semantic search

See requirements.txt for full list and versions.

Usage





Map a Codebase: Enter a GitHub repo URL to generate an interactive knowledge graph.



Ask Questions: Query in natural language (e.g., "How is user authentication implemented?").



Find Contributions: Request beginner-friendly issues or high-impact areas based on code smells.

Development Roadmap





Phase 1: Mapper Agent with basic graph visualization.



Phase 2: Navigator Agent for NLP queries and issue suggestions.



Phase 3: Analyst Agent for code smells and risk scoring.



Future: Multi-language support (JS, Java), voice input (Whisper), PR generation.

Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines. Focus areas:





Enhance parsing for non-Python languages.



Optimize graph storage (e.g., Neo4j integration).



Add new agent tools for deeper analysis.

License

MIT License. See LICENSE for details.

Contact

For issues or suggestions, open a GitHub issue or contact the maintainers at [your-email@example.com].



Built with ❤️ to make open-source accessible for students!

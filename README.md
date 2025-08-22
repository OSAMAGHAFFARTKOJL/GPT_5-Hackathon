# 🔗 Open-Source Contribution Helper

A powerful Streamlit application that helps developers understand, visualize, and contribute to GitHub repositories through AI-powered analysis and interactive visualizations.

## ✨ Features

- **🔍 Smart Repository Analysis**: Automatically analyze code structure and dependencies using AST parsing and AI
- **🌐 Interactive Visualization**: Explore file relationships with dynamic, filterable network graphs
- **🤖 AI-Powered Q&A**: Ask questions about the code and get intelligent answers with code examples
- **🚀 Contribution Guidance**: Get personalized suggestions for how to contribute to projects
- **🐛 Error Solving**: Ask about errors and get step-by-step solutions
- **⚡ Real-time Filters**: Adjust visualization parameters without re-analyzing
- **📊 Statistics Dashboard**: Comprehensive repository statistics and metrics

## 🏗️ Architecture

The project is organized into modular components for maintainability and scalability:

<img width="1076" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-08-22-095901" src="https://github.com/user-attachments/assets/1340a725-7126-4031-b796-7acc0b4e81fc" />


## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd open_source_helper

# Install dependencies
pip install -r requirements.txt
```

### 2. Get API Keys

**Required:**
- **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

**Optional but Recommended:**
- **Groq API Key**: Get from [Groq Console](https://console.groq.com/)
- **GitHub Token**: Get from [GitHub Settings](https://github.com/settings/tokens)

### 3. Run the Application

```bash
streamlit run main.py
```

### 4. Usage

1. **Enter Repository URL**: Paste any public GitHub repository URL
2. **Add API Keys**: Enter your API keys in the sidebar
3. **Configure Options**: Adjust visualization settings
4. **Analyze**: Click "Analyze Repository" 
5. **Explore**: Use the interactive graph and ask questions!

## 🎯 Example Use Cases

### For Contributors
- "How does the authentication system work?"
- "What are good first issues to work on?"
- "How do I set up the development environment?"

### For Maintainers  
- "What parts of the code need documentation?"
- "Which files have the most dependencies?"
- "How is the project structured?"

### For Debugging
- "Fix this error: ModuleNotFoundError"
- "Why am I getting a connection timeout?"
- "How to resolve import conflicts?"

## 🛠️ Technical Details

### Supported Languages
- **Python**: AST parsing + regex fallback
- **JavaScript/TypeScript**: Regex-based extraction
- **HTML**: Link and script detection
- **CSS**: Import and URL extraction

### AI Features
- **Vector Embeddings**: Google's embedding-001 model
- **Language Model**: Groq's Llama 3.1 70B
- **RAG System**: Retrieval-Augmented Generation for accurate answers

### Visualization
- **Network Graphs**: PyVis-powered interactive visualizations
- **Statistics**: Plotly charts and metrics
- **Real-time Updates**: Filter changes without re-analysis

## 📝 Configuration

Key settings in `config/settings.py`:

```python
# File processing limits
MAX_FILE_SIZE = 1_000_000  # 1MB per file
CHUNK_SIZE = 800           # Text chunk size for embeddings
CHUNK_OVERLAP = 120        # Overlap between chunks

# AI models
GROQ_MODEL = "llama-3.1-70b-versatile"
EMBEDDING_MODEL = "models/embedding-001"
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** if applicable
5. **Commit**: `git commit -m 'Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Groq** for fast AI inference  
- **Google** for embedding models
- **PyVis** for network visualizations
- **NetworkX** for graph algorithms

## 🐛 Issues & Support

If you encounter any issues:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information

## 🔮 Roadmap

- [ ] Support for more programming languages
- [ ] Local LLM integration
- [ ] Repository comparison features  
- [ ] Export functionality for graphs
- [ ] CI/CD pipeline analysis
- [ ] Code quality metrics

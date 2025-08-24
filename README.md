# 🔗 Code Compass

**Open-Source Contribution Helper**

Code Compass is an intelligent tool that helps developers understand, visualize, and contribute to GitHub repositories. It combines advanced dependency analysis, interactive visualizations, and AI-powered assistance to make open-source contribution easier and more accessible.

## ✨ Features

### 🧠 AI-Powered Analysis
- **GPT-5 Integration**: Advanced code understanding and explanation
- **Smart Repository Summarization**: Get comprehensive overviews of any repository
- **Intelligent Q&A**: Ask questions about code, get explanations, and solve errors
- **Contribution Guidance**: Personalized suggestions for how to contribute

### 📊 Visual Analytics
- **Interactive Dependency Graph**: Explore file relationships and dependencies
- **Real-time Filtering**: Adjust visualizations without re-analyzing
- **Multiple Layout Options**: Spring, circular, shell, and more
- **File Type Analysis**: Comprehensive breakdown of repository structure

### 🔍 Code Analysis
- **Multi-Language Support**: Python, JavaScript, TypeScript, HTML, CSS, and more
- **Dependency Detection**: Imports, function calls, file references
- **AI-Enhanced Parsing**: Combines regex analysis with GPT-5 intelligence
- **Repository Statistics**: Detailed metrics and insights

### 🚀 Developer Experience
- **Error Troubleshooting**: Get step-by-step solutions for coding issues
- **File Summaries**: Quick understanding of what each file does
- **Contribution Opportunities**: Find TODO items, missing tests, and improvement areas
- **Interactive Interface**: User-friendly Streamlit-based web application


![Untitled diagram _ Mermaid Chart-2025-08-23-062842](https://github.com/user-attachments/assets/09e8d3c4-c095-49e5-9dd8-5513bc41bed5)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git installed on your system
- API keys for AIML (GPT-5) and Google (for embeddings)

### Quick Setup

1. **Clone the repository:**
```bash
https://github.com/OSAMAGHAFFARTKOJL/GPT_5-Hackathon
cd code-compass
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser:**
Navigate to `http://localhost:8501`

## 📋 Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:

- **streamlit** - Web application framework
- **openai** - GPT-5 API integration
- **google-generativeai** - Google AI embeddings
- **networkx** - Graph analysis and visualization
- **plotly** - Interactive charts and graphs
- **langchain** - Document processing and RAG
- **faiss-cpu** - Vector similarity search
- **PyGithub** - GitHub API integration
- **pyvis** - Interactive network visualizations

## ⚙️ Configuration

### API Keys Required

1. **AIML API Key** (Required for AI features):
   - Sign up at [AIML API](https://aimlapi.com)
   - Get your API key for GPT-5 access
   - Enter in the sidebar under "AIML API Key"

2. **Google API Key** (Required for embeddings):
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Enter in the sidebar under "Google API Key"

3. **GitHub Token** (Optional but recommended):
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Generate a new token for higher rate limits
   - Enter in the sidebar under "GitHub Token"

## 🚀 Usage

### Basic Workflow

1. **Enter Repository URL**: Paste any GitHub repository URL
2. **Configure API Keys**: Add your AIML and Google API keys
3. **Analyze Repository**: Click "Analyze Repository" button
4. **Explore Results**:
   - View interactive dependency graph
   - Read AI-generated repository summary
   - Check contribution opportunities
   - Browse file summaries

### Advanced Features

#### Interactive Q&A
Ask questions about the repository:
```
- "How does the authentication system work?"
- "Fix this error: ModuleNotFoundError: No module named 'requests'"
- "What are the main entry points of this application?"
- "Explain the database connection logic"
```

#### Visualization Filters
Customize the dependency graph:
- **Show Function Calls**: Display function relationships
- **Show Imports/Requires**: Highlight import dependencies
- **Show File Links**: CSS/JS/Image references
- **Show Folder Structure**: Directory relationships
- **Minimum Connections**: Filter nodes by connection count

#### Graph Layouts
Choose from multiple layout algorithms:
- **Spring**: Force-directed layout (default)
- **Kamada-Kawai**: Stress minimization
- **Circular**: Nodes arranged in a circle
- **Shell**: Concentric circles
- **Random**: Random positioning

## 📁 Project Structure

```
code-compass/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── faiss_index/          # Vector store cache (auto-generated)
└── enhanced_network.html # Visualization output (auto-generated)
```

## 🔧 Technical Details

### Architecture

- **Frontend**: Streamlit web interface
- **AI Backend**: GPT-5 via AIML API
- **Embeddings**: Google Generative AI embeddings
- **Vector Store**: FAISS for similarity search
- **Graph Analysis**: NetworkX for dependency mapping
- **Visualization**: Pyvis for interactive networks

### Supported File Types

- **Python** (.py): AST parsing + AI analysis
- **JavaScript/TypeScript** (.js, .ts, .jsx, .tsx): Regex + AI analysis
- **HTML** (.html): Link and asset extraction
- **CSS** (.css): Import and URL reference detection
- **Configuration** (.json, .yaml, .yml): Basic structure analysis

### Performance Optimizations

- **File Size Limits**: 1MB per file for analysis
- **Chunked Processing**: Large files split into manageable chunks
- **Caching**: Vector stores cached locally
- **Selective Analysis**: Skip binary files and common build directories

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `python -m pytest` (if applicable)
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run with debug mode
streamlit run app.py --server.runOnSave true
```

## 🐛 Troubleshooting

### Common Issues

**Error: "Invalid AIML API key"**
- Verify your API key is correct
- Check your AIML API account has credits
- Ensure the key has proper permissions

**Error: "Google API key required"**
- Get a Google AI Studio API key
- Make sure the Generative AI API is enabled
- Check API quotas and limits

**Graph shows no connections**
- Try reducing the "Minimum Connections" filter
- Enable more visualization options (imports, function calls, etc.)
- Check if the repository has analyzable code files

**Memory issues with large repositories**
- The tool automatically skips files larger than 1MB
- Consider analyzing smaller repositories first
- Close other applications to free memory

## 📊 Usage Statistics

Code Compass provides detailed analytics:
- Total files analyzed
- Connection counts and patterns
- File type distributions
- Most connected files
- AI analysis success rates

## 🔒 Privacy & Security

- **Local Processing**: Repository cloning and analysis happen locally
- **API Calls**: Only code snippets sent to AI APIs for analysis
- **No Data Storage**: No permanent storage of repository data
- **Secure Keys**: API keys stored in session only

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI/AIML**: For GPT-5 API access
- **Google**: For Generative AI embeddings
- **Streamlit**: For the excellent web framework
- **NetworkX**: For graph analysis capabilities
- **PyGithub**: For GitHub API integration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/code-compass/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/code-compass/discussions)
- **Documentation**: This README and inline code comments


**Made with ❤️ for the open-source community**

*Happy coding and contributing! 🚀*

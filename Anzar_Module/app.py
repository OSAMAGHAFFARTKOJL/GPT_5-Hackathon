# app.py - Enhanced Multi-Language Code Compass with Improved Visualization
import os
import re
import sys
import ast
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional, Any
import json
import math

import streamlit as st
from git import Repo, GitCommandError
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components


# --------------------------
# Language Detection & File Patterns
# --------------------------

LANGUAGE_EXTENSIONS = {
    'python': ['.py', '.pyx', '.pyw'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx', '.mjs'],
    'java': ['.java'],
    'cpp': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h', '.hh', '.hxx'],
    'c': ['.c', '.h'],
    'go': ['.go'],
    'rust': ['.rs'],
    'php': ['.php'],
    'ruby': ['.rb'],
    'kotlin': ['.kt', '.kts'],
    'swift': ['.swift'],
    'scala': ['.scala'],
    'dart': ['.dart'],
    'r': ['.r', '.R'],
    'matlab': ['.m'],
    'shell': ['.sh', '.bash', '.zsh'],
}

def detect_language(file_path: Path) -> Optional[str]:
    """Detect programming language from file extension."""
    ext = file_path.suffix.lower()
    for lang, extensions in LANGUAGE_EXTENSIONS.items():
        if ext in extensions:
            return lang
    return None

def get_language_files(repo_root: Path, max_size_mb: float = 2.0) -> Dict[str, List[Path]]:
    """Get all programming language files organized by language."""
    files_by_lang = {}
    
    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue
            
        try:
            if file_path.stat().st_size > max_size_mb * 1024 * 1024:
                continue
        except Exception:
            continue
            
        lang = detect_language(file_path)
        if lang:
            if lang not in files_by_lang:
                files_by_lang[lang] = []
            files_by_lang[lang].append(file_path)
    
    return files_by_lang

def safe_read_text(p: Path) -> Optional[str]:
    """Safely read text file with multiple encoding attempts."""
    encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'windows-1252', 'ascii']
    for encoding in encodings:
        try:
            with open(p, 'r', encoding=encoding, errors='replace') as f:
                return f.read()
        except Exception:
            continue
    return None


# --------------------------
# Base Language Analyzer
# --------------------------

class BaseAnalyzer:
    """Base class for language analyzers."""
    
    def __init__(self, file_path: Path, repo_root: Path):
        self.file_path = file_path
        self.repo_root = repo_root
        self.module_name = self.get_module_name()
        self.imports = []
        self.functions = {}  # name -> line_number
        self.classes = {}    # name -> line_number
        self.function_calls = []  # (caller, line, callee)
        
    def get_module_name(self) -> str:
        """Convert file path to module-like name."""
        rel_path = self.file_path.relative_to(self.repo_root)
        return str(rel_path.with_suffix("")).replace(os.sep, ".")
    
    def analyze(self) -> bool:
        """Analyze the file. Return True if successful."""
        content = safe_read_text(self.file_path)
        if not content:
            return False
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> bool:
        """Parse file content - to be implemented by subclasses."""
        raise NotImplementedError


# --------------------------
# Python Analyzer (Enhanced)
# --------------------------

class PythonAnalyzer(BaseAnalyzer):
    """Enhanced Python analyzer using AST."""
    
    def parse_content(self, content: str) -> bool:
        try:
            tree = ast.parse(content, filename=str(self.file_path))
            visitor = PythonVisitor(self.module_name)
            visitor.visit(tree)
            
            self.imports = visitor.imports
            self.functions = visitor.functions
            self.classes = visitor.classes
            self.function_calls = visitor.function_calls
            return True
        except SyntaxError:
            return False

class PythonVisitor(ast.NodeVisitor):
    """AST visitor for Python files."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.imports = []
        self.functions = {}
        self.classes = {}
        self.function_calls = []
        self.current_class = None
        self.import_aliases = {}
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
            self.import_aliases[alias.asname or alias.name.split('.')[0]] = alias.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports.append(full_name)
                self.import_aliases[alias.asname or alias.name] = full_name
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        func_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.functions[func_name] = node.lineno
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node):
        self.classes[node.name] = node.lineno
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Call(self, node):
        callee = self._get_call_name(node.func)
        if callee:
            caller = self.current_class or self.module_name
            self.function_calls.append((caller, node.lineno, callee))
        self.generic_visit(node)
    
    def _get_call_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None


# --------------------------
# JavaScript/TypeScript Analyzer
# --------------------------

class JavaScriptAnalyzer(BaseAnalyzer):
    """JavaScript/TypeScript analyzer using regex patterns."""
    
    def parse_content(self, content: str) -> bool:
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Find imports
        self._find_imports(content)
        
        # Find function definitions
        self._find_functions(content)
        
        # Find classes
        self._find_classes(content)
        
        # Find function calls
        self._find_function_calls(content)
        
        return True
    
    def _find_imports(self, content: str):
        # ES6 imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'const\s+.*?\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                self.imports.append(match.group(1))
    
    def _find_functions(self, content: str):
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=\s*function\s*\(',
            r'(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        ]
        
        for pattern in function_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.functions[match.group(1)] = line_num
    
    def _find_classes(self, content: str):
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{'
        matches = re.finditer(class_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            self.classes[match.group(1)] = line_num
    
    def _find_function_calls(self, content: str):
        call_pattern = r'(\w+(?:\.\w+)*)\s*\('
        matches = re.finditer(call_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            callee = match.group(1)
            # Simple heuristic: avoid keywords and built-ins
            if not re.match(r'^(if|for|while|switch|catch|typeof|instanceof)$', callee.split('.')[0]):
                self.function_calls.append((self.module_name, line_num, callee))


# --------------------------
# Java Analyzer
# --------------------------

class JavaAnalyzer(BaseAnalyzer):
    """Java analyzer using regex patterns."""
    
    def parse_content(self, content: str) -> bool:
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Find package and imports
        self._find_imports(content)
        
        # Find classes and interfaces
        self._find_classes(content)
        
        # Find methods
        self._find_methods(content)
        
        # Find method calls
        self._find_method_calls(content)
        
        return True
    
    def _find_imports(self, content: str):
        import_pattern = r'import\s+(?:static\s+)?([^;]+);'
        matches = re.finditer(import_pattern, content, re.MULTILINE)
        for match in matches:
            self.imports.append(match.group(1).strip())
    
    def _find_classes(self, content: str):
        class_patterns = [
            r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)',
            r'(?:public\s+)?interface\s+(\w+)',
            r'(?:public\s+)?enum\s+(\w+)'
        ]
        
        for pattern in class_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.classes[match.group(1)] = line_num
    
    def _find_methods(self, content: str):
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*\{'
        matches = re.finditer(method_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            method_name = match.group(1)
            if method_name not in ['if', 'for', 'while', 'switch', 'catch']:
                self.functions[method_name] = line_num
    
    def _find_method_calls(self, content: str):
        call_pattern = r'(\w+(?:\.\w+)*)\s*\('
        matches = re.finditer(call_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            callee = match.group(1)
            # Filter out keywords
            if not re.match(r'^(if|for|while|switch|catch|new|super|this)$', callee.split('.')[0]):
                self.function_calls.append((self.module_name, line_num, callee))


# --------------------------
# C++ Analyzer
# --------------------------

class CppAnalyzer(BaseAnalyzer):
    """C++ analyzer using regex patterns."""
    
    def parse_content(self, content: str) -> bool:
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Find includes
        self._find_includes(content)
        
        # Find classes and structs
        self._find_classes(content)
        
        # Find functions
        self._find_functions(content)
        
        # Find function calls
        self._find_function_calls(content)
        
        return True
    
    def _find_includes(self, content: str):
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        matches = re.finditer(include_pattern, content, re.MULTILINE)
        for match in matches:
            self.imports.append(match.group(1))
    
    def _find_classes(self, content: str):
        class_patterns = [
            r'class\s+(\w+)(?:\s*:\s*[^{]+)?\s*\{',
            r'struct\s+(\w+)(?:\s*:\s*[^{]+)?\s*\{'
        ]
        
        for pattern in class_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.classes[match.group(1)] = line_num
    
    def _find_functions(self, content: str):
        # Function definition pattern (simplified)
        func_pattern = r'(?:inline\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?\s*\{'
        matches = re.finditer(func_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            func_name = match.group(1)
            # Filter out keywords and common patterns
            if func_name not in ['if', 'for', 'while', 'switch', 'catch', 'class', 'struct', 'namespace']:
                self.functions[func_name] = line_num
    
    def _find_function_calls(self, content: str):
        call_pattern = r'(\w+(?:::\w+)*(?:\.\w+)*)\s*\('
        matches = re.finditer(call_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            callee = match.group(1)
            # Filter out keywords
            if not re.match(r'^(if|for|while|switch|catch|sizeof|typeof|new|delete)$', callee.split('::')[0]):
                self.function_calls.append((self.module_name, line_num, callee))


# --------------------------
# Go Analyzer
# --------------------------

class GoAnalyzer(BaseAnalyzer):
    """Go analyzer using regex patterns."""
    
    def parse_content(self, content: str) -> bool:
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Find imports
        self._find_imports(content)
        
        # Find structs and interfaces
        self._find_types(content)
        
        # Find functions
        self._find_functions(content)
        
        # Find function calls
        self._find_function_calls(content)
        
        return True
    
    def _find_imports(self, content: str):
        # Single import
        single_import = r'import\s+"([^"]+)"'
        matches = re.finditer(single_import, content, re.MULTILINE)
        for match in matches:
            self.imports.append(match.group(1))
        
        # Block import
        block_pattern = r'import\s*\(\s*((?:[^)]+\n?)*)\s*\)'
        block_matches = re.finditer(block_pattern, content, re.DOTALL)
        for block_match in block_matches:
            import_block = block_match.group(1)
            import_lines = re.finditer(r'"([^"]+)"', import_block)
            for line_match in import_lines:
                self.imports.append(line_match.group(1))
    
    def _find_types(self, content: str):
        type_patterns = [
            r'type\s+(\w+)\s+struct\s*\{',
            r'type\s+(\w+)\s+interface\s*\{'
        ]
        
        for pattern in type_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.classes[match.group(1)] = line_num
    
    def _find_functions(self, content: str):
        # Function definition pattern
        func_pattern = r'func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)\s*\([^)]*\)(?:\s*[^{]+)?\s*\{'
        matches = re.finditer(func_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            self.functions[match.group(1)] = line_num
    
    def _find_function_calls(self, content: str):
        call_pattern = r'(\w+(?:\.\w+)*)\s*\('
        matches = re.finditer(call_pattern, content, re.MULTILINE)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            callee = match.group(1)
            # Filter out keywords
            if not re.match(r'^(if|for|while|switch|range|select|go|defer|return|make|new|len|cap|append|copy|delete|panic|recover)$', callee.split('.')[0]):
                self.function_calls.append((self.module_name, line_num, callee))


# --------------------------
# Analyzer Factory
# --------------------------

def create_analyzer(file_path: Path, repo_root: Path, language: str) -> Optional[BaseAnalyzer]:
    """Create appropriate analyzer for the given language."""
    analyzers = {
        'python': PythonAnalyzer,
        'javascript': JavaScriptAnalyzer,
        'java': JavaAnalyzer,
        'cpp': CppAnalyzer,
        'c': CppAnalyzer,  # Use C++ analyzer for C files
        'go': GoAnalyzer,
    }
    
    analyzer_class = analyzers.get(language)
    if analyzer_class:
        return analyzer_class(file_path, repo_root)
    return None


# --------------------------
# Enhanced Graph Layout Functions
# --------------------------

def calculate_hierarchical_layout(G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """Calculate hierarchical layout based on file dependencies."""
    pos = {}
    
    # Separate nodes by type
    file_nodes = [n for n in G.nodes() if n.startswith("file::")]
    func_nodes = [n for n in G.nodes() if n.startswith("func::")]
    class_nodes = [n for n in G.nodes() if n.startswith("class::")]
    
    # Group files by directory level
    file_levels = {}
    for node in file_nodes:
        path = G.nodes[node].get('path', '')
        level = path.count('/') + path.count('\\')
        if level not in file_levels:
            file_levels[level] = []
        file_levels[level].append(node)
    
    # Position files in layers
    y_offset = 0
    layer_height = 200
    
    for level in sorted(file_levels.keys()):
        files = file_levels[level]
        x_spacing = max(800 / max(len(files), 1), 100)
        
        for i, node in enumerate(files):
            x = (i - len(files) / 2) * x_spacing
            pos[node] = (x, y_offset)
        
        y_offset -= layer_height
    
    # Position functions and classes around their parent files
    for node in func_nodes + class_nodes:
        # Find parent file
        module_name = node.split("::", 1)[1].rsplit(".", 1)[0]
        parent_file = f"file::{module_name}"
        
        if parent_file in pos:
            px, py = pos[parent_file]
            # Create a circular arrangement around the parent
            angle = hash(node) % 360
            radius = 80 if node.startswith("func::") else 60
            
            x = px + radius * math.cos(math.radians(angle))
            y = py + radius * math.sin(math.radians(angle))
            pos[node] = (x, y)
        else:
            # Fallback position
            pos[node] = (0, 0)
    
    return pos

def apply_force_directed_improvements(G: nx.DiGraph, pos: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Apply force-directed adjustments to improve layout."""
    # Use spring layout with initial positions
    try:
        improved_pos = nx.spring_layout(
            G, 
            pos=pos, 
            k=3, 
            iterations=50,
            weight=None
        )
        return improved_pos
    except:
        return pos

def group_nodes_by_language(G: nx.DiGraph) -> Dict[str, List[str]]:
    """Group nodes by programming language."""
    language_groups = {}
    
    for node, data in G.nodes(data=True):
        language = data.get('language', 'unknown')
        if language not in language_groups:
            language_groups[language] = []
        language_groups[language].append(node)
    
    return language_groups


# --------------------------
# Multi-Language Graph Builder
# --------------------------

def build_multi_language_graph(repo_root: Path, selected_languages: Set[str] = None) -> nx.DiGraph:
    """Build dependency graph for multiple programming languages."""
    G = nx.DiGraph()
    
    # Get all files organized by language
    files_by_lang = get_language_files(repo_root)
    
    # Filter by selected languages if specified
    if selected_languages:
        files_by_lang = {k: v for k, v in files_by_lang.items() if k in selected_languages}
    
    # Analyze each file
    all_analyzers = {}
    
    for language, files in files_by_lang.items():
        st.write(f"Analyzing {len(files)} {language} files...")
        
        for file_path in files:
            analyzer = create_analyzer(file_path, repo_root, language)
            if analyzer and analyzer.analyze():
                all_analyzers[analyzer.module_name] = analyzer
                
                # Calculate file importance metrics
                importance = len(analyzer.functions) + len(analyzer.classes) * 2
                complexity = len(analyzer.function_calls)
                
                # Add file node with enhanced attributes
                file_node = f"file::{analyzer.module_name}"
                G.add_node(
                    file_node,
                    label=file_path.name,
                    title=f"File: {file_path.relative_to(repo_root)}\nLanguage: {language.title()}\nFunctions: {len(analyzer.functions)}\nClasses: {len(analyzer.classes)}",
                    type="file",
                    language=language,
                    group=language,
                    path=str(file_path.relative_to(repo_root)),
                    importance=importance,
                    complexity=complexity,
                    directory=str(file_path.parent.relative_to(repo_root))
                )
                
                # Add function nodes
                for func_name, line_num in analyzer.functions.items():
                    func_node = f"func::{analyzer.module_name}.{func_name}"
                    G.add_node(
                        func_node,
                        label=func_name,
                        title=f"Function: {func_name}\nFile: {analyzer.module_name}\nLine: {line_num}\nLanguage: {language.title()}",
                        type="function",
                        language=language,
                        group=analyzer.module_name,
                        line=line_num,
                        parent=file_node
                    )
                
                # Add class nodes
                for class_name, line_num in analyzer.classes.items():
                    class_node = f"class::{analyzer.module_name}.{class_name}"
                    G.add_node(
                        class_node,
                        label=class_name,
                        title=f"Class: {class_name}\nFile: {analyzer.module_name}\nLine: {line_num}\nLanguage: {language.title()}",
                        type="class",
                        language=language,
                        group=analyzer.module_name,
                        line=line_num,
                        parent=file_node
                    )
    
    # Add import/dependency edges
    for analyzer in all_analyzers.values():
        file_node = f"file::{analyzer.module_name}"
        
        for import_name in analyzer.imports:
            # Try to find matching file in our analyzed files
            target_file = None
            
            # Direct match
            if import_name in all_analyzers:
                target_file = f"file::{import_name}"
            else:
                # Partial match (e.g., import might be "utils" matching "src.utils")
                for module_name in all_analyzers:
                    if module_name.endswith(f".{import_name}") or module_name == import_name:
                        target_file = f"file::{module_name}"
                        break
                    # For relative imports
                    if import_name.replace("/", ".").replace("\\", ".") in module_name:
                        target_file = f"file::{module_name}"
                        break
            
            if target_file and target_file != file_node:
                G.add_edge(
                    file_node, target_file,
                    relation="imports",
                    title=f"Imports {import_name}",
                    weight=1
                )
    
    # Add function call edges
    for analyzer in all_analyzers.values():
        for caller, line_num, callee in analyzer.function_calls:
            caller_node = f"file::{analyzer.module_name}"
            
            # Try to find the target function
            target_func = None
            
            # Look for exact matches first
            for module_name, target_analyzer in all_analyzers.items():
                # Direct function match
                if callee in target_analyzer.functions:
                    target_func = f"func::{module_name}.{callee}"
                    break
                
                # Method call (Class.method)
                if "." in callee:
                    parts = callee.split(".")
                    if len(parts) == 2 and parts[0] in target_analyzer.classes and parts[1] in target_analyzer.functions:
                        target_func = f"func::{module_name}.{parts[1]}"
                        break
                
                # Qualified call (module.function)
                if callee.startswith(module_name.split(".")[-1] + "."):
                    func_name = callee.split(".", 1)[1]
                    if func_name in target_analyzer.functions:
                        target_func = f"func::{module_name}.{func_name}"
                        break
            
            if target_func:
                G.add_edge(
                    caller_node, target_func,
                    relation="calls",
                    title=f"Calls {callee} (line {line_num})",
                    line=line_num,
                    weight=2
                )
    
    return G


# --------------------------
# Enhanced Visualization with Better Layout
# --------------------------

def render_enhanced_pyvis(G: nx.DiGraph, show_functions: bool, show_classes: bool, 
                         min_degree: int, physics: bool, height_px: int = 720,
                         filter_languages: Set[str] = None, layout_style: str = "hierarchical") -> str:
    """Enhanced PyVis renderer with improved layout and visual organization."""
    H = G.copy()
    
    # Language filter
    if filter_languages:
        nodes_to_keep = []
        for n, data in H.nodes(data=True):
            if data.get('language') in filter_languages:
                nodes_to_keep.append(n)
        
        # Keep nodes and their connections
        H = H.subgraph(nodes_to_keep).copy()
    
    # Node type filters
    if not show_functions:
        remove_nodes = [n for n, d in H.nodes(data=True) if d.get("type") == "function"]
        H.remove_nodes_from(remove_nodes)
    
    if not show_classes:
        remove_nodes = [n for n, d in H.nodes(data=True) if d.get("type") == "class"]
        H.remove_nodes_from(remove_nodes)
    
    # Degree filter
    if min_degree > 0:
        to_remove = [n for n in H.nodes if H.degree(n) < min_degree]
        H.remove_nodes_from(to_remove)
    
    if H.number_of_nodes() == 0:
        st.warning("No nodes to display with current filters.")
        return None
    
    # Enhanced language-specific colors with better contrast
    language_colors = {
        'python': '#306998',      # Python blue
        'javascript': '#F0DB4F',  # JavaScript yellow
        'java': '#ED8B00',        # Java orange
        'cpp': '#00599C',         # C++ blue
        'c': '#A8B9CC',          # C gray-blue
        'go': '#00ADD8',         # Go cyan
        'rust': '#CE422B',       # Rust orange-red
        'php': '#777BB4',        # PHP purple
        'ruby': '#CC342D',       # Ruby red
        'kotlin': '#7F52FF',     # Kotlin purple
        'swift': '#FA7343',      # Swift orange
        'scala': '#DC322F',      # Scala red
        'dart': '#0175C2',       # Dart blue
        'r': '#276DC3',          # R blue
        'matlab': '#0076A8',     # MATLAB blue
        'shell': '#4EAA25',      # Shell green
    }
    
    # Create the network with enhanced settings
    net = Network(
        height=f"{height_px}px", 
        width="100%", 
        directed=True, 
        notebook=False, 
        cdn_resources="in_line",
        bgcolor="#ffffff",
        font_color="#2c3e50"
    )
    
    # Configure physics based on layout style
    if layout_style == "hierarchical" and not physics:
        net.set_options("""
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "nodeSpacing": 200,
              "levelSeparation": 300,
              "treeSpacing": 200
            }
          },
          "physics": {
            "enabled": false
          },
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "vertical",
              "roundness": 0.4
            }
          }
        }
        """)
    elif physics:
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          },
          "edges": {
            "smooth": {
              "type": "dynamic"
            }
          }
        }
        """)
    else:
        net.set_options("""
        var options = {
          "physics": {"enabled": false},
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "roundness": 0.2
            }
          }
        }
        """)
    
    # Group nodes by language for better organization
    language_groups = group_nodes_by_language(H)
    
    # Calculate layout positions if not using physics
    if not physics and layout_style == "force_directed":
        try:
            pos = calculate_hierarchical_layout(H)
            pos = apply_force_directed_improvements(H, pos)
        except:
            pos = {}
    else:
        pos = {}
    
    # Add nodes with enhanced styling and positioning
    for n, data in H.nodes(data=True):
        node_type = data.get("type", "file")
        language = data.get("language", "unknown")
        label = data.get("label", str(n))
        title = data.get("title", label)
        importance = data.get("importance", 1)
        
        # Enhanced color scheme based on language and type
        base_color = language_colors.get(language, "#95A5A6")
        
        # Size based on node type and importance
        if node_type == "function":
            color = base_color
            border_color = "#34495E"
            shape = "dot"
            size = max(15, min(30, 15 + importance * 2))
            font_size = 14
        elif node_type == "class":
            color = base_color
            border_color = "#2C3E50"
            shape = "diamond" 
            size = max(20, min(35, 20 + importance * 2))
            font_size = 16
        else:  # file
            color = base_color
            border_color = "#2C3E50"
            shape = "box"
            size = max(25, min(50, 25 + importance * 3))
            font_size = 18
        
        # Enhanced title with more information
        enhanced_title = title
        if node_type == "file":
            directory = data.get("directory", "")
            complexity = data.get("complexity", 0)
            enhanced_title += f"\nDirectory: {directory}\nComplexity Score: {complexity}"
        
        # Position from calculated layout
        x, y = pos.get(n, (None, None))
        
        net.add_node(
            n, 
            label=label, 
            title=enhanced_title,
            color={
                'background': color,
                'border': border_color,
                'highlight': {'background': color, 'border': '#E74C3C'},
                'hover': {'background': color, 'border': '#E67E22'}
            },
            shape=shape, 
            size=size,
            font={'size': font_size, 'color': '#2C3E50', 'face': 'Arial'},
            borderWidth=2,
            shadow={'enabled': True, 'color': 'rgba(0,0,0,0.2)', 'size': 5},
            x=x,
            y=y
        )
    
    # Add edges with enhanced styling based on relationship type
    for u, v, data in H.edges(data=True):
        relation = data.get("relation", "connected")
        title = data.get("title", relation)
        weight = data.get("weight", 1)
        
        if relation == "imports":
            color = "#7F8C8D"  # Gray for imports
            dashes = [5, 5]    # Dashed line
            width = max(1, weight)
            arrows = {"to": {"enabled": True, "scaleFactor": 0.8}}
        elif relation == "calls":
            color = "#E74C3C"  # Red for function calls
            dashes = False     # Solid line
            width = max(2, weight)
            arrows = {"to": {"enabled": True, "scaleFactor": 1.0}}
        else:
            color = "#95A5A6"  # Default gray
            dashes = False
            width = 1
            arrows = {"to": {"enabled": True, "scaleFactor": 0.6}}
        
        net.add_edge(
            u, v, 
            title=title,
            color={'color': color, 'highlight': '#3498DB', 'hover': '#3498DB'},
            dashes=dashes,
            width=width,
            arrows=arrows,
            smooth={'enabled': True, 'type': 'dynamic', 'roundness': 0.2}
        )
    
    # Generate and save HTML with UTF-8 encoding
    html_path = Path(tempfile.gettempdir()) / f"enhanced_graph_{hash(str(H.nodes))}.html"
    
    try:
        # Generate HTML content with custom styling
        html_content = net.generate_html()
        
        # Add custom CSS for better visual appearance
        custom_css = """
        <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            margin: 0;
            padding: 20px;
        }
        #mynetworkid {
            border: 2px solid #34495E;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            background: white;
        }
        .vis-network:focus {
            outline: none;
        }
        </style>
        """
        
        # Insert custom CSS into the HTML
        html_content = html_content.replace('<head>', f'<head>{custom_css}')
        
        # Add custom JavaScript for enhanced interactions
        custom_js = """
        <script>
        // Enhanced interaction handlers
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                console.log("Selected node:", nodeId);
                // Add custom click behavior here
            }
        });
        
        network.on("hoverNode", function(params) {
            // Enhanced hover effects
            network.canvas.body.container.style.cursor = 'pointer';
        });
        
        network.on("blurNode", function(params) {
            network.canvas.body.container.style.cursor = 'default';
        });
        </script>
        """
        
        # Insert custom JavaScript before closing body tag
        html_content = html_content.replace('</body>', f'{custom_js}</body>')
        
        # Write with explicit UTF-8 encoding
        with open(html_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(html_content)
        
        return str(html_path)
    except Exception as e:
        st.error(f"Graph rendering error: {e}")
        return None


# --------------------------
# Enhanced Streamlit App with Better UI
# --------------------------

st.set_page_config(
    page_title="Multi-Language Code Compass", 
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.5rem 2rem;
    border-radius: 25px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Header with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🧭 Multi-Language Code Compass</h1>
    <p>Analyze <strong>any programming language</strong> from public GitHub repositories and visualize code dependencies with enhanced interactive graphs!</p>
</div>
""", unsafe_allow_html=True)

# Input section with better layout
col1, col2 = st.columns([3, 1])
with col1:
    url = st.text_input(
        "🔗 GitHub Repository URL", 
        placeholder="https://github.com/microsoft/vscode",
        help="Enter any public GitHub repository URL to analyze its code structure"
    )

with col2:
    st.write("")  # Spacing
    analyze_button = st.button("🚀 Analyze Repository", type="primary", use_container_width=True)

# Enhanced configuration section
st.markdown("## ⚙️ Analysis Configuration")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🔤 Languages")
    available_languages = list(LANGUAGE_EXTENSIONS.keys())
    selected_languages = st.multiselect(
        "Select languages to analyze:",
        available_languages,
        default=['python', 'javascript', 'java', 'cpp', 'go'],
        help="Choose which programming languages to include in the analysis"
    )

with col2:
    st.markdown("### 👁️ Display Options")
    show_functions = st.checkbox("Show Functions", value=True, help="Display function nodes and call relationships")
    show_classes = st.checkbox("Show Classes", value=True, help="Display class/struct nodes")
    
with col3:
    st.markdown("### 🎛️ Graph Filters")
    min_degree = st.slider("Min Connections", 0, 10, 0, help="Hide nodes with fewer connections")
    layout_style = st.selectbox(
        "Layout Style", 
        ["hierarchical", "force_directed", "physics"],
        index=0,
        help="Choose the graph layout algorithm"
    )
    physics = layout_style == "physics"

with col4:
    st.markdown("### 🎨 Visualization")
    height = st.slider("Graph Height (px)", 500, 1200, 720, help="Adjust the height of the visualization")
    max_file_size = st.slider("Max File Size (MB)", 0.5, 5.0, 2.0, step=0.5, help="Skip files larger than this size")

# Main analysis logic with enhanced error handling and progress tracking
if analyze_button and url.strip():
    if not selected_languages:
        st.error("🚫 Please select at least one programming language to analyze.")
    else:
        # Create temporary directory for cloning
        tmp_dir = Path(tempfile.mkdtemp(prefix="multi_lang_compass_"))
        repo_dir = tmp_dir / "repo"
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Clone repository
            status_text.text("🔄 Cloning repository...")
            progress_bar.progress(10)
            
            Repo.clone_from(url.strip(), repo_dir, depth=1)
            st.success("✅ Repository cloned successfully")
            progress_bar.progress(25)
            
            # Discover files
            status_text.text("🔍 Discovering source files...")
            files_by_lang = get_language_files(repo_dir, max_file_size)
            
            # Filter by selected languages
            files_by_lang = {k: v for k, v in files_by_lang.items() if k in selected_languages}
            progress_bar.progress(40)
            
            # Display file statistics with enhanced cards
            if not files_by_lang:
                st.error("❌ No source files found for selected languages. Try different languages or check the repository.")
            else:
                st.markdown("## 📊 Repository Analysis")
                
                # Create enhanced statistics display
                stats_cols = st.columns(len(files_by_lang) + 1)
                total_files = 0
                
                for i, (lang, files) in enumerate(files_by_lang.items()):
                    with stats_cols[i]:
                        st.metric(
                            label=f"📄 {lang.title()}",
                            value=f"{len(files)}",
                            delta=f"files"
                        )
                        total_files += len(files)
                
                with stats_cols[-1]:
                    st.metric(
                        label="🎯 Total",
                        value=f"{total_files}",
                        delta=f"files across {len(files_by_lang)} languages"
                    )
                
                progress_bar.progress(60)
                
                # Build dependency graph
                status_text.text("🗂️ Building dependency graph...")
                G = build_multi_language_graph(repo_dir, set(selected_languages))
                progress_bar.progress(80)
                
                # Display enhanced graph statistics
                st.markdown("### 📈 Graph Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔗 Nodes", G.number_of_nodes())
                with col2:
                    st.metric("➡️ Edges", G.number_of_edges())
                with col3:
                    node_types = {}
                    for _, data in G.nodes(data=True):
                        node_type = data.get('type', 'unknown')
                        node_types[node_type] = node_types.get(node_type, 0) + 1
                    st.metric("📋 Node Types", len(node_types))
                with col4:
                    density = nx.density(G) if G.number_of_nodes() > 1 else 0
                    st.metric("🌐 Density", f"{density:.3f}")
                
                # Show detailed breakdown in an expandable section
                with st.expander("📈 Detailed Analysis Breakdown"):
                    breakdown_cols = st.columns(len(node_types))
                    for i, (node_type, count) in enumerate(node_types.items()):
                        with breakdown_cols[i]:
                            st.metric(f"📦 {node_type.title()}s", count)
                
                if G.number_of_nodes() == 0:
                    st.warning("⚠️ No analyzable code found. The repository might not contain compatible source files.")
                else:
                    # Render enhanced interactive graph
                    status_text.text("🎨 Rendering interactive visualization...")
                    html_path = render_enhanced_pyvis(
                        G, show_functions, show_classes, min_degree, 
                        physics, height, set(selected_languages), layout_style
                    )
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    if html_path:
                        st.markdown("## 🕸️ Interactive Dependency Graph")
                        
                        # Enhanced legend with better styling
                        with st.expander("📖 Graph Legend & Navigation Tips", expanded=True):
                            legend_col1, legend_col2, legend_col3 = st.columns(3)
                            
                            with legend_col1:
                                st.markdown("""
                                **🎯 Node Types:**
                                - 📦 **Boxes** = Files
                                - 🔵 **Circles** = Functions  
                                - 💎 **Diamonds** = Classes/Structs
                                """)
                            
                            with legend_col2:
                                st.markdown("""
                                **🔗 Edge Types:**
                                - **Dashed lines** = Import/Include relationships
                                - **Solid lines** = Function calls
                                - **Color intensity** = Relationship strength
                                """)
                            
                            with legend_col3:
                                st.markdown("""
                                **🖱️ Interaction:**
                                - **Click & Drag** = Move nodes
                                - **Mouse Wheel** = Zoom in/out
                                - **Hover** = View details
                                - **Double Click** = Focus on node
                                """)
                        
                        # Display the interactive graph with enhanced container
                        graph_html = safe_read_text(Path(html_path))
                        if graph_html:
                            try:
                                components.html(graph_html, height=height + 50, scrolling=True)
                            except Exception as e:
                                st.error(f"Display error: {e}")
                                st.info("Graph generated but cannot display due to encoding issues. Try a different repository.")
                        else:
                            st.error("Failed to load graph visualization")
                        
                        # Enhanced insights section
                        st.markdown("## 🔍 Code Analysis Insights")
                        
                        insights_col1, insights_col2, insights_col3 = st.columns(3)
                        
                        with insights_col1:
                            st.markdown("**🎯 Most Connected Files:**")
                            file_degrees = [(n, G.degree(n)) for n in G.nodes() if n.startswith("file::")]
                            file_degrees.sort(key=lambda x: x[1], reverse=True)
                            
                            for i, (node, degree) in enumerate(file_degrees[:5]):
                                file_name = G.nodes[node].get('path', node.split("::")[-1])
                                language = G.nodes[node].get('language', 'unknown')
                                st.write(f"`{i+1}.` **{file_name}** ({language}) - {degree} connections")
                        
                        with insights_col2:
                            st.markdown("**🌐 Language Distribution:**")
                            lang_counts = {}
                            for _, data in G.nodes(data=True):
                                if data.get('type') == 'file':
                                    lang = data.get('language', 'unknown')
                                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                            
                            for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = (count / sum(lang_counts.values())) * 100
                                st.write(f"**{lang.title()}**: {count} files ({percentage:.1f}%)")
                        
                        with insights_col3:
                            st.markdown("**⚡ Complexity Analysis:**")
                            complexity_scores = []
                            for n, data in G.nodes(data=True):
                                if data.get('type') == 'file':
                                    complexity = data.get('complexity', 0)
                                    if complexity > 0:
                                        complexity_scores.append((n, complexity))
                            
                            complexity_scores.sort(key=lambda x: x[1], reverse=True)
                            for i, (node, complexity) in enumerate(complexity_scores[:5]):
                                file_name = G.nodes[node].get('path', node.split("::")[-1])
                                st.write(f"`{i+1}.` **{file_name}** - {complexity} calls")

        except GitCommandError as e:
            st.error(f"❌ Failed to clone repository: {e}")
            st.markdown("""
            **Common issues:**
            - Repository URL is incorrect
            - Repository is private (try a public repository)
            - Network connection issues
            - Repository is too large or empty
            """)
        
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            st.write("Please try again or contact support if the issue persists.")
        
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except:
                pass

# Enhanced information section
st.markdown("---")
st.markdown("## ℹ️ Supported Languages & Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ Fully Supported Languages:")
    supported_langs = {
        "🐍 Python": "Functions, classes, imports, method calls",
        "🟨 JavaScript/TypeScript": "Functions, classes, imports, calls",
        "☕ Java": "Methods, classes, imports, method calls",
        "⚡ C/C++": "Functions, classes, includes, calls",
        "🐹 Go": "Functions, structs, imports, calls"
    }
    
    for lang, features in supported_langs.items():
        st.write(f"**{lang}**: {features}")

with col2:
    st.markdown("### 🔄 Additional Supported Languages:")
    other_langs = {
        "🦀 Rust": "Basic function/struct detection",
        "🐘 PHP": "Function and class detection",
        "💎 Ruby": "Method and class detection", 
        "🎯 Kotlin": "Function and class detection",
        "🧡 Swift": "Function and class detection",
        "📊 R/MATLAB": "Function detection",
        "🐚 Shell Scripts": "Function detection"
    }
    
    for lang, support in other_langs.items():
        st.write(f"**{lang}**: {support}")

# Enhanced usage guide
st.markdown("---")
st.markdown("## 🚀 How to Use Code Compass")

st.markdown("""
### 📋 Step-by-Step Guide:

1. **🔗 Enter a GitHub URL** - Any public repository (private repos not supported)
2. **🔤 Select Languages** - Choose which programming languages to analyze  
3. **⚙️ Configure Options** - Adjust display settings, filters, and layout
4. **🚀 Click Analyze** - Wait for the analysis to complete (may take a few minutes for large repos)
5. **🕸️ Explore the Graph** - Interactive visualization with:
   - 🖱️ **Pan & Zoom** - Navigate around the graph
   - 🎯 **Click Nodes** - See detailed information
   - 🔍 **Hover** - Quick previews
   - ⚡ **Physics** - Dynamic node positioning
   - 🎨 **Enhanced Colors** - Language-specific color coding

### 🎯 Pro Tips:
- Start with smaller repositories for faster analysis
- Use filters to focus on specific parts of the codebase
- Try different layout styles for better visualization
- Hover over edges to see relationship details
""")

# Example repositories with enhanced presentation
with st.expander("🌟 Try These Example Repositories"):
    st.markdown("### Popular Open Source Projects:")
    
    examples = {
        "🖥️ Microsoft VS Code": "https://github.com/microsoft/vscode",
        "⚛️ Facebook React": "https://github.com/facebook/react", 
        "🤖 Google TensorFlow": "https://github.com/tensorflow/tensorflow",
        "☸️ Kubernetes": "https://github.com/kubernetes/kubernetes",
        "🐍 Django": "https://github.com/django/django",
        "🟢 Node.js": "https://github.com/nodejs/node",
        "🔺 Angular": "https://github.com/angular/angular",
        "🎮 Godot Engine": "https://github.com/godotengine/godot"
    }
    
    cols = st.columns(2)
    for i, (name, repo_url) in enumerate(examples.items()):
        with cols[i % 2]:
            st.code(repo_url, language="text")
            if st.button(f"Analyze {name}", key=f"example_{name}"):
                st.rerun()

# Footer with enhanced styling
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <h3>🧭 Multi-Language Code Compass</h3>
    <p><em>Discover the architecture of any codebase with AI-powered visual analysis</em></p>
    <p>Made with ❤️ by Anzar Ahmad</p>
</div>
""", unsafe_allow_html=True)
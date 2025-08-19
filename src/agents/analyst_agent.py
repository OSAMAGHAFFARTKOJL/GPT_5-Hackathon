from typing import Dict, Any, List, Optional
import subprocess
import json
from pathlib import Path
import ast
import numpy as np

from .base_agent import BaseAgent

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    RandomForestClassifier = None
    TfidfVectorizer = None

class AnalystAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Analyst", config)
        if RandomForestClassifier is None or TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for AnalystAgent")
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.bug_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self._initialize_models()
    
    def _initialize_models(self):
        # In production, load pre-trained models
        # For now, create dummy training data
        dummy_features = np.random.rand(100, 1000)
        dummy_labels = np.random.randint(0, 2, 100)
        self.bug_predictor.fit(dummy_features, dummy_labels)
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        temp_dir = state.get("temp_dir")
        code_structure = state.get("code_structure")
        
        if not temp_dir or not code_structure:
            raise ValueError("Repository code structure is required")
        
        self.log_execution("Starting code analysis")
        
        # Run static analysis
        static_analysis = await self._run_static_analysis(temp_dir)
        
        # Detect code smells
        code_smells = await self._detect_code_smells(code_structure)
        
        # Predict bug-prone areas
        bug_predictions = await self._predict_bug_prone_areas(code_structure)
        
        # Generate contribution suggestions
        suggestions = await self._generate_contribution_suggestions(
            static_analysis, code_smells, bug_predictions
        )
        
        analysis_results = {
            "static_analysis": static_analysis,
            "code_smells": code_smells,
            "bug_predictions": bug_predictions,
            "contribution_suggestions": suggestions,
            "quality_score": await self._calculate_quality_score(
                static_analysis, code_smells
            )
        }
        
        state.update({"analysis_results": analysis_results})
        
        self.log_execution("Code analysis completed")
        return state
    
    async def _run_static_analysis(self, repo_path: str) -> Dict[str, Any]:
        results = {
            "pylint_results": [],
            "complexity_issues": [],
            "security_issues": []
        }
        
        try:
            # Run pylint on Python files
            python_files = list(Path(repo_path).rglob("*.py"))
            
            for py_file in python_files[:10]:  # Limit for demo
                if "/.git/" in str(py_file) or "__pycache__" in str(py_file):
                    continue
                
                try:
                    # Simplified pylint-like analysis
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    issues = self._analyze_python_file(content, str(py_file))
                    if issues:
                        results["pylint_results"].extend(issues)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {py_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Static analysis failed: {e}")
        
        return results
    
    def _analyze_python_file(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        issues = []
        
        try:
            tree = ast.parse(content)
            
            # Check for common code smells
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Long function detection
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        length = node.end_lineno - node.lineno
                        if length > 50:
                            issues.append({
                                "type": "long_function",
                                "severity": "warning",
                                "message": f"Function '{node.name}' is too long ({length} lines)",
                                "line": node.lineno,
                                "file": file_path
                            })
                    
                    # Too many parameters
                    if len(node.args.args) > 6:
                        issues.append({
                            "type": "too_many_parameters",
                            "severity": "warning",
                            "message": f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                            "line": node.lineno,
                            "file": file_path
                        })
                
                elif isinstance(node, ast.ClassDef):
                    # Large class detection
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > 20:
                        issues.append({
                            "type": "large_class",
                            "severity": "warning",
                            "message": f"Class '{node.name}' has too many methods ({len(methods)})",
                            "line": node.lineno,
                            "file": file_path
                        })
        
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "severity": "error",
                "message": f"Syntax error: {e.msg}",
                "line": e.lineno,
                "file": file_path
            })
        
        return issues
    
    async def _detect_code_smells(self, code_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        smells = []
        
        # Analyze function complexity and coupling
        for file_info in code_structure.get("files", []):
            file_path = file_info.get("path", "")
            
            # Detect duplicate code patterns
            functions = file_info.get("functions", [])
            if len(functions) > 10:
                smells.append({
                    "type": "feature_envy",
                    "severity": "medium",
                    "description": f"File {file_path} has many functions, consider splitting",
                    "location": file_path,
                    "suggestion": "Consider breaking this file into smaller modules"
                })
            
            # Check for missing documentation
            for func in functions:
                if len(func.get("args", [])) > 3:
                    smells.append({
                        "type": "long_parameter_list",
                        "severity": "low",
                        "description": f"Function {func['name']} has many parameters",
                        "location": f"{file_path}:{func.get('line', 0)}",
                        "suggestion": "Consider using a parameter object or breaking down the function"
                    })
        
        return smells
    
    async def _predict_bug_prone_areas(self, code_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions = []
        
        for file_info in code_structure.get("files", []):
            file_path = file_info.get("path", "")
            
            # Simple heuristics for bug prediction
            functions = file_info.get("functions", [])
            classes = file_info.get("classes", [])
            
            # Files with many functions/classes might be more complex
            complexity_score = len(functions) + len(classes) * 2
            
            if complexity_score > 15:
                risk_score = min(0.9, complexity_score / 20.0)
                predictions.append({
                    "file": file_path,
                    "risk_score": float(risk_score),  # Ensure JSON serializable
                    "complexity_score": int(complexity_score),  # Ensure JSON serializable
                    "reason": f"High complexity: {len(functions)} functions, {len(classes)} classes",
                    "recommendation": "Consider refactoring to reduce complexity"
                })
        
        # Sort by risk score
        predictions.sort(key=lambda x: x["risk_score"], reverse=True)
        return predictions[:10]
    
    async def _generate_contribution_suggestions(
        self, 
        static_analysis: Dict[str, Any], 
        code_smells: List[Dict[str, Any]], 
        bug_predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        suggestions = []
        
        # Generate suggestions from static analysis
        for issue in static_analysis.get("pylint_results", [])[:5]:
            suggestions.append({
                "type": "fix_static_issue",
                "difficulty": "beginner" if issue["type"] in ["long_function", "too_many_parameters"] else "intermediate",
                "description": f"Fix {issue['type']}: {issue['message']}",
                "location": f"{issue['file']}:{issue['line']}",
                "estimated_effort": "30 minutes",
                "impact": "code_quality"
            })
        
        # Generate suggestions from code smells
        for smell in code_smells[:3]:
            difficulty = "intermediate" if smell["severity"] == "high" else "beginner"
            suggestions.append({
                "type": "refactor_code_smell",
                "difficulty": difficulty,
                "description": f"Refactor {smell['type']}: {smell['description']}",
                "location": smell["location"],
                "estimated_effort": "1-2 hours",
                "impact": "maintainability"
            })
        
        # Generate suggestions from bug predictions
        for prediction in bug_predictions[:2]:
            suggestions.append({
                "type": "improve_reliability",
                "difficulty": "intermediate",
                "description": f"Add tests for high-risk file: {prediction['file']}",
                "location": prediction["file"],
                "estimated_effort": "2-3 hours",
                "impact": "reliability"
            })
        
        return suggestions
    
    async def _calculate_quality_score(
        self, 
        static_analysis: Dict[str, Any], 
        code_smells: List[Dict[str, Any]]
    ) -> float:
        
        # Simple quality scoring
        base_score = 100.0
        
        # Deduct points for issues
        error_count = len([i for i in static_analysis.get("pylint_results", []) 
                          if i["severity"] == "error"])
        warning_count = len([i for i in static_analysis.get("pylint_results", []) 
                            if i["severity"] == "warning"])
        
        base_score -= error_count * 10
        base_score -= warning_count * 2
        base_score -= len(code_smells) * 1
        
        return max(0.0, min(100.0, base_score))
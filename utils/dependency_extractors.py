import ast
import re
import os

def extract_python_dependencies(content, file_path):
    """Extract dependencies from Python files"""
    dependencies = {
        "imports": [],
        "functions": [],
        "classes": [],
        "function_calls": [],
        "file_references": []
    }
    
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies["imports"].append(node.module)
            
            elif isinstance(node, ast.FunctionDef):
                dependencies["functions"].append(node.name)
            
            elif isinstance(node, ast.ClassDef):
                dependencies["classes"].append(node.name)
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dependencies["function_calls"].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    dependencies["function_calls"].append(node.func.attr)
    
    except Exception as e:
        dependencies.update(extract_with_regex(content, "python"))
    
    return dependencies

def extract_javascript_dependencies(content, file_path):
    """Extract dependencies from JavaScript/TypeScript files"""
    dependencies = {
        "imports": [],
        "functions": [],
        "classes": [],
        "function_calls": [],
        "file_references": []
    }
    
    import_patterns = [
        r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
        r"import\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        r"import\(\s*['\"]([^'\"]+)['\"]\s*\)"
    ]
    
    for pattern in import_patterns:
        matches = re.findall(pattern, content)
        dependencies["imports"].extend(matches)
    
    func_patterns = [
        r"function\s+(\w+)\s*\(",
        r"const\s+(\w+)\s*=\s*function\s*\(",
        r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>",
        r"(\w+)\s*:\s*function\s*\(",
        r"(\w+)\s*:\s*\([^)]*\)\s*=>"
    ]
    
    for pattern in func_patterns:
        matches = re.findall(pattern, content)
        dependencies["functions"].extend(matches)
    
    class_matches = re.findall(r"class\s+(\w+)", content)
    dependencies["classes"].extend(class_matches)
    
    call_matches = re.findall(r"(\w+)\s*\(", content)
    dependencies["function_calls"].extend(call_matches)
    
    return dependencies

def extract_html_dependencies(content, file_path):
    """Extract dependencies from HTML files"""
    dependencies = {
        "css_links": [],
        "js_links": [],
        "image_links": [],
        "other_links": [],
        "imports": [],
        "file_references": []
    }
    
    css_matches = re.findall(r'<link[^>]*href=["\']([^"\']+\.css)["\']', content, re.IGNORECASE)
    dependencies["css_links"].extend(css_matches)
    
    js_matches = re.findall(r'<script[^>]*src=["\']([^"\']+\.js[^"\']*)["\']', content, re.IGNORECASE)
    dependencies["js_links"].extend(js_matches)
    
    img_matches = re.findall(r'<img[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE)
    dependencies["image_links"].extend(img_matches)
    
    asset_matches = re.findall(r'href=["\']([^"\']+\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot))["\']', content, re.IGNORECASE)
    dependencies["other_links"].extend([match[0] for match in asset_matches])
    
    return dependencies

def extract_css_dependencies(content, file_path):
    """Extract dependencies from CSS files"""
    dependencies = {
        "imports": [],
        "url_references": [],
        "file_references": []
    }
    
    import_matches = re.findall(r'@import\s+["\']([^"\']+)["\']', content)
    dependencies["imports"].extend(import_matches)
    
    url_matches = re.findall(r'url\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)', content)
    dependencies["url_references"].extend(url_matches)
    
    return dependencies

def extract_with_regex(content, file_type):
    """Fallback regex extraction for when AST parsing fails"""
    dependencies = {"imports": [], "functions": [], "function_calls": []}
    
    if file_type == "python":
        import_matches = re.findall(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
        for match in import_matches:
            dependencies["imports"].extend([m for m in match if m])
        
        func_matches = re.findall(r'def\s+(\w+)\s*\(', content)
        dependencies["functions"].extend(func_matches)
    
    return dependencies
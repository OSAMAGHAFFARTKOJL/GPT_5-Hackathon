"""
Test script to validate the project structure without dependencies
"""
import os
import sys
from pathlib import Path

def test_project_structure():
    """Test that all expected files and directories exist"""
    
    project_root = Path(__file__).parent
    print(f"Testing project structure in: {project_root}")
    
    # Expected directories
    expected_dirs = [
        "src",
        "src/agents",
        "src/api", 
        "src/orchestrator",
        "src/data",
        "src/ui",
        "src/utils",
        "src/models",
        "tests",
        "config",
        "docs"
    ]
    
    # Expected files
    expected_files = [
        "requirements.txt",
        "setup.py",
        "main.py",
        "run_streamlit.py",
        ".env.example",
        "src/__init__.py",
        "src/agents/__init__.py",
        "src/agents/base_agent.py",
        "src/agents/mapper_agent.py",
        "src/agents/navigator_agent.py",
        "src/agents/analyst_agent.py",
        "src/api/__init__.py",
        "src/api/main.py",
        "src/orchestrator/__init__.py",
        "src/orchestrator/state_graph.py",
        "src/data/__init__.py",
        "src/data/knowledge_graph.py",
        "src/data/vector_store.py",
        "src/ui/__init__.py",
        "src/ui/streamlit_app.py",
        "src/utils/__init__.py",
        "src/utils/github_client.py"
    ]
    
    print("\n=== Testing Directory Structure ===")
    missing_dirs = []
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"[OK] {dir_path}")
        else:
            print(f"[MISSING] {dir_path}")
            missing_dirs.append(dir_path)
    
    print(f"\n=== Testing File Structure ===")
    missing_files = []
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            missing_files.append(file_path)
    
    print(f"\n=== Testing Import Structure (without dependencies) ===")
    
    # Add src to path
    sys.path.insert(0, str(project_root / "src"))
    
    import_tests = [
        ("agents.base_agent", "BaseAgent"),
        ("data.knowledge_graph", "KnowledgeGraphBuilder"),
        ("data.vector_store", "VectorStore"),
        ("utils.github_client", "GitHubClient")
    ]
    
    import_failures = []
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"[OK] {module_name}.{class_name}")
        except Exception as e:
            print(f"[FAILED] {module_name}.{class_name}: {e}")
            import_failures.append((module_name, class_name, str(e)))
    
    print(f"\n=== Summary ===")
    print(f"Missing directories: {len(missing_dirs)}")
    print(f"Missing files: {len(missing_files)}")  
    print(f"Import failures: {len(import_failures)}")
    
    if not missing_dirs and not missing_files and not import_failures:
        print("[SUCCESS] All structure tests passed!")
        return True
    else:
        print("[FAILED] Some tests failed. See details above.")
        return False

def test_requirements_file():
    """Test that requirements.txt has all necessary packages"""
    print("\n=== Testing Requirements File ===")
    
    required_packages = [
        "fastapi",
        "streamlit", 
        "networkx",
        "numpy",
        "requests",
        "pydantic"
    ]
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        
        missing_packages = []
        for package in required_packages:
            if package.lower() not in content:
                missing_packages.append(package)
                print(f"[MISSING] {package}")
            else:
                print(f"[OK] {package}")
        
        if not missing_packages:
            print("[SUCCESS] All required packages found in requirements.txt")
            return True
        else:
            print(f"[FAILED] Missing packages: {missing_packages}")
            return False
            
    except FileNotFoundError:
        print("[FAILED] requirements.txt not found")
        return False

if __name__ == "__main__":
    print("Code Compass Project Structure Test")
    print("=" * 50)
    
    structure_ok = test_project_structure()
    requirements_ok = test_requirements_file()
    
    if structure_ok and requirements_ok:
        print("\n[SUCCESS] PROJECT STRUCTURE VALIDATION SUCCESSFUL!")
        print("Ready to install dependencies and run the application.")
        print("\nNext steps:")
        print("1. pip install -r requirements.txt")
        print("2. python main.py  (to start API server)")
        print("3. python run_streamlit.py  (to start UI)")
    else:
        print("\n[FAILED] Project structure validation failed.")
        print("Please fix the issues listed above.")
    
    sys.exit(0 if structure_ok and requirements_ok else 1)
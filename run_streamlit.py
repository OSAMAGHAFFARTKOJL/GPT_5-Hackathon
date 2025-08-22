"""
Script to run the Streamlit frontend
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit application"""
    # Add src directory to Python path
    # src_path = Path(__file__).parent / "src"
    src_path = Path(__file__).parent / "Anzar_Module"
    os.environ["PYTHONPATH"] = str(src_path)
    
    # Run Streamlit app
    streamlit_file = src_path / "app.py"
    # streamlit_file = src_path / "ui" / "str"
    # "eamlit_app.py"
    
    if not streamlit_file.exists():
        print(f"Error: Streamlit app not found at {streamlit_file}")
        return 1
    
    print("Starting Code Compass UI...")
    print("Navigate to http://localhost:8501 in your browser")
    print("Make sure the API server is running on http://localhost:8000")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_file),
            "--server.port=8501",
            # "--server.address=0.0.0.0"
              "--server.address=localhost"  # Changed from 0.0.0.0 to localhost
        ])
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
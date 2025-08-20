"""
Main entry point for Code Compass application
"""
import os
import sys
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('codecompass.log')
        ]
    )

def main():
    """Main application entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Code Compass application")
    
    # Import after adding src to path
    try:
        from api.main import app
        import uvicorn
        
        # Start the FastAPI server
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
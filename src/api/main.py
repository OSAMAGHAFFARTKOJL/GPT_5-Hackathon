
# from dotenv import load_dotenv
# load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

try:
    from orchestrator.state_graph import CodeCompassOrchestrator
    from utils.json_utils import clean_response_data
except ImportError:
    from src.orchestrator.state_graph import CodeCompassOrchestrator
    from src.utils.json_utils import clean_response_data

app = FastAPI(title="Code Compass API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RepoRequest(BaseModel):
    repo_url: str
    github_token: Optional[str] = None

class QueryRequest(BaseModel):
    repo_url: str
    query: str

class AnalysisRequest(BaseModel):
    repo_url: str

orchestrator = CodeCompassOrchestrator()

@app.post("/map-repo")
async def map_repository(request: RepoRequest):
    try:
        result = await orchestrator.map_repository(request.repo_url, request.github_token)
        clean_result = clean_response_data(result)
        return {"status": "success", "data": clean_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_codebase(request: QueryRequest):
    try:
        result = await orchestrator.query_codebase(request.repo_url, request.query)
        clean_result = clean_response_data(result)
        return {"status": "success", "data": clean_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_codebase(request: AnalysisRequest):
    try:
        result = await orchestrator.analyze_codebase(request.repo_url)
        clean_result = clean_response_data(result)
        return {"status": "success", "data": clean_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
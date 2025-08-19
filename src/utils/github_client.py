import requests
from typing import List, Dict, Any, Optional
import logging
import os

class GitHubClient:
    """Client for interacting with GitHub API"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CodeCompass/1.0"
        }
        
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
    
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get basic repository information"""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to get repository info: {e}")
            raise
    
    async def get_beginner_issues(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get beginner-friendly issues from a repository"""
        # Search for issues with beginner-friendly labels
        labels = [
            "good first issue",
            "beginner friendly", 
            "help wanted",
            "easy",
            "starter",
            "newcomer"
        ]
        
        issues = []
        
        for label in labels:
            try:
                url = f"{self.base_url}/repos/{owner}/{repo}/issues"
                params = {
                    "labels": label,
                    "state": "open",
                    "sort": "created",
                    "direction": "desc",
                    "per_page": 10
                }
                
                response = requests.get(url, headers=self.headers, params=params)
                if response.status_code == 200:
                    label_issues = response.json()
                    
                    for issue in label_issues:
                        if not any(existing["id"] == issue["id"] for existing in issues):
                            issues.append({
                                "id": issue["id"],
                                "title": issue["title"],
                                "url": issue["html_url"],
                                "body": issue["body"][:200] + "..." if issue["body"] and len(issue["body"]) > 200 else issue["body"],
                                "labels": [label["name"] for label in issue["labels"]],
                                "created_at": issue["created_at"],
                                "comments": issue["comments"]
                            })
                            
                            if len(issues) >= 10:  # Limit total issues
                                break
                    
                    if len(issues) >= 10:
                        break
                        
            except requests.RequestException as e:
                self.logger.warning(f"Failed to fetch issues for label '{label}': {e}")
                continue
        
        # Sort by creation date (newest first)
        issues.sort(key=lambda x: x["created_at"], reverse=True)
        return issues[:10]
    
    async def get_repository_structure(self, owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """Get repository file structure"""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            contents = response.json()
            if isinstance(contents, list):
                return contents
            else:
                return [contents]  # Single file
                
        except requests.RequestException as e:
            self.logger.error(f"Failed to get repository structure: {e}")
            return []
    
    async def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Get content of a specific file"""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            content_data = response.json()
            if content_data.get("encoding") == "base64":
                import base64
                return base64.b64decode(content_data["content"]).decode("utf-8")
            else:
                return content_data.get("content", "")
                
        except requests.RequestException as e:
            self.logger.error(f"Failed to get file content: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to decode file content: {e}")
            return None
    
    async def search_code(self, query: str, owner: str = None, repo: str = None) -> List[Dict[str, Any]]:
        """Search for code in repositories"""
        url = f"{self.base_url}/search/code"
        
        search_query = query
        if owner and repo:
            search_query += f" repo:{owner}/{repo}"
        
        params = {
            "q": search_query,
            "sort": "indexed",
            "per_page": 20
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            results = response.json()
            items = results.get("items", [])
            
            return [{
                "name": item["name"],
                "path": item["path"],
                "repository": item["repository"]["full_name"],
                "url": item["html_url"],
                "score": item["score"]
            } for item in items]
            
        except requests.RequestException as e:
            self.logger.error(f"Code search failed: {e}")
            return []
    
    async def get_commits(self, owner: str, repo: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commits from a repository"""
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        
        params = {
            "per_page": limit,
            "page": 1
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            commits = response.json()
            return [{
                "sha": commit["sha"][:8],
                "message": commit["commit"]["message"].split("\n")[0],  # First line only
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "url": commit["html_url"]
            } for commit in commits]
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to get commits: {e}")
            return []
    
    def extract_owner_repo(self, github_url: str) -> tuple[str, str]:
        """Extract owner and repo name from GitHub URL"""
        # Remove .git suffix if present
        url = github_url.rstrip("/").replace(".git", "")
        
        # Handle different GitHub URL formats
        if "github.com/" in url:
            parts = url.split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
        
        raise ValueError(f"Invalid GitHub URL format: {github_url}")
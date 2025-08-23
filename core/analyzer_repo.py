import base64
import os
import re
import streamlit as st
from github import Github
import tempfile
import subprocess


def extract_repo_info(self, url):
    pattern = r"github\.com\/([\w.-]+)\/([\w.-]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_repo_contents(self, username, repo_name, github_token=None):
    try:
        g = Github(github_token) if github_token else Github()
        repo = g.get_repo(f"{username}/{repo_name}")
        
        contents = []
        dirs_to_process = [""]
        
        # Get repository structure
        repo_structure = {"dirs": set(), "files": []}
        
        while dirs_to_process:
            current_dir = dirs_to_process.pop(0)
            try:
                items = repo.get_contents(current_dir)
                for item in items:
                    if item.type == "dir":
                        dirs_to_process.append(item.path)
                        repo_structure["dirs"].add(item.path)
                    elif item.type == "file":
                        try:
                            content = ""
                            if item.size < 1000000:  # Only decode files smaller than 1MB
                                if item.encoding == "base64":
                                    content = base64.b64decode(item.content).decode('utf-8', errors='ignore')
                            
                            file_info = {
                                "name": item.name,
                                "path": item.path,
                                "content": content,
                                "size": item.size,
                                "download_url": item.download_url,
                                "directory": os.path.dirname(item.path)
                            }
                            contents.append(file_info)
                            repo_structure["files"].append(file_info)
                        except Exception as e:
                            st.warning(f"Error reading {item.path}: {str(e)}")
            except Exception as e:
                st.warning(f"Error accessing {current_dir}: {str(e)}")
        
        return contents, repo_structure
    except Exception as e:
        st.error(f"Error accessing repository: {str(e)}")
        return [], {"dirs": set(), "files": []}


def clone_repo(self, repo_url):
    print("⬇ Cloning repository...")
    repo_dir = tempfile.mkdtemp(prefix="repo_")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
    return repo_dir

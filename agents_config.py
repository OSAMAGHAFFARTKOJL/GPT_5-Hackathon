config = {
    # REQUIRED
    "postgres_connection_string": os.getenv("DATABASE_URL"),
    
    # OPTIONAL
    "search_params": {
        "similarity_threshold": 0.3,
        "default_top_k": 10
    },
    "github": {
        "token": os.getenv("GITHUB_TOKEN")
    }
}
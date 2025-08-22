# debug_env_vars.py
import os
from dotenv import load_dotenv

# Debug script to check environment variables
print("=== Environment Variables Debug ===")

# Try to load .env file
try:
    load_dotenv()
    print("✓ .env file loaded successfully")
except Exception as e:
    print(f"✗ Failed to load .env file: {e}")

# Check if .env file exists
env_file = ".env"
if os.path.exists(env_file):
    print(f"✓ .env file exists at: {os.path.abspath(env_file)}")
    
    # Read and display .env content (without sensitive values)
    with open(env_file, 'r') as f:
        lines = f.readlines()
    print(f"✓ .env file has {len(lines)} lines")
    
    for i, line in enumerate(lines, 1):
        if line.strip() and not line.startswith('#'):
            key = line.split('=')[0]
            print(f"  Line {i}: {key}=...")
else:
    print(f"✗ .env file not found at: {os.path.abspath(env_file)}")

# Check environment variables
env_vars = ["DATABASE_URL", "GITHUB_TOKEN", "LOG_LEVEL"]
print("\n=== Environment Variable Values ===")

for var in env_vars:
    value = os.getenv(var)
    if value:
        # Show only first 10 chars for security
        masked_value = value[:10] + "..." if len(value) > 10 else value
        print(f"✓ {var}: {masked_value}")
    else:
        print(f"✗ {var}: Not set")

# Test the config
print("\n=== Config Test ===")
config = {
    "postgres_connection_string": os.getenv("DATABASE_URL"),
    "search_params": {
        "similarity_threshold": 0.3,
        "default_top_k": 10
    },
    "github": {
        "token": os.getenv("GITHUB_TOKEN")
    }
}

print(f"postgres_connection_string: {'✓ Set' if config['postgres_connection_string'] else '✗ None'}")
print(f"github token: {'✓ Set' if config['github']['token'] else '✗ None'}")
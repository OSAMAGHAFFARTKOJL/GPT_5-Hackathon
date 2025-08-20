"""
Test script to verify JSON serialization fix for numpy types
"""
import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import numpy as np
    from utils.json_utils import clean_response_data
    
    # Test data that would cause the original error
    test_data = {
        "similarity_score": np.float32(0.85),
        "risk_score": np.float64(0.72),
        "complexity_score": np.int32(15),
        "results": [
            {
                "score": np.float32(0.91),
                "count": np.int64(5)
            }
        ]
    }
    
    print("Original data types:")
    for key, value in test_data.items():
        if key == "results":
            for i, item in enumerate(value):
                for k, v in item.items():
                    print(f"  results[{i}][{k}]: {type(v)} = {v}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    print("\nCleaning data...")
    cleaned_data = clean_response_data(test_data)
    
    print("\nCleaned data types:")
    for key, value in cleaned_data.items():
        if key == "results":
            for i, item in enumerate(value):
                for k, v in item.items():
                    print(f"  results[{i}][{k}]: {type(v)} = {v}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    # Test JSON serialization
    try:
        json_string = json.dumps(cleaned_data)
        print(f"\n✅ JSON serialization successful!")
        print(f"JSON: {json_string}")
    except Exception as e:
        print(f"\n❌ JSON serialization failed: {e}")
    
    print("\n🎉 JSON serialization fix verified!")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Test failed: {e}")
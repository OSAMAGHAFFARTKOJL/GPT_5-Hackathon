"""
Utility functions for JSON serialization of numpy and other types
"""
import numpy as np
from typing import Any, Dict, List, Union

def make_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON serializable objects to JSON serializable types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def clean_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean response data to ensure all values are JSON serializable
    """
    return make_json_serializable(data)
# src/tof_ml/utils/json_utils.py
import json
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy data types.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return super().default(obj)


def serialize_numpy(obj: Any) -> Any:
    """
    Convert numpy types to standard Python types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return [serialize_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize obj to a JSON formatted string with NumPy support.

    Args:
        obj: The object to serialize
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)


def json_dump(obj: Any, fp, **kwargs) -> None:
    """
    Serialize obj to a JSON formatted file with NumPy support.

    Args:
        obj: The object to serialize
        fp: File-like object
        **kwargs: Additional arguments to pass to json.dump
    """
    json.dump(obj, fp, cls=NumpyEncoder, **kwargs)


def json_loads(s: str, **kwargs) -> Any:
    """
    Deserialize a JSON string to Python object.

    Args:
        s: JSON string
        **kwargs: Additional arguments to pass to json.loads

    Returns:
        Deserialized Python object
    """
    return json.loads(s, **kwargs)


def json_load(fp, **kwargs) -> Any:
    """
    Deserialize a JSON file to Python object.

    Args:
        fp: File-like object
        **kwargs: Additional arguments to pass to json.load

    Returns:
        Deserialized Python object
    """
    return json.load(fp, **kwargs)
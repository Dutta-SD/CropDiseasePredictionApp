from ast import Dict
from datetime import datetime
from typing import Any


def success_response(o: Any) -> Dict:
    """Converts Given Object to Standard API Response

    Args:
        o (Any): Object to send as response

    Returns:
        Dict: Standard JSON response
    """
    return {
        "message": "Successful",
        "response": o,
        "time": datetime.now(),
    }

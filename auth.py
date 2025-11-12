from fastapi import Header, HTTPException
import os

API_KEY = os.getenv("API_KEY", None)

def check_api_key(x_api_key: str = Header(None)):
    """Simple API key validation"""
    if API_KEY is None:
        return True  # no key required in dev
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

"""
Entrypoint module for uvicorn main:app
Re-exports the FastAPI app instance from server.py
"""
import os
import uvicorn
from server import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)

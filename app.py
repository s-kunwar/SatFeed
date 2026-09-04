"""Runnable compatibility entry point for the SatFeed FastAPI service."""

import os

from src.core.inference import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
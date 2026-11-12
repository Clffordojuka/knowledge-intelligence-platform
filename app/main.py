from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="An AI-powered knowledge intelligence platform.",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    """Return the application health status."""

    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }

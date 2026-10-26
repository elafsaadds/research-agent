from fastapi import FastAPI
from app.api import run_research_request


app = FastAPI(
    title="Research Agent API",
    description="Evidence-grounded Agentic AI Research System",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Research Agent API is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/research")
def research(question: str):
    return run_research_request(question)
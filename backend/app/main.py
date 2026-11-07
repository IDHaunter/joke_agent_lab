from fastapi import FastAPI

from backend.app.api.routes import router

app = FastAPI(
    title="Joke Agent Lab",
    description="A small learning application built with LangChain and LangGraph.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Return backend process health without initializing the LLM runtime."""
    return {"status": "ok"}
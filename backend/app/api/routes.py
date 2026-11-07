import logging

from fastapi import APIRouter, HTTPException

from backend.app.agent.graph import chat as run_agent
from backend.app.agent.graph import reset_chat as reset_agent_chat
from backend.app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ResetRequest,
    ResetResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Agent"])


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        answer = run_agent(
            session_id=request.session_id,
            message=request.message,
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
        )
    except Exception as exc:
        logger.exception("Agent execution failed for session %s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail="The agent could not process this message. Check backend logs.",
        ) from exc


@router.post("/reset", response_model=ResetResponse)
def reset_endpoint(request: ResetRequest) -> ResetResponse:
    try:
        reset_agent_chat(request.session_id)
        return ResetResponse(
            session_id=request.session_id,
            status="reset",
        )
    except Exception as exc:
        logger.exception("Conversation reset failed for session %s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail="The conversation could not be reset. Check backend logs.",
        ) from exc
"""POST /chat and GET /chat/status (T-058): local Ollama tool-calling chat. See services/chat_service.py."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import chat_service

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatContext(BaseModel):
    track_id: str | None = Field(default=None, max_length=64)
    race: str | None = Field(default=None, max_length=100)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    context: ChatContext | None = None


class ToolCallOut(BaseModel):
    name: str
    args: dict
    result_summary: str


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallOut]
    caveats: list[str]


class ChatStatus(BaseModel):
    available: bool
    ollama_reachable: bool
    model: str
    model_pulled: bool
    url: str
    message: str


@router.get("/chat/status", response_model=ChatStatus)
def chat_status() -> dict:
    return chat_service.status()


@router.post("/chat", response_model=ChatResponse)
def chat_route(req: ChatRequest) -> dict:
    try:
        return chat_service.chat([m.model_dump() for m in req.messages],
                                 req.context.model_dump() if req.context else None)
    except chat_service.OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

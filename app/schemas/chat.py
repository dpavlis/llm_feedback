from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    id: str
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str
    timestamp: str
    feedback: Optional[dict] = None
    generation_ms: Optional[int] = None


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    conversation_id: str
    message: str = Field(..., min_length=1, max_length=10000)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0)


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    conversation_id: str
    message_id: str
    response: str
    timestamp: str
    generation_ms: Optional[int] = None


class ConversationInfo(BaseModel):
    """Basic info about a conversation for listing."""

    conversation_id: str
    created_at: str
    message_count: int
    token_count: int = 0
    token_breakdown: dict[str, int] = Field(default_factory=dict)
    preview: str = ""


class ConversationDetail(BaseModel):
    """Full conversation with all messages."""

    conversation_id: str
    created_at: str
    messages: list[Message]


class FeedbackRequest(BaseModel):
    """Request to submit feedback on a message."""

    message_id: str
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, max_length=2000)
    preferred_response: Optional[str] = Field(
        None, max_length=10000, description="What would be a better response"
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""

    success: bool
    message: str


class NewConversationRequest(BaseModel):
    """Request to create a new conversation."""

    user_name: Optional[str] = Field(None, max_length=100)


class NewConversationResponse(BaseModel):
    """Response when creating a new conversation."""

    conversation_id: str
    created_at: str


class ConversationListResponse(BaseModel):
    """Response containing list of conversations."""

    conversations: list[ConversationInfo]


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None

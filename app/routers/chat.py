import logging
import json
import time
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationInfo,
    ConversationDetail,
    FeedbackRequest,
    FeedbackResponse,
    NewConversationRequest,
    NewConversationResponse,
    ConversationListResponse,
    Message,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def get_session_id(request: Request) -> Optional[str]:
    """Extract session ID from cookie."""
    return request.cookies.get(settings.cookie_name)


def set_session_cookie(response: Response, session_id: str) -> None:
    """Set the session cookie on the response."""
    response.set_cookie(
        key=settings.cookie_name,
        value=session_id,
        httponly=True,
        max_age=settings.session_timeout_hours * 3600,
        samesite="lax",
    )


@router.post("/conversations", response_model=NewConversationResponse)
async def create_conversation(
    request: Request,
    response: Response,
    conv_request: Optional[NewConversationRequest] = None,
):
    """Create a new conversation."""
    session_manager = request.app.state.session_manager
    persistence = request.app.state.persistence
    llm_manager = request.app.state.llm_manager

    # Get or create session
    session_id = get_session_id(request)
    session_id, session = await session_manager.get_or_create_session(session_id)
    set_session_cookie(response, session_id)

    # Extract user_name if provided
    user_name = None
    if conv_request and conv_request.user_name:
        user_name = conv_request.user_name

    # Create new conversation
    conversation_id = await session_manager.create_conversation(session_id, user_name=user_name)
    if conversation_id is None:
        raise HTTPException(status_code=500, detail="Failed to create conversation")

    # Get conversation data for persistence
    conversation = await session_manager.get_conversation(session_id, conversation_id)
    created_at = conversation["created_at"]

    # Persist the new conversation
    await persistence.save_conversation(
        conversation_id=conversation_id,
        session_id=session_id,
        created_at=created_at,
        messages=[],
        model_name=llm_manager.model_name,
        user_name=user_name,
    )

    return NewConversationResponse(
        conversation_id=conversation_id,
        created_at=created_at.isoformat(),
    )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(request: Request, response: Response):
    """List all conversations for the current session."""
    session_manager = request.app.state.session_manager
    llm_manager = request.app.state.llm_manager

    session_id = get_session_id(request)
    if not session_id:
        # No session yet, return empty list
        return ConversationListResponse(conversations=[])

    session_id, session = await session_manager.get_or_create_session(session_id)
    set_session_cookie(response, session_id)

    conversations = await session_manager.list_conversations(session_id)

    for conversation in conversations:
        messages = await session_manager.get_messages_for_llm(
            session_id, conversation["conversation_id"]
        )
        try:
            breakdown = llm_manager.count_token_breakdown(messages)
            conversation["token_breakdown"] = breakdown
            conversation["token_count"] = breakdown.get(
                "total", llm_manager.count_tokens(messages)
            )
        except Exception as exc:
            logger.warning(
                "Failed to count tokens for conversation %s: %s",
                conversation["conversation_id"],
                exc,
            )
            conversation["token_count"] = 0
            conversation["token_breakdown"] = {}

    return ConversationListResponse(
        conversations=[ConversationInfo(**conv) for conv in conversations]
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str, request: Request, response: Response
):
    """Get full conversation history."""
    session_manager = request.app.state.session_manager

    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session_id, session = await session_manager.get_or_create_session(session_id)
    set_session_cookie(response, session_id)

    conversation = await session_manager.get_conversation(session_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetail(
        conversation_id=conversation_id,
        created_at=conversation["created_at"].isoformat(),
        messages=[Message(**msg) for msg in conversation["messages"]],
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, request: Request, response: Response
):
    """Delete a conversation from the session (keeps the file on disk)."""
    session_manager = request.app.state.session_manager

    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")

    session_id, session = await session_manager.get_or_create_session(session_id)
    set_session_cookie(response, session_id)

    success = await session_manager.delete_conversation(session_id, conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "message": "Conversation deleted from session"}


def _sse_event(payload: dict) -> str:
    """Serialize one server-sent event data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def send_message(chat_request: ChatRequest, request: Request, response: Response):
    """Send a message and get an LLM response."""
    session_manager = request.app.state.session_manager
    persistence = request.app.state.persistence
    llm_manager = request.app.state.llm_manager

    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session. Create a conversation first.")

    session_id, session = await session_manager.get_or_create_session(session_id)
    set_session_cookie(response, session_id)

    # Verify conversation exists
    conversation = await session_manager.get_conversation(
        session_id, chat_request.conversation_id
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Add user message to session
    await session_manager.add_message(
        session_id,
        chat_request.conversation_id,
        role="user",
        content=chat_request.message,
    )

    # Get conversation history for LLM
    messages = await session_manager.get_messages_for_llm(
        session_id, chat_request.conversation_id
    )

    if chat_request.stream:
        async def event_stream():
            start_time = time.perf_counter()
            response_parts: list[str] = []

            try:
                iterator = llm_manager.stream_response(
                    messages,
                    temperature=chat_request.temperature,
                    top_p=chat_request.top_p,
                    top_k=chat_request.top_k,
                    repetition_penalty=chat_request.repetition_penalty,
                )

                while True:
                    chunk = await asyncio.to_thread(lambda: next(iterator, None))
                    if chunk is None:
                        break
                    response_parts.append(chunk)
                    yield _sse_event({"type": "token", "content": chunk})

                llm_response = "".join(response_parts).strip()
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                assistant_message_id = await session_manager.add_message(
                    session_id,
                    chat_request.conversation_id,
                    role="assistant",
                    content=llm_response,
                    generation_ms=elapsed_ms,
                )

                conversation_data = await session_manager.get_conversation(
                    session_id,
                    chat_request.conversation_id,
                )
                await persistence.save_conversation(
                    conversation_id=chat_request.conversation_id,
                    session_id=session_id,
                    created_at=conversation_data["created_at"],
                    messages=conversation_data["messages"],
                    model_name=llm_manager.model_name,
                    user_name=conversation_data.get("user_name"),
                )

                yield _sse_event(
                    {
                        "type": "done",
                        "conversation_id": chat_request.conversation_id,
                        "message_id": assistant_message_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "generation_ms": elapsed_ms,
                        "response": llm_response,
                    }
                )
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")
                yield _sse_event({"type": "error", "detail": "Failed to generate response"})

        stream_response = StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        set_session_cookie(stream_response, session_id)
        return stream_response

    # Non-streaming fallback
    try:
        start_time = time.perf_counter()
        llm_response = llm_manager.generate_response(
            messages,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            top_k=chat_request.top_k,
            repetition_penalty=chat_request.repetition_penalty,
        )
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

    assistant_message_id = await session_manager.add_message(
        session_id,
        chat_request.conversation_id,
        role="assistant",
        content=llm_response,
        generation_ms=elapsed_ms,
    )

    conversation = await session_manager.get_conversation(
        session_id, chat_request.conversation_id
    )
    await persistence.save_conversation(
        conversation_id=chat_request.conversation_id,
        session_id=session_id,
        created_at=conversation["created_at"],
        messages=conversation["messages"],
        model_name=llm_manager.model_name,
        user_name=conversation.get("user_name"),
    )

    return ChatResponse(
        conversation_id=chat_request.conversation_id,
        message_id=assistant_message_id,
        response=llm_response,
        timestamp=datetime.utcnow().isoformat(),
        generation_ms=elapsed_ms,
    )


@router.post(
    "/conversations/{conversation_id}/feedback", response_model=FeedbackResponse
)
async def submit_feedback(
    conversation_id: str,
    feedback_request: FeedbackRequest,
    request: Request,
    response: Response,
):
    """Submit feedback for a specific message."""
    session_manager = request.app.state.session_manager
    persistence = request.app.state.persistence

    session_id = get_session_id(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session")

    session_id, session = await session_manager.get_or_create_session(session_id)
    set_session_cookie(response, session_id)

    # Verify conversation exists
    conversation = await session_manager.get_conversation(session_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Add feedback to session
    success = await session_manager.add_feedback(
        session_id,
        conversation_id,
        feedback_request.message_id,
        rating=feedback_request.rating,
        comment=feedback_request.comment,
        preferred_response=feedback_request.preferred_response,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Message not found")

    # Update persisted file
    feedback_data = {
        "rating": feedback_request.rating,
        "comment": feedback_request.comment,
        "preferred_response": feedback_request.preferred_response,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    await persistence.add_feedback(
        conversation_id=conversation_id,
        created_at=conversation["created_at"],
        message_id=feedback_request.message_id,
        feedback=feedback_data,
    )

    return FeedbackResponse(success=True, message="Feedback submitted successfully")

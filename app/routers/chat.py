import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException, Depends

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
            conversation["token_count"] = llm_manager.count_tokens(messages)
        except Exception as exc:
            logger.warning(
                "Failed to count tokens for conversation %s: %s",
                conversation["conversation_id"],
                exc,
            )
            conversation["token_count"] = 0

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


@router.post("/chat", response_model=ChatResponse)
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
    user_message_id = await session_manager.add_message(
        session_id,
        chat_request.conversation_id,
        role="user",
        content=chat_request.message,
    )

    # Get conversation history for LLM
    messages = await session_manager.get_messages_for_llm(
        session_id, chat_request.conversation_id
    )

    # Generate response
    try:
        llm_response = llm_manager.generate_response(messages)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # Add assistant message to session
    assistant_message_id = await session_manager.add_message(
        session_id,
        chat_request.conversation_id,
        role="assistant",
        content=llm_response,
    )

    # Persist to file
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

    timestamp = datetime.utcnow().isoformat()

    return ChatResponse(
        conversation_id=chat_request.conversation_id,
        message_id=assistant_message_id,
        response=llm_response,
        timestamp=timestamp,
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

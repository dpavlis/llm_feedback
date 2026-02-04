import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions and conversations in memory.

    Each session (identified by a cookie) can have multiple conversations.
    Sessions automatically expire after the configured timeout.
    """

    def __init__(self, timeout_hours: Optional[int] = None):
        self.sessions: dict[str, dict] = {}
        self.timeout = timedelta(hours=timeout_hours or settings.session_timeout_hours)
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        async with self._lock:
            self.sessions[session_id] = {
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow(),
                "conversations": {},  # conversation_id -> list of messages
            }
        logger.info(f"Created new session: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID, returning None if not found or expired."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None

            # Check if session has expired
            if datetime.utcnow() - session["last_active"] > self.timeout:
                del self.sessions[session_id]
                logger.info(f"Session expired: {session_id}")
                return None

            # Update last active time
            session["last_active"] = datetime.utcnow()
            return session

    async def get_or_create_session(self, session_id: Optional[str]) -> tuple[str, dict]:
        """Get an existing session or create a new one."""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session_id, session

        # Create new session
        new_id = await self.create_session()
        session = self.sessions[new_id]
        return new_id, session

    async def create_conversation(self, session_id: str, user_name: Optional[str] = None) -> Optional[str]:
        """Create a new conversation within a session."""
        session = await self.get_session(session_id)
        if session is None:
            return None

        conversation_id = str(uuid.uuid4())
        async with self._lock:
            session["conversations"][conversation_id] = {
                "created_at": datetime.utcnow(),
                "messages": [],
                "user_name": user_name,
            }
        logger.info(f"Created conversation {conversation_id} in session {session_id}")
        return conversation_id

    async def get_conversation(
        self, session_id: str, conversation_id: str
    ) -> Optional[dict]:
        """Get a specific conversation from a session."""
        session = await self.get_session(session_id)
        if session is None:
            return None

        return session["conversations"].get(conversation_id)

    async def delete_conversation(
        self, session_id: str, conversation_id: str
    ) -> bool:
        """Delete a conversation from the session (not from disk)."""
        session = await self.get_session(session_id)
        if session is None:
            return False

        async with self._lock:
            if conversation_id in session["conversations"]:
                del session["conversations"][conversation_id]
                logger.info(f"Deleted conversation {conversation_id} from session {session_id}")
                return True

        return False

    async def add_message(
        self,
        session_id: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Optional[str]:
        """
        Add a message to a conversation.

        Returns the message ID if successful, None otherwise.
        """
        session = await self.get_session(session_id)
        if session is None:
            return None

        conversation = session["conversations"].get(conversation_id)
        if conversation is None:
            return None

        message_id = str(uuid.uuid4())
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "feedback": None,
        }

        async with self._lock:
            conversation["messages"].append(message)

        return message_id

    async def add_feedback(
        self,
        session_id: str,
        conversation_id: str,
        message_id: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        preferred_response: Optional[str] = None,
    ) -> bool:
        """Add feedback to a specific message."""
        session = await self.get_session(session_id)
        if session is None:
            return False

        conversation = session["conversations"].get(conversation_id)
        if conversation is None:
            return False

        async with self._lock:
            for message in conversation["messages"]:
                if message["id"] == message_id:
                    message["feedback"] = {
                        "rating": rating,
                        "comment": comment,
                        "preferred_response": preferred_response,
                        "submitted_at": datetime.utcnow().isoformat(),
                    }
                    return True

        return False

    async def list_conversations(self, session_id: str) -> list[dict]:
        """List all conversations in a session with basic info."""
        session = await self.get_session(session_id)
        if session is None:
            return []

        conversations = []
        for conv_id, conv_data in session["conversations"].items():
            # Get preview from first message if available
            preview = ""
            if conv_data["messages"]:
                first_msg = conv_data["messages"][0]
                preview = first_msg["content"][:50]
                if len(first_msg["content"]) > 50:
                    preview += "..."

            conversations.append(
                {
                    "conversation_id": conv_id,
                    "created_at": conv_data["created_at"].isoformat(),
                    "message_count": len(conv_data["messages"]),
                    "preview": preview,
                }
            )

        # Sort by creation time, newest first
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        return conversations

    async def get_messages_for_llm(
        self, session_id: str, conversation_id: str
    ) -> list[dict[str, str]]:
        """
        Get conversation messages formatted for LLM input.

        Returns list of dicts with 'role' and 'content' keys only.
        """
        conversation = await self.get_conversation(session_id, conversation_id)
        if conversation is None:
            return []

        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation["messages"]
        ]

    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions. Returns count of removed sessions."""
        now = datetime.utcnow()
        expired = []

        async with self._lock:
            for session_id, session in self.sessions.items():
                if now - session["last_active"] > self.timeout:
                    expired.append(session_id)

            for session_id in expired:
                del self.sessions[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

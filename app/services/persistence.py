import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationPersistence:
    """
    Handles persisting conversations to JSON files.

    Conversations are stored in a date-based directory structure:
    data/conversations/YYYY/MM/DD/conv_{conversation_id}.json
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or settings.data_dir
        self.base_path = Path(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific conversation."""
        async with self._global_lock:
            if conversation_id not in self._write_locks:
                self._write_locks[conversation_id] = asyncio.Lock()
            return self._write_locks[conversation_id]

    def _get_conversation_path(
        self, conversation_id: str, created_at: datetime
    ) -> Path:
        """Get the file path for a conversation."""
        date_path = self.base_path / created_at.strftime("%Y/%m/%d")
        date_path.mkdir(parents=True, exist_ok=True)
        return date_path / f"conv_{conversation_id}.json"

    async def save_conversation(
        self,
        conversation_id: str,
        session_id: str,
        created_at: datetime,
        messages: list[dict],
        model_name: str,
        user_name: Optional[str] = None,
    ) -> Path:
        """
        Save a complete conversation to a JSON file.

        Args:
            conversation_id: Unique conversation identifier
            session_id: Session that owns this conversation
            created_at: When the conversation was created
            messages: List of message dicts
            model_name: Name of the model used
            user_name: Optional name of the user

        Returns:
            Path to the saved file
        """
        lock = await self._get_lock(conversation_id)

        conversation_data = {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "created_at": created_at.isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {
                "model_name": model_name,
                "user_name": user_name,
            },
            "messages": messages,
        }

        file_path = self._get_conversation_path(conversation_id, created_at)

        async with lock:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(conversation_data, indent=2, default=str))

        logger.debug(f"Saved conversation {conversation_id} to {file_path}")
        return file_path

    async def load_conversation(
        self, conversation_id: str, date: datetime
    ) -> Optional[dict]:
        """
        Load a conversation from disk.

        Args:
            conversation_id: The conversation ID
            date: The date the conversation was created

        Returns:
            The conversation data dict, or None if not found
        """
        file_path = self._get_conversation_path(conversation_id, date)

        if not file_path.exists():
            return None

        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def append_message(
        self,
        conversation_id: str,
        created_at: datetime,
        message: dict,
    ) -> bool:
        """
        Append a message to an existing conversation file.

        Returns True if successful, False if conversation not found.
        """
        lock = await self._get_lock(conversation_id)
        file_path = self._get_conversation_path(conversation_id, created_at)

        async with lock:
            if not file_path.exists():
                return False

            # Load existing data
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

            # Append message and update timestamp
            data["messages"].append(message)
            data["updated_at"] = datetime.utcnow().isoformat()

            # Write back
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2, default=str))

        logger.debug(f"Appended message to conversation {conversation_id}")
        return True

    async def add_feedback(
        self,
        conversation_id: str,
        created_at: datetime,
        message_id: str,
        feedback: dict,
    ) -> bool:
        """
        Add feedback to a specific message in a conversation.

        Returns True if successful, False if conversation/message not found.
        """
        lock = await self._get_lock(conversation_id)
        file_path = self._get_conversation_path(conversation_id, created_at)

        async with lock:
            if not file_path.exists():
                return False

            # Load existing data
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

            # Find and update the message
            found = False
            for message in data["messages"]:
                if message.get("id") == message_id:
                    message["feedback"] = feedback
                    found = True
                    break

            if not found:
                return False

            data["updated_at"] = datetime.utcnow().isoformat()

            # Write back
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2, default=str))

        logger.debug(f"Added feedback to message {message_id}")
        return True

    async def list_conversations(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """
        List all saved conversations, optionally filtered by date range.

        Returns a list of conversation metadata (without full messages).
        """
        conversations = []

        # Walk through date directories
        for year_dir in sorted(self.base_path.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue

                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue

                    # Check date filter
                    dir_date = datetime(
                        int(year_dir.name),
                        int(month_dir.name),
                        int(day_dir.name),
                    )
                    if start_date and dir_date < start_date:
                        continue
                    if end_date and dir_date > end_date:
                        continue

                    # Read conversation files
                    for conv_file in sorted(day_dir.glob("conv_*.json")):
                        try:
                            async with aiofiles.open(conv_file, "r") as f:
                                content = await f.read()
                                data = json.loads(content)

                            # Extract metadata only
                            conversations.append(
                                {
                                    "conversation_id": data["conversation_id"],
                                    "session_id": data.get("session_id"),
                                    "created_at": data["created_at"],
                                    "updated_at": data.get("updated_at"),
                                    "message_count": len(data.get("messages", [])),
                                    "model_name": data.get("metadata", {}).get(
                                        "model_name"
                                    ),
                                }
                            )
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Error reading {conv_file}: {e}")
                            continue

        return conversations

    def cleanup_locks(self) -> None:
        """Remove all conversation locks (call during shutdown)."""
        self._write_locks.clear()

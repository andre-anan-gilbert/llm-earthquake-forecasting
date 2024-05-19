"""Groq."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ChatMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str

    class Config:
        use_enum_values = True


class GroqLanguageModel(BaseModel):
    """Class that implements the Groq models."""

    groq_client: Any
    model: str
    max_tokens: int = 256
    temperature: float = 0.0

    def get_completion(self, messages: list[ChatMessage]) -> str:
        """Creates a model response for the given chat conversation.

        Args:
            messages: A list of messages comprising the conversation so far.

        Returns:
            LLM response.
        """
        chat_completion = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in messages],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return chat_completion.choices[0].message.content

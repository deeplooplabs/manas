"""SSE streaming utilities for OpenAI-compatible chat completions.

Since HiRAG's query() is synchronous and returns a complete answer,
streaming is simulated by splitting the answer into word-level chunks
and yielding them as Server-Sent Events (SSE).
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

from hirag_haystack.api.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)


async def generate_stream_response(
    answer: str,
    model: str,
    chunk_delay: float = 0.02,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream from a complete answer.

    Args:
        answer: The complete answer text to stream.
        model: Model name for the response.
        chunk_delay: Delay between chunks in seconds (simulates typing).

    Yields:
        SSE-formatted strings: "data: {json}\n\n"
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # First chunk: role announcement
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant", content=""),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Split answer into words, preserving whitespace
    words = _split_preserving_whitespace(answer)

    # Stream each word as a chunk
    for word in words:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content=word),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Small delay to simulate token-by-token generation
        if chunk_delay > 0:
            await asyncio.sleep(chunk_delay)

    # Final chunk: finish reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"

    # SSE termination
    yield "data: [DONE]\n\n"


def _split_preserving_whitespace(text: str) -> list[str]:
    """Split text into words while preserving leading/trailing whitespace.

    This ensures the reconstructed text matches the original exactly.

    Args:
        text: Text to split.

    Returns:
        List of word tokens with appropriate spacing.
    """
    if not text:
        return []

    result = []
    current_word = ""
    in_whitespace = text[0].isspace() if text else False

    for char in text:
        is_space = char.isspace()
        if is_space == in_whitespace:
            current_word += char
        else:
            if current_word:
                result.append(current_word)
            current_word = char
            in_whitespace = is_space

    if current_word:
        result.append(current_word)

    # Combine words with their following whitespace for better streaming UX
    combined = []
    i = 0
    while i < len(result):
        token = result[i]
        # If this is a word (not whitespace) and next token is whitespace, combine
        if not token[0].isspace() and i + 1 < len(result) and result[i + 1][0].isspace():
            combined.append(token + result[i + 1])
            i += 2
        else:
            combined.append(token)
            i += 1

    return combined if combined else [text]


def format_error_event(error_message: str) -> str:
    """Format an error as an SSE event.

    Args:
        error_message: Error message to send.

    Returns:
        SSE-formatted error string.
    """
    error_data = {
        "error": {
            "message": error_message,
            "type": "server_error",
        }
    }
    return f"data: {json.dumps(error_data)}\n\n"

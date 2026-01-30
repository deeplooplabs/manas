"""Token utilities for HiRAG.

This module provides utilities for working with tokens using tiktoken,
including encoding, decoding, truncation, and counting functions.
"""

import hashlib
import json
import re
from typing import Any, Callable, List, Optional

import tiktoken


# ===== Token Encoding/Decoding =====

def get_tokenizer(model_name: str = "gpt-4o") -> tiktoken.Encoding:
    """Get a tiktoken tokenizer for the specified model.

    Args:
        model_name: Name of the model to get tokenizer for.

    Returns:
        tiktoken.Encoding instance.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base (GPT-4/GPT-3.5-Turbo encoding)
        return tiktoken.get_encoding("cl100k_base")


def encode_string_by_tiktoken(
    text: str,
    model_name: str = "gpt-4o",
) -> List[int]:
    """Encode a string into tokens using tiktoken.

    Args:
        text: The text to encode.
        model_name: Model name for tokenizer selection.

    Returns:
        List of token IDs.
    """
    encoder = get_tokenizer(model_name)
    return encoder.encode(text)


def decode_tokens_by_tiktoken(
    tokens: List[int],
    model_name: str = "gpt-4o",
) -> str:
    """Decode tokens back into text using tiktoken.

    Args:
        tokens: List of token IDs.
        model_name: Model name for tokenizer selection.

    Returns:
        Decoded text string.
    """
    encoder = get_tokenizer(model_name)
    return encoder.decode(tokens)


def count_tokens(
    text: str,
    model_name: str = "gpt-4o",
) -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens in.
        model_name: Model name for tokenizer selection.

    Returns:
        Number of tokens.
    """
    return len(encode_string_by_tiktoken(text, model_name))


def count_tokens_batch(
    texts: List[str],
    model_name: str = "gpt-4o",
) -> List[int]:
    """Count tokens for multiple texts efficiently.

    Args:
        texts: List of texts to count tokens in.
        model_name: Model name for tokenizer selection.

    Returns:
        List of token counts.
    """
    encoder = get_tokenizer(model_name)
    return [len(encoder.encode(text)) for text in texts]


# ===== Token Truncation =====

def truncate_string_by_token_size(
    text: str,
    max_tokens: int,
    model_name: str = "gpt-4o",
) -> str:
    """Truncate a string to fit within max_tokens.

    Args:
        text: The text to truncate.
        max_tokens: Maximum number of tokens allowed.
        model_name: Model name for tokenizer selection.

    Returns:
        Truncated text.
    """
    tokens = encode_string_by_tiktoken(text, model_name)
    if len(tokens) <= max_tokens:
        return text
    return decode_tokens_by_tiktoken(tokens[:max_tokens], model_name)


def truncate_list_by_token_size(
    items: List[Any],
    key: Optional[Callable] = None,
    max_tokens: int = 10000,
    model_name: str = "gpt-4o",
) -> List[Any]:
    """Truncate a list to fit within max_tokens based on a key function.

    Args:
        items: List of items to truncate.
        key: Function to extract text from each item. Defaults to str(item).
        max_tokens: Maximum total tokens allowed.
        model_name: Model name for tokenizer selection.

    Returns:
        Truncated list of items.
    """
    if key is None:
        key = lambda x: str(x)

    result = []
    total_tokens = 0

    for item in items:
        text = key(item)
        tokens = count_tokens(text, model_name)

        if total_tokens + tokens > max_tokens:
            break

        result.append(item)
        total_tokens += tokens

    return result


# ===== String Splitting =====

def split_string_by_multi_markers(
    text: str,
    markers: List[str],
) -> List[str]:
    """Split a string by multiple possible markers.

    Args:
        text: The text to split.
        markers: List of marker strings to split by.

    Returns:
        List of split strings.
    """
    if not markers:
        return [text]

    # Start with the first marker
    result = [text]

    for marker in markers:
        new_result = []
        for part in result:
            new_result.extend(part.split(marker))
        result = new_result

    return [s.strip() for s in result if s.strip()]


def list_of_list_to_csv(
    data: List[List[Any]],
) -> str:
    """Convert a list of lists to CSV format.

    Args:
        data: List of rows (each row is a list of values).

    Returns:
        CSV-formatted string.
    """
    lines = []
    for row in data:
        # Escape quotes and wrap in quotes if contains comma
        escaped_row = []
        for item in row:
            item_str = str(item)
            if "," in item_str or '"' in item_str or "\n" in item_str:
                # Escape existing quotes and wrap in quotes
                item_str = '"' + item_str.replace('"', '""') + '"'
            escaped_row.append(item_str)
        lines.append(",".join(escaped_row))
    return "\n".join(lines)


# ===== Hash Utilities =====

def compute_mdhash_id(
    content: str,
    prefix: str = "",
) -> str:
    """Compute an MD5 hash for content.

    Args:
        content: The content to hash.
        prefix: Optional prefix for the hash ID.

    Returns:
        Hash ID string.
    """
    content_str = str(content)
    hash_obj = hashlib.md5(content_str.encode("utf-8"))
    return f"{prefix}{hash_obj.hexdigest()}"


# ===== Text Cleaning =====

def clean_str(text: str) -> str:
    """Clean and normalize a string.

    Args:
        text: The text to clean.

    Returns:
        Cleaned string.
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def is_float_regex(value: str) -> bool:
    """Check if a string represents a float.

    Args:
        value: The string to check.

    Returns:
        True if the string is a float representation.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# ===== JSON Utilities =====

def convert_response_to_json(
    response: str,
) -> dict:
    """Convert an LLM response to JSON.

    Args:
        response: The LLM response string.

    Returns:
        Parsed JSON dictionary.
    """
    response = response.strip()

    # Try to find JSON in the response
    start = response.find("{")
    end = response.rfind("}") + 1

    if start >= 0 and end > start:
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def pack_user_ass_to_openai_messages(
    user_prompt: str,
    assistant_response: str,
) -> List[dict]:
    """Pack user and assistant messages into OpenAI format.

    Args:
        user_prompt: The user's prompt.
        assistant_response: The assistant's response.

    Returns:
        List of message dictionaries in OpenAI format.
    """
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

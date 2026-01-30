"""OpenAI-compatible API routes.

Provides /v1/models and /v1/chat/completions endpoints that are compatible
with the OpenAI API specification, allowing integration with tools like
Open WebUI, LangChain, and other OpenAI-compatible clients.

Endpoints:
- GET /v1/models - List available models
- POST /v1/chat/completions - Chat completions (streaming + non-streaming)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from hirag_haystack import HiRAG, QueryParam
from hirag_haystack.api.dependencies import get_hirag, run_in_executor
from hirag_haystack.api.models import (
    AVAILABLE_MODELS,
    MODEL_MODE_MAPPING,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ErrorDetail,
    ErrorResponse,
    OpenAIModel,
    OpenAIModelList,
)
from hirag_haystack.api.streaming import generate_stream_response

router = APIRouter(prefix="/v1", tags=["openai"])


@router.get("/models", response_model=OpenAIModelList)
async def list_models() -> OpenAIModelList:
    """List available models.

    Each model maps to a HiRAG retrieval mode:
    - hirag / hirag-hi: Full hierarchical (hi mode)
    - hirag-local: Entity-level only (hi_local mode)
    - hirag-global: Community reports only (hi_global mode)
    - hirag-bridge: Cross-community paths (hi_bridge mode)
    - hirag-nobridge: Local + global without paths (hi_nobridge mode)
    - hirag-naive: Simple chunk retrieval (naive mode)
    """
    models = [OpenAIModel(id=model_id) for model_id in AVAILABLE_MODELS]
    return OpenAIModelList(data=models)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    project_id: str | None = Query(None, description="Project ID for data isolation"),
    hirag: HiRAG = Depends(get_hirag),
):
    """Create a chat completion.

    Supports both streaming and non-streaming responses.

    The model field determines the retrieval mode:
    - hirag, hirag-hi -> hi (full hierarchical)
    - hirag-local -> hi_local
    - hirag-global -> hi_global
    - hirag-bridge -> hi_bridge
    - hirag-nobridge -> hi_nobridge
    - hirag-naive -> naive

    Unknown model names default to 'hi' mode.

    HiRAG-specific extensions can be passed in the request body:
    - top_k: Number of entities to retrieve
    - top_m: Key entities per community
    - response_type: Expected response format

    Use project_id query parameter for multi-project isolation.
    """
    # Extract the last user message as the query
    user_message = _extract_last_user_message(request.messages)
    if not user_message:
        error_response = ErrorResponse(
            error=ErrorDetail(
                message="No user message found in request",
                param="messages",
            )
        )
        raise HTTPException(status_code=400, detail=error_response.model_dump())

    # Map model to retrieval mode
    mode = MODEL_MODE_MAPPING.get(request.model, "hi")

    # Build query parameters
    param = QueryParam(
        mode=mode,
        top_k=request.top_k,
        top_m=request.top_m,
        response_type=request.response_type,
    )

    # Use project_id if provided, else default
    pid = project_id or "default"

    if request.stream:
        # Streaming response
        return StreamingResponse(
            _stream_chat_completion(hirag, user_message, mode, param, request.model, pid),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        try:
            result = await run_in_executor(
                lambda: hirag.query(query=user_message, mode=mode, param=param, project_id=pid)
            )
        except Exception as e:
            error_response = ErrorResponse(
                error=ErrorDetail(
                    message=f"Query failed: {str(e)}",
                    type="server_error",
                )
            )
            raise HTTPException(status_code=500, detail=error_response.model_dump())

        answer = result.get("answer", "")

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=answer),
                )
            ],
        )


async def _stream_chat_completion(
    hirag: HiRAG,
    query: str,
    mode: str,
    param: QueryParam,
    model: str,
    project_id: str = "default",
):
    """Generate streaming chat completion response.

    HiRAG's query() is synchronous, so we:
    1. Run it to completion in a thread executor
    2. Simulate streaming by yielding word-level chunks
    """
    try:
        # Run the full query in executor
        result = await run_in_executor(
            lambda: hirag.query(query=query, mode=mode, param=param, project_id=project_id)
        )
        answer = result.get("answer", "")

        # Stream the answer as SSE events
        async for chunk in generate_stream_response(answer, model):
            yield chunk

    except Exception as e:
        # Yield error as SSE event
        from hirag_haystack.api.streaming import format_error_event
        yield format_error_event(f"Query failed: {str(e)}")


def _extract_last_user_message(messages: list[ChatMessage]) -> str | None:
    """Extract the last user message from the conversation.

    Args:
        messages: List of chat messages.

    Returns:
        Content of the last user message, or None if not found.
    """
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return None

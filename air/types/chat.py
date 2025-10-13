"""
Pydantic models for ChatCompletion responses (non-streaming and streaming),
aligned with OpenAI's wire format and extended for specialized usage.

This module provides:
  • Function                          – Details of a tool/function call (non-streaming)
  • ChatCompletionMessageToolCall     – Tool-call metadata within a non-streaming message
  • ChatCompletionMessage             – One message in a non-streaming conversation
  • Choice                            – A single choice in a non-streaming ChatCompletion
  • CompletionUsage                   – Usage metadata (token counts)
  • ChatCompletion                    – Top-level non-streaming ChatCompletion response

Streaming types (strictly aligned to OpenAI's spec):
  • ChoiceDeltaFunctionCall           – Deprecated function-call delta (kept for compatibility)
  • ChoiceDeltaToolCallFunction       – Function details in a streamed tool-call delta
  • ChoiceDeltaToolCall               – A single tool-call delta in a streamed chunk
  • ChoiceDelta                       – Delta payload for streamed chunks (content/tool_calls/etc.)
  • ChoiceChunk                       – A single streamed choice
  • ChatCompletionChunk               – Top-level streamed chunk (strict schema)
"""

from typing import Any, List, Literal, Optional

from air.types.base import CustomBaseModel


class Function(CustomBaseModel):
    """Details of a function call (non-streaming).

    Attributes:
        name: The name of the function to invoke.
        arguments: A serialized (JSON string) set of arguments for the function call.
    """

    name: str
    arguments: str


class ChatCompletionMessageToolCall(CustomBaseModel):
    """Tool-call metadata for a non-streaming message.

    Attributes:
        id: Unique identifier of the tool call.
        type: The type of tool call (e.g., "function").
        function: Function call details, if provided.
    """

    id: str
    type: str
    function: Optional[Function] = None


class ChatCompletionMessage(CustomBaseModel):
    """One message within a non-streaming conversation.

    Attributes:
        role: Message author role (e.g., "assistant", "user", "system").
        content: Primary text content of the message, if any.
        refusal: A refusal statement, if present.
        annotations: Arbitrary annotations/metadata, if any.
        audio: Audio content or metadata, if any.
        function_call: Direct function-call details, if present.
        reasoning_content: Optional reasoning output (if provided).
        tool_calls: A list of tool-call records used within this message, if any.
    """

    role: str
    content: Optional[str] = None
    refusal: Optional[str] = None
    annotations: Optional[Any] = None
    audio: Optional[Any] = None
    function_call: Optional[Any] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class Choice(CustomBaseModel):
    """A single choice from a non-streaming ChatCompletion.

    Attributes:
        index: Index of this choice within the list of choices.
        finish_reason: Why the model stopped (e.g., "stop", "tool_calls").
        message: The message returned for this choice.
        stop_reason: Optional stop code, if provided.
        logprobs: Optional log-probability metadata, if available.
    """

    index: int
    finish_reason: Optional[str]
    message: ChatCompletionMessage
    stop_reason: Optional[int] = None
    logprobs: Optional[Any] = None


class CompletionUsage(CustomBaseModel):
    """Token usage details for a (non-streaming) ChatCompletion or final streaming chunk.

    Attributes:
        prompt_tokens: Number of tokens used in the prompt.
        completion_tokens: Number of tokens produced in the completion.
        total_tokens: Total tokens used (prompt + completion).
        completion_tokens_details: Optional breakdown of completion-token usage.
        prompt_tokens_details: Optional breakdown of prompt-token usage.

    Notes:
        In streaming, usage may be present only in the last chunk if requested
        via `stream_options: {"include_usage": true}`. If the stream is interrupted,
        the final usage chunk might not be delivered.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[Any] = None
    prompt_tokens_details: Optional[Any] = None


class ChatCompletion(CustomBaseModel):
    """Top-level non-streaming ChatCompletion response.

    Attributes:
        id: Unique identifier for the completion.
        object: Object type, typically "chat.completion".
        created: UNIX timestamp indicating creation time.
        model: The language model used.
        choices: A list of choice objects describing possible completions.
        usage: Token usage statistics, if available.
        service_tier: Optional service-tier metadata.
        system_fingerprint: Optional backend configuration fingerprint.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage] = None

    service_tier: Optional[Any] = None
    system_fingerprint: Optional[Any] = None


# Streaming types (strictly aligned to OpenAI's spec)


class ChoiceDeltaFunctionCall(CustomBaseModel):
    """Deprecated function-call delta in a streamed chunk.

    Attributes:
        arguments: JSON string of arguments (may be None).
        name: Function name (may be None).

    Notes:
        Deprecated in favor of `tool_calls` and kept for compatibility with older models.
    """

    arguments: Optional[str] = None
    name: Optional[str] = None


class ChoiceDeltaToolCallFunction(CustomBaseModel):
    """Function details within a streamed tool-call delta.

    Attributes:
        arguments: JSON string of arguments (may be None).
        name: Function name (may be None).
    """

    arguments: Optional[str] = None
    name: Optional[str] = None


class ChoiceDeltaToolCall(CustomBaseModel):
    """A single tool-call delta within a streamed chunk.

    Attributes:
        index: Index of this tool call within the list.
        id: Optional unique identifier of the tool call.
        function: Optional function details for this tool call.
        type: Type of tool call ("function"), optional.
    """

    index: int
    id: Optional[str] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None
    type: Optional[Literal["function"]] = None


class ChoiceDelta(CustomBaseModel):
    """Delta payload for a streaming chunk.

    Attributes:
        content: Incremental text content for this chunk, if any.
        function_call: Deprecated direct function-call details; prefer `tool_calls`.
        refusal: A refusal message, if present.
        role: Author role for this chunk; appears in the first chunk.
        tool_calls: A list of tool-call deltas used within this chunk, if any.

    Notes:
        Regular content chunks typically populate `content`, while tool-call chunks
        populate `tool_calls` entries (each with `index`, optional `id`, and a
        `function` with optional `name` and `arguments`).
    """

    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaFunctionCall] = None
    refusal: Optional[str] = None
    role: Optional[Literal["developer", "system", "user", "assistant", "tool"]] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class ChoiceChunk(CustomBaseModel):
    """A single choice from a streaming ChatCompletion chunk.

    Attributes:
        index: Index of this choice within the list of choices.
        finish_reason: Reason the model stopped generating tokens.
            One of: "stop", "length", "tool_calls", "content_filter", "function_call".
        delta: Incremental changes for this chunk (content, tool_calls, etc.).
        logprobs: Optional log-probability metadata, if available.
    """

    index: int
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ] = None
    delta: ChoiceDelta
    logprobs: Optional[Any] = None


class ChatCompletionChunk(CustomBaseModel):
    """Top-level streaming ChatCompletion chunk.

    Attributes:
        id: Unique identifier for the completion; constant across chunks.
        object: Literal "chat.completion.chunk".
        created: UNIX timestamp for creation; constant across chunks.
        model: The language model used.
        choices: A list of streamed choice chunks.
        usage: Token usage statistics (if requested via `stream_options.include_usage`).
            Present only in the final chunk.
        service_tier: Optional processing tier used, one of:
            "auto", "default", "flex", "scale", "priority".
        system_fingerprint: Optional backend configuration fingerprint.

    Notes:
        - Strict schema: payloads must conform to this structure for validation.
        - Tool-call chunks are represented via `choices[*].delta.tool_calls`.
        - Regular content chunks use `choices[*].delta.content`.
    """

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChoiceChunk]
    usage: Optional[CompletionUsage] = None

    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = (
        None
    )
    system_fingerprint: Optional[str] = None

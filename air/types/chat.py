"""
Pydantic models for ChatCompletion responses, extended to handle specialized usage.

This module provides:
  • Function               – Details of a tool/function call
  • ChatCompletionMessage  – One message in the conversation
  • Choice                 – A single choice in a ChatCompletion
  • CompletionUsage        – Usage metadata (token counts)
  • ChatCompletion         – Top-level ChatCompletion response
"""

from typing import Optional, List, Any

from air.types.base import CustomBaseModel


class Function(CustomBaseModel):
    """Represents the function details used in a tool call.

    Attributes:
        name: The name of the function being invoked.
        arguments: A serialized set of arguments for the function call.
    """

    name: str
    arguments: str


class ChatCompletionMessageToolCall(CustomBaseModel):
    """A single tool-call object within a message.

    Attributes:
        id: The unique identifier of this tool call.
        function: The function details, if any.
        type: The type of call (e.g., 'function').
    """

    id: str
    type: str
    function: Optional[Function] = None


class ChatCompletionMessage(CustomBaseModel):
    """Represents one message within a conversation, possibly including tool calls.

    Attributes:
        role: The role of the sender (e.g., 'assistant', 'user', or 'system').
        content: The main text of the message, if any.
        refusal: A refusal statement, if present.
        annotations: Additional data or metadata, if any.
        audio: Audio data or reference to audio content, if any.
        function_call: Details of a function call if invoked directly.
        reasoning_content: Reasoning or chain-of-thought content, if provided.
        tool_calls: A list of tool calls used within this message, if any.
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
    """Represents a single choice from a ChatCompletion response.

    Attributes:
        index: The index of this choice within the list of choices.
        finish_reason: The reason this choice completed (e.g., 'stop', 'tool_calls').
        message: The message returned with this choice.
        stop_reason: An optional stop reason code.
        logprobs: Optional log-probability data, if available.
    """

    index: int
    finish_reason: Optional[str]
    message: ChatCompletionMessage
    stop_reason: Optional[int] = None
    logprobs: Optional[Any] = None


class CompletionUsage(CustomBaseModel):
    """Tracks usage details for a ChatCompletion response.

    Attributes:
        prompt_tokens: Number of tokens used in the prompt.
        completion_tokens: Number of tokens produced in the completion.
        total_tokens: Total tokens used (prompt + completion).
        completion_tokens_details: A detailed breakdown of completion-token usage, if available.
        prompt_tokens_details: A detailed breakdown of prompt-token usage, if available.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[Any] = None
    prompt_tokens_details: Optional[Any] = None


class ChatCompletion(CustomBaseModel):
    """Top-level ChatCompletion response returned by the API.

    Attributes:
        id: Unique identifier for this ChatCompletion.
        object: The object type, typically "chat.completion".
        created: A UNIX timestamp indicating creation time.
        model: The language model used.
        choices: A list of choice objects describing possible completions.
        usage: Token usage statistics for this completion, if available.
        service_tier: Possible service-tier metadata, if provided.
        system_fingerprint: System or model fingerprint, if provided.
        prompt_logprobs: Log-probability data for the prompt, if available.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage] = None

    service_tier: Optional[Any] = None
    system_fingerprint: Optional[Any] = None
    prompt_logprobs: Optional[Any] = None

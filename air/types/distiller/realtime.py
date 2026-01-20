"""Type definitions for realtime events in the AI Refinery voice distiller."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class RealtimeEvent(Enum):
    """Enumerates key events in the voice pipeline."""

    # Server -> Client
    SESSION_CREATED = "session.created"
    RESPONSE_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_TRANSCRIPT_DONE = "response.audio_transcript.done"

    RESPONSE_CREATED = "response.created"  # not in design

    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"  # not in design
    RESPONSE_DONE = "response.done"

    # Use for distiller text output
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"

    # Client -> Server
    INPUT_AUDIO_APPEND = "input_audio_buffer.append"
    INPUT_TEXT = "input_text"
    RESPONSE_CANCEL = "response.cancel"

    # Additional
    INPUT_AUDIO_COMMIT = "input_audio_buffer.commit"  # Client VAD
    INPUT_AUDIO_CLEAR = "input_audio_buffer.clear"  # Client VAD

    SESSION_UPDATE = "session.update"


# ----------------------------
# Base Event Class
# ----------------------------


class RealtimeEventBase(BaseModel):
    """
    Base Realtime Event, used to set loose schema and other configurations.
    """

    model_config = ConfigDict(extra="allow")  # loose schema


# ----------------------------
# Server -> Client Events
# ----------------------------


class SessionCreatedEvent(RealtimeEventBase):
    """
    Event sent when a session is created and the WebSocket connection is established.
    """

    type: Literal["session.created"] = Field(
        default=RealtimeEvent.SESSION_CREATED.value,
        description="The event type, must be 'session.created'.",
    )


class ResponseTranscriptDeltaEvent(RealtimeEventBase):
    """Partial transcription result sent during audio processing."""

    type: Literal["response.audio_transcript.delta"] = Field(
        default=RealtimeEvent.RESPONSE_TRANSCRIPT_DELTA.value,
        description="The event type identifier for partial transcript deltas.",
    )
    delta: str = Field(..., description="Partial transcript text.")


class ResponseTranscriptDoneEvent(RealtimeEventBase):
    """Final transcription result once audio input has been fully processed."""

    type: Literal["response.audio_transcript.done"] = Field(
        default=RealtimeEvent.RESPONSE_TRANSCRIPT_DONE.value,
        description="The event type identifier for final transcript results.",
    )
    text: str = Field(..., description="Final transcript text.")


class ResponseCreatedEvent(RealtimeEventBase):
    """Event indicating that a response object has been created (not part of original design)."""

    type: Literal["response.created"] = Field(
        default=RealtimeEvent.RESPONSE_CREATED.value,
        description="The event type identifier for response creation.",
    )


class ResponseAudioDeltaEvent(RealtimeEventBase):
    """Chunked audio data being streamed to the client."""

    type: Literal["response.audio.delta"] = Field(
        default=RealtimeEvent.RESPONSE_AUDIO_DELTA.value,
        description="The event type identifier for streaming audio chunks.",
    )
    audio: str = Field(..., description="Base64-encoded audio chunk.")


class ResponseAudioDoneEvent(RealtimeEventBase):
    """Event indicating that all audio chunks have been sent (not part of original design)."""

    type: Literal["response.audio.done"] = Field(
        default=RealtimeEvent.RESPONSE_AUDIO_DONE.value,
        description="The event type identifier for end of audio streaming.",
    )


class ResponseDoneEvent(RealtimeEventBase):
    """Event signaling that a full response has been completely generated."""

    type: Literal["response.done"] = Field(
        default=RealtimeEvent.RESPONSE_DONE.value,
        description="The event type identifier for response completion.",
    )


class ResponseTextDeltaEvent(RealtimeEventBase):
    """Partial text output (distiller result) streamed to the client."""

    type: Literal["response.text.delta"] = Field(
        default=RealtimeEvent.RESPONSE_TEXT_DELTA.value,
        description="The event type identifier for partial text deltas.",
    )
    content: str = Field(..., description="Partial text output content.")


class ResponseTextDoneEvent(RealtimeEventBase):
    """Final text output once the distiller has finished generating text."""

    type: Literal["response.text.done"] = Field(
        default=RealtimeEvent.RESPONSE_TEXT_DONE.value,
        description="The event type identifier for final text outputs.",
    )


# ----------------------------
# Client -> Server Events
# ----------------------------


class InputAudioAppendEvent(RealtimeEventBase):
    """Client sends this event to append audio data to the input buffer."""

    type: Literal["input_audio_buffer.append"] = Field(
        default=RealtimeEvent.INPUT_AUDIO_APPEND.value,
        description="The event type identifier for appending audio input.",
    )
    audio: str = Field(..., description="Base64-encoded audio chunk to append.")


class InputTextEvent(RealtimeEventBase):
    """Client sends this event in case of text input."""

    type: Literal["input_text"] = Field(
        default=RealtimeEvent.INPUT_TEXT.value,
        description="The event type identifier for text input.",
    )
    text: str = Field(..., description="Text query content.")


class ResponseCancelEvent(RealtimeEventBase):
    """Client requests cancellation of an in-progress response."""

    type: Literal["response.cancel"] = Field(
        default=RealtimeEvent.RESPONSE_CANCEL.value,
        description="The event type identifier for response cancellation.",
    )


# ----------------------------
# Additional Events
# ----------------------------


class InputAudioCommitEvent(RealtimeEventBase):
    """Client signals that input audio is complete (often after VAD triggers end of speech)."""

    type: Literal["input_audio_buffer.commit"] = Field(
        default=RealtimeEvent.INPUT_AUDIO_COMMIT.value,
        description="The event type identifier for committing input audio.",
    )


class InputAudioClearEvent(RealtimeEventBase):
    """Client signals that the input audio buffer should be cleared (VAD reset)."""

    type: Literal["input_audio_buffer.clear"] = Field(
        default=RealtimeEvent.INPUT_AUDIO_CLEAR.value,
        description="The event type identifier for clearing the input audio buffer.",
    )


class SessionUpdateEvent(RealtimeEventBase):
    """Event used to update session parameters during an active WebSocket connection."""

    type: Literal["session.update"] = Field(
        default=RealtimeEvent.SESSION_UPDATE.value,
        description="The event type identifier for session updates.",
    )


# ----------------------------
# Type Adapter for Client Request validation
# ----------------------------

ClientRequestEvent = TypeAdapter(
    Annotated[
        Union[
            InputAudioAppendEvent,
            InputTextEvent,
            ResponseCancelEvent,
            InputAudioCommitEvent,
            InputAudioClearEvent,
            SessionUpdateEvent,
        ],
        Field(discriminator="type"),
    ]
)

# ----------------------------
# Type Adapter for Server Response validation
# ----------------------------

ServerResponseEvent = TypeAdapter(
    Annotated[
        Union[
            SessionCreatedEvent,
            ResponseTranscriptDeltaEvent,
            ResponseTranscriptDoneEvent,
            ResponseCreatedEvent,
            ResponseAudioDeltaEvent,
            ResponseAudioDoneEvent,
            ResponseDoneEvent,
            ResponseTextDeltaEvent,
            ResponseTextDoneEvent,
        ],
        Field(discriminator="type"),
    ]
)

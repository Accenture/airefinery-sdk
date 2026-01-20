"""Realtime voice distiller client for AI Refinery.

This module provides the AsyncRealtimeDistillerClient class for real-time voice
interactions with the AI Refinery's distiller service.
"""

import asyncio
import base64
import logging
from typing import Any, AsyncGenerator, Callable, Optional

from air import BASE_URL, __version__
from air.distiller.client import AsyncDistillerClient
from air.distiller.utils import realtime_helper
from air.types.distiller.client import DistillerIncomingMessage
from air.types.distiller.realtime import (
    InputAudioAppendEvent,
    InputAudioClearEvent,
    InputAudioCommitEvent,
    InputTextEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ServerResponseEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class AsyncRealtimeDistillerClient(AsyncDistillerClient):
    """
    Realtime Distiller SDK for AI Refinery.

    This class provides an interface for interacting real-time voice with the AI Refinery's
    distiller service, allowing users to create projects, download configurations,
    and run distiller sessions with audio input/output.
    """

    # Define API endpoints for various operations
    run_suffix = "distiller/realtime"
    create_suffix = "distiller/create"
    download_suffix = "distiller/download"
    reset_suffix = "distiller/reset"
    max_size_ws_recv = 167772160
    ping_interval = 10

    def __init__(self, *, base_url: str = "", **kwargs) -> None:
        """
        Initialize the AsyncRealtimeDistillerClient.

        Args:
            base_url (str, optional): Base URL for the API. Defaults to "".
            **kwargs: Additional arguments passed to parent AsyncDistillerClient.
        """
        super().__init__(base_url=base_url, **kwargs)

        # Initialize audio buffer
        self._audio_buffer = bytearray()
        self._audio_buffer_lock = asyncio.Lock()

        self._session_lock = asyncio.Lock()
        self._session_created = False

        self._response_lock = asyncio.Lock()
        self._response_created = False

        # For Realtime client, add PING messages to queue
        self._queue_ping_messages = True

    def __call__(
        self, **kwargs
    ) -> "_VoiceDistillerContextManager":  # pylint: disable=protected-access
        """
        Return a context manager for connecting to the Distiller server.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            _VoiceDistillerContextManager: The context manager instance.
        """
        return self._VoiceDistillerContextManager(self, **kwargs)

    class _VoiceDistillerContextManager(
        AsyncDistillerClient._DistillerContextManager
    ):  # pylint: disable=too-few-public-methods
        def __init__(self, client: "AsyncRealtimeDistillerClient", **kwargs):
            super().__init__(client, **kwargs)
            self.client: "AsyncRealtimeDistillerClient" = client

        async def __aenter__(self) -> "AsyncRealtimeDistillerClient":
            await super().__aenter__()  # connects
            return self.client

    async def get_session_created(self) -> bool:
        """Get the current session creation status.

        Returns:
            True if session has been created, False otherwise.
        """
        async with self._session_lock:
            return self._session_created

    async def set_session_created(self, value: bool) -> None:
        """Set the session creation status.

        Args:
            value: New session creation status.
        """
        async with self._session_lock:
            self._session_created = value

    async def get_response_created(self) -> bool:
        """Get the current response creation status.

        Returns:
            True if response is currently being created/processed, False otherwise.
        """
        async with self._response_lock:
            return self._response_created

    async def set_response_created(self, value: bool) -> None:
        """Set the response creation status.

        Args:
            value: New response creation status.
        """
        async with self._response_lock:
            self._response_created = value

    async def wait_for_session_created(self, timeout: float | None = 30.0) -> None:
        """Wait until the server sends `session.created`, or until *timeout* expires."""
        loop = asyncio.get_running_loop()
        start = loop.time()
        poll_interval = 0.05
        while not await self.get_session_created():
            if timeout is not None and loop.time() - start >= timeout:
                raise asyncio.TimeoutError(
                    "Timed out waiting for realtime session creation."
                )
            await asyncio.sleep(poll_interval)

    async def _process_message_hook(self, msg: dict) -> None:
        """Process realtime-specific messages.

        Args:
            msg: WebSocket message containing event type and data.
        """
        try:
            # Skip PING messages and non-typed messages for now
            if msg.get("type") == "PING" or not msg.get("type"):
                return

            # Validate against server response events
            try:
                parsed_event = ServerResponseEvent.validate_python(msg)

                if isinstance(parsed_event, SessionCreatedEvent):
                    await self.set_session_created(True)
                elif isinstance(parsed_event, ResponseCreatedEvent):
                    await self.set_response_created(True)
                    logger.info("Realtime response stream started.")
                elif isinstance(parsed_event, ResponseDoneEvent):
                    await self.set_response_created(False)
                    logger.info("Realtime response stream completed.")

            except Exception as validation_error:
                logger.warning(
                    f"Failed to validate server response: {validation_error}"
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Send audio chunk to server.

        Args:
            audio_bytes: Raw audio data to send to the server.

        Raises:
            ValueError: If audio_bytes is empty or invalid.
        """
        if not audio_bytes:
            raise ValueError("Audio bytes cannot be empty")

        try:
            encoded = await asyncio.to_thread(base64.b64encode, bytes(audio_bytes))
            audio_b64 = encoded.decode("utf-8")

            if await self.get_response_created():
                return
            payload = InputAudioAppendEvent(audio=audio_b64)
            if self.send_queue:
                await self.send_queue.put(payload)
            else:
                logger.error("Send queue not available")
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
            raise

    async def commit_audio(self) -> None:
        """Signal end of audio input to the server."""
        payload = InputAudioCommitEvent()
        if self.send_queue:
            await self.send_queue.put(payload)
        else:
            logger.error("Send queue not available for audio commit")

    async def clear_audio_buffer(self) -> None:
        """Clear server-side audio buffer."""
        payload = InputAudioClearEvent()
        if self.send_queue:
            await self.send_queue.put(payload)
        else:
            logger.error("Send queue not available for buffer clear")

    async def send_text_query(self, text: str) -> None:
        """Send queries of type text to the server.

        Args:
            text: The text query to send.

        Raises:
            ValueError: If text query is empty.
        """
        if not text.strip():
            raise ValueError("Text query cannot be empty")
        payload = InputTextEvent(text=text)
        if self.send_queue:
            await self.send_queue.put(payload)
        else:
            logger.error("Send queue not available for text query")

    async def send_update(self) -> None:
        """Send audio chunk to server."""
        payload = SessionUpdateEvent()
        if self.send_queue:
            await self.send_queue.put(payload)
        else:
            logger.error("Send queue not available for session update")

    async def connect(
        self,
        project: str,
        uuid: str,
        project_version: Optional[str] = None,
        custom_agent_gallery: Optional[dict[str, Callable | dict]] = None,
        executor_dict: Optional[dict[str, Callable | dict]] = None,
    ) -> None:
        """
        Connect to the account/project/uuid-specific URL.

        Args:
            project (str): Name of the project.
            uuid (str): Unique identifier for the session.
            project_version: Specific version of the project (optional).
            custom_agent_gallery (Optional[dict[str, Callable]], optional):
                        Custom agent handlers. Defaults to None.
            executor_dict: Custom executor configurations (optional).

        Raises:
            Exception: If unable to establish WebSocket connection or session creation fails.
        """
        try:
            # Inherited connect
            await super().connect(
                project, uuid, project_version, custom_agent_gallery, executor_dict
            )

            await self.wait_for_session_created()
            logger.info("Realtime session created.")

        except asyncio.TimeoutError as exc:
            logger.error("Timed out waiting for realtime session acknowledgement.")
            await self.close()
            raise

        except Exception as e:
            logger.error(f"Failed to connect to realtime voice session: {e}")
            raise

    async def close(self) -> None:
        """
        Close the websocket connection and cancel background tasks.

        Raises:
            Exception: If error occurs during session cleanup.
        """
        try:
            # inherited close
            await super().close()

            await self.set_session_created(False)
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            raise

    # use for get audio output
    async def get_responses(
        self, **kwargs: Any
    ) -> AsyncGenerator[DistillerIncomingMessage, None]:
        """Get response messages from the server, handling tool execution.

        Yields:
            DistillerIncomingMessage: Server messages streamed until completion.

        Raises:
            ConnectionError: If WebSocket connection is lost.
        """
        try:
            while True:
                try:
                    msg = await self.recv()

                    # Handle wait messages for tool execution
                    if msg.get("status") == "wait":
                        role = msg.get("role")
                        msg_kwargs = msg.get("kwargs", {})
                        request_id = msg.get("request_id", "")

                        # Execute the tool
                        if self.executor_dict and role in self.executor_dict:
                            wait_task = asyncio.create_task(
                                self.executor_dict[role](
                                    request_id=request_id, **msg_kwargs
                                )
                            )
                            if (
                                hasattr(self, "_wait_task_list")
                                and self._wait_task_list is not None
                            ):
                                self._wait_task_list.append(wait_task)
                        continue

                    yield msg
                    try:
                        parsed_event = ServerResponseEvent.validate_python(msg)
                        if isinstance(parsed_event, ResponseDoneEvent):
                            logger.info("Realtime response finished.")
                            return
                    except Exception:
                        pass
                except ConnectionError:
                    logger.warning("Connection lost during response stream")
                    return
        except Exception as e:
            logger.error(f"Voice response error: {e}")
            raise

    async def listen_and_respond(
        self,
        sample_rate: int = 16000,
    ) -> None:
        """
        Listen to microphone input and play audio responses.

        Args:
            sample_rate: Audio sample rate in Hz (must match YAML speech_config)
        """

        mic_task = None
        audio_queue: asyncio.Queue = asyncio.Queue()
        tts_received = asyncio.Event()
        tts_received.clear()
        tts_complete = asyncio.Event()
        tts_complete.clear()

        try:

            # Start microphone streaming
            mic_task = asyncio.create_task(
                realtime_helper.stream_microphone_input(self)
            )

            # Start speaker output
            speaker_task = asyncio.create_task(
                realtime_helper.stream_audio_output(
                    audio_queue=audio_queue,
                    tts_received=tts_received,
                    tts_complete=tts_complete,
                    sample_rate=sample_rate,
                )
            )

            # Process responses
            async for response in self.get_responses():
                if response.get("type") == "PING":
                    continue

                if response.get("type") == "response.created":
                    if mic_task:
                        mic_task.cancel()
                        mic_task = None
                        logger.info("Mic Stream Closed")
                    continue

                if response.get("type") == "response.audio.delta":
                    await audio_queue.put(response.get("audio"))
                    continue

                if response.get("type") in ["response.audio.done", "response.done"]:
                    tts_received.set()
                    continue

                if response.get("type") == "response.audio_transcript.delta":
                    logger.info(f"Transcription Delta: {response.get('delta')}")
                    continue

                if response.get("type") == "response.audio_transcript.done":
                    logger.info(f"Transcription Final: {response.get('text')}")
                    continue

                if response.get("type") == "response.text.delta":
                    logger.info(
                        f"%%% [Delta] AGENT {response.get('role', 'ASSISTANT')} %%%"
                    )
                    logger.info(response.get("content"))
                    continue

            # Wait for playback completion
            while not (tts_received.is_set() and audio_queue.qsize() == 0):
                await asyncio.sleep(0.01)

            speaker_task.cancel()

        finally:
            if mic_task and not mic_task.done():
                mic_task.cancel()

    async def send_text_and_respond(
        self,
        text: str,
        sample_rate: int = 16000,
    ) -> None:
        """
        Send text query and play audio responses.

        Args:
            text: The text query to send
            sample_rate: Audio sample rate in Hz (must match YAML speech_config)

        Raises:
            ValueError: If text is empty
        """
        if not text.strip():
            raise ValueError("Text query cannot be empty")

        audio_queue: asyncio.Queue = asyncio.Queue()
        tts_received = asyncio.Event()
        tts_received.clear()
        tts_complete = asyncio.Event()
        tts_complete.clear()

        try:
            # Send text query
            await self.send_text_query(text)

            # Start speaker output
            speaker_task = asyncio.create_task(
                realtime_helper.stream_audio_output(
                    audio_queue=audio_queue,
                    tts_received=tts_received,
                    tts_complete=tts_complete,
                    sample_rate=sample_rate,
                )
            )

            # Process responses
            async for response in self.get_responses():
                if response.get("type") == "PING":
                    continue

                if response.get("type") == "response.audio.delta":
                    await audio_queue.put(response.get("audio"))
                    continue

                if response.get("type") in ["response.audio.done", "response.done"]:
                    tts_received.set()
                    continue

                if response.get("type") == "response.audio_transcript.delta":
                    logger.info(f"Transcription Delta: {response.get('delta')}")
                    continue

                if response.get("type") == "response.audio_transcript.done":
                    logger.info(f"Transcription Final: {response.get('text')}")
                    continue

                if response.get("type") == "response.text.delta":
                    logger.info(
                        f"%%% [Delta] AGENT {response.get('role', 'ASSISTANT')} %%%"
                    )
                    logger.info(response.get("content"))
                    continue

            # Wait for playback completion
            while not (tts_received.is_set() and audio_queue.qsize() == 0):
                await asyncio.sleep(0.01)

            speaker_task.cancel()

        except Exception as e:
            logger.error(f"Error in text query: {e}")
            raise

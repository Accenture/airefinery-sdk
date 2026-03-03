"""
Realtime voice helper functions for AI Refinery SDK.
"""

import asyncio
import base64
import logging
import sys
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# try:
#     import sounddevice as sd
# except (ImportError, OSError) as e:
#     logger.warning(f"sounddevice not available: {e}")
#     sd = None


# Audio configuration constant
CHUNK_DURATION = 0.3  # seconds - capture/playback chunk duration (e.g. 300ms)


def import_sounddevice():
    try:
        import sounddevice as sd

        return sd

    except ImportError as e:
        logger.error(
            "The 'sounddevice' package is not installed.\n"
            "Install with:\n"
            '    pip install ".[realtime]"\n'
            "Or manually:\n"
            "    pip install sounddevice\n\n"
            f"Original error: {e}",
            exc_info=True,
        )
        raise RuntimeError(
            "Audio I/O unavailable: sounddevice is not installed."
        ) from e

    except OSError as e:
        logger.error(
            "sounddevice failed to load its PortAudio backend.\n"
            "Install PortAudio:\n"
            "  macOS:    brew install portaudio\n"
            "  Ubuntu:   sudo apt-get install portaudio19-dev\n"
            "  Fedora:   sudo dnf install portaudio-devel\n"
            "  Windows:  conda install -c conda-forge portaudio\n\n"
            f"Original error: {e}",
            exc_info=True,
        )
        raise RuntimeError(
            "Audio I/O unavailable: PortAudio is missing or misconfigured."
        ) from e

    except Exception as e:
        logger.error(
            "Unexpected error while importing sounddevice.",
            exc_info=True,
        )
        raise RuntimeError("Audio I/O unavailable due to an unexpected error.") from e


async def stream_microphone_input(voice_client):  # pragma: no cover
    """Stream microphone input to the voice client.

    Args:
        voice_client: Voice client instance to send audio to.
    """
    sd = import_sounddevice()

    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    # Calculate chunk size based on sample rate
    chunk_size = int(16000 * CHUNK_DURATION)

    def callback(indata, frames, time, status):
        if status:
            logging.warning(f"Microphone input error: {status}")
        try:
            # Put raw bytes into queue
            audio_bytes = indata.tobytes()
            asyncio.run_coroutine_threadsafe(audio_queue.put(audio_bytes), loop)
        except Exception as e:
            logging.error(f"Error processing microphone data: {e}")

    try:
        # Open mic stream
        with sd.InputStream(
            samplerate=16000,
            channels=1,  # mono audio always
            dtype="int16",
            blocksize=chunk_size,  # Scales with sample rate
            callback=callback,
        ):
            logger.info("Mic stream started")
            while True:
                audio_bytes = await audio_queue.get()
                await voice_client.send_audio_chunk(audio_bytes)
    except Exception as e:
        logging.error(f"Microphone stream error: {e}")
        raise


async def stream_audio_output(  # pragma: no cover
    audio_queue, tts_received, tts_complete, sample_rate=16000, cancel_event=None
):
    """Stream audio output to speakers.

    Supports per-agent cancellation: when cancel_event is set, buffered audio
    is drained and playback pauses until the event is cleared (next agent
    boundary), then resumes automatically for the next agent's audio.

    Args:
        audio_queue: Queue containing base64-encoded audio chunks.
        tts_received: Event indicating synthesized audio data has been received.
        tts_complete: Event to set when playback is complete.
        sample_rate: Playback sample rate in Hz (default: 16000).
        cancel_event: Optional asyncio.Event; when set, current agent's buffered
            audio is drained and playback pauses until the event is cleared.
    """
    sd = import_sounddevice()

    if sd is None:
        raise RuntimeError("sounddevice library not available")

    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,  # mono audio always
            dtype="int16",
            blocksize=0,
        ) as stream:
            while True:
                # Per-agent cancel: drain buffered audio and pause until cleared.
                if cancel_event and cancel_event.is_set():
                    # Drain any buffered chunks from the cancelled agent.
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    logger.info("Audio playback paused for agent skip.")

                    # Wait for cancel_event to be cleared (next agent starting).
                    while cancel_event and cancel_event.is_set():
                        await asyncio.sleep(0.05)

                    # Drain any stale chunks that arrived during the pause
                    # before next agent's fresh audio starts coming in.
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    logger.info("Audio playback resuming for next agent.")
                    continue

                # Check if we should stop playback normally.
                if audio_queue.qsize() == 0 and tts_received.is_set():
                    tts_complete.set()
                    break

                # Use short timeout so we can re-check cancel promptly.
                try:
                    audio_chunk = await asyncio.wait_for(
                        audio_queue.get(), timeout=0.05
                    )
                except asyncio.TimeoutError:
                    continue

                audio_bytes = base64.b64decode(audio_chunk)
                audio_bytes = np.frombuffer(audio_bytes, dtype=np.int16)
                await asyncio.sleep(0)  # Yield so cancel_event.set() can execute
                stream.write(audio_bytes)
    except asyncio.CancelledError:
        logger.info("Audio output task cancelled.")
    except Exception as e:
        logging.error(f"Speaker stream error: {e}")
        raise


def _poll_unix(stop, cancel_event, loop):  # pragma: no cover
    """Unix/macOS keypress polling using termios/tty/select.

    Blocks in cbreak mode and uses ``select()`` with a 100 ms timeout for
    non-blocking spacebar detection.  Restores terminal settings on exit.
    """
    import select  # pylint: disable=import-outside-toplevel
    import termios  # pylint: disable=import-outside-toplevel
    import tty  # pylint: disable=import-outside-toplevel

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            # Pause while a cancel is already pending (event is set).
            # Prevents double-firing for the same agent and waits for
            # the event to be cleared before arming again.
            if cancel_event.is_set():
                time.sleep(0.05)
                continue

            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                ch = sys.stdin.read(1)
                if ch == " ":
                    logger.info("Spacebar pressed - cancelling current agent.")
                    # Must use call_soon_threadsafe: asyncio.Event is not
                    # thread-safe to set directly from a background thread.
                    loop.call_soon_threadsafe(cancel_event.set)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _poll_win32(stop, cancel_event, loop):  # pragma: no cover
    """Windows keypress polling using msvcrt.

    Uses ``msvcrt.kbhit()`` (non-blocking check) and ``msvcrt.getwch()``
    (read without echo) for spacebar detection.  No terminal mode changes
    are needed on Windows.
    """
    import msvcrt  # pylint: disable=import-outside-toplevel,import-error

    while not stop.is_set():
        # Pause while a cancel is already pending (event is set).
        # Prevents double-firing for the same agent and waits for
        # the event to be cleared before arming again.
        if cancel_event.is_set():
            time.sleep(0.05)
            continue

        if msvcrt.kbhit():  # type: ignore[attr-defined]
            ch = msvcrt.getwch()  # type: ignore[attr-defined]
            if ch == " ":
                logger.info("Spacebar pressed - cancelling current agent.")
                loop.call_soon_threadsafe(cancel_event.set)
        else:
            # No key available — sleep briefly to avoid busy-waiting.
            # Matches the ~100 ms cadence of the Unix select() timeout.
            time.sleep(0.1)


class CancelOnKeypress:  # pragma: no cover
    """Async context manager that listens for a spacebar press.

    Starts a background keyboard listener on entry and yields a
    cancel_event (asyncio.Event) that is set when the spacebar is
    pressed.  Cleans up the listener thread on exit regardless of
    whether cancellation occurred.

    Usage::

        async with realtime_helper.CancelOnKeypress() as cancel_event:
            await vc.send_text_and_respond(..., cancel_event=cancel_event)
    """

    def __init__(self) -> None:
        self._cancel_event: asyncio.Event | None = None
        self._cancel_stop: threading.Event | None = None
        self._cancel_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def __aenter__(self) -> asyncio.Event:
        self._cancel_event = asyncio.Event()
        self._cancel_stop = threading.Event()
        self._cancel_task = asyncio.create_task(
            wait_for_cancel_keypress(self._cancel_event, stop_event=self._cancel_stop)
        )
        return self._cancel_event

    async def __aexit__(self, *_) -> None:
        if self._cancel_stop:
            self._cancel_stop.set()
        if self._cancel_task and not self._cancel_task.done():
            try:
                await asyncio.wait_for(self._cancel_task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._cancel_task.cancel()


async def wait_for_cancel_keypress(  # pragma: no cover
    cancel_event, stop_event=None
) -> None:
    """Persistently listen for spacebar presses for the lifetime of a response.

    Sets cancel_event on each press, then pauses until the event is cleared
    (signalling that the next agent has started) before listening again.
    This allows each agent's TTS to be individually skipped with separate
    SPACE key presses.

    Platform support:
        - macOS / Linux: uses termios + select for non-blocking keypress detection.
        - Windows: uses msvcrt.kbhit() + msvcrt.getwch() for non-blocking detection.
        - Other platforms: logs a warning and exits (no cancel support).

    Falls back gracefully on non-TTY environments (no cancel support).

    Args:
        cancel_event: asyncio.Event to set when spacebar is pressed.
        stop_event: threading.Event the caller sets to exit the thread.
    """
    stop = stop_event if stop_event is not None else threading.Event()
    loop = asyncio.get_running_loop()

    def _poll_continuously():
        try:
            if sys.platform == "win32":
                _poll_win32(stop, cancel_event, loop)
            else:
                _poll_unix(stop, cancel_event, loop)
        except ImportError:
            logger.warning(
                "Cancel-on-keypress is not available on this platform. "
                "Spacebar cancellation of TTS playback will be disabled. "
                "Supported platforms: macOS, Linux, Windows."
            )
        except Exception:
            logger.error("Unexpected error in cancel keypress listener.", exc_info=True)

    try:
        await asyncio.to_thread(_poll_continuously)
    except asyncio.CancelledError:
        stop.set()

"""
Realtime voice helper functions for AI Refinery SDK.
"""

import asyncio
import base64
import logging

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


async def stream_microphone_input(voice_client):
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


async def stream_audio_output(
    audio_queue, tts_received, tts_complete, sample_rate=16000
):
    """Stream audio output to speakers.

    Args:
        audio_queue: Queue containing base64-encoded audio chunks.
        tts_received: Event indicating synthesized audio data has been received.
        tts_complete: Event to set when playback is complete.
        sample_rate: Playback sample rate in Hz (default: 16000).
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

                # Check if we should stop playback
                if audio_queue.qsize() == 0 and tts_received.is_set():
                    tts_complete.set()
                    break
                audio_bytes = base64.b64decode(await audio_queue.get())
                audio_bytes = np.frombuffer(audio_bytes, dtype=np.int16)
                stream.write(audio_bytes)
    except Exception as e:
        logging.error(f"Speaker stream error: {e}")
        raise

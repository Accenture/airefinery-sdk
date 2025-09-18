"""
A module that defines Audio clients (synchronous and asynchronous)
to interact with sub-clients such as asr and tts.
"""

from air import BASE_URL
from air.audio.asr_client import ASRClient, AsyncASRClient
from air.audio.tts_client import AsyncTTSClient, TTSClient
from air.auth.token_provider import TokenProvider


class AsyncAudio:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    An audio client that exposes asr and tts in a single interface,
    operating asynchronously.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
        **kwargs
    ):
        """
        Initializes the asynchronous unified audio client with sub-clients.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.

            base_url (str, optional): Base URL for your API endpoints.
                Defaults to "https://api.airefinery.accenture.com".
            default_headers (dict[str, str] | None): Headers that apply to
                every request made by sub-clients (e.g., {"X-Client-Version": "1.2.3"}).
            **kwargs: Additional configuration parameters, if any.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.kwargs = kwargs

        # Provides async tts functionalities
        self.speech = AsyncTTSClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )

        # Provides async asr functionalities
        self.transcriptions = AsyncASRClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )


class Audio:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    An audio client that exposes asr and tts in a single interface,
    operating synchronously.
    """

    def __init__(
        self,
        api_key: str | TokenProvider,
        *,
        base_url: str = BASE_URL,
        default_headers: dict[str, str] | None = None,
        **kwargs
    ):
        """
        Initializes the synchronous unified client with sub-clients.

        Args:
            api_key (str | TokenProvider): Credential that will be placed in the
                ``Authorization`` header of every request.
                * **str** – a raw bearer token / API key.
                * **TokenProvider** – an object whose ``token()`` (and
                  ``token_async()``) method returns a valid bearer token.  The
                  client calls the provider automatically before each request.

            base_url (str, optional): Base URL for your API endpoints.
                Defaults to "https://api.airefinery.accenture.com".
            default_headers (dict[str, str] | None): Headers that apply to
                every request made by sub-clients (e.g., {"X-Client-Version": "1.2.3"}).
            **kwargs: Additional configuration parameters, if any.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.kwargs = kwargs

        # Provides sync tts functionalities
        self.speech = TTSClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )

        # Provides sync asr functionalities
        self.transcriptions = ASRClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )

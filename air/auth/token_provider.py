"""Token caching layer.

The module keeps a single JSON file per (url, client_id) pair inside
air.CACHE_DIR.  All processes on the same machine/container share that file so
only one of them has to hit the Identity Provider when a token is about to
expire.
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Final

from filelock import FileLock

from air import CACHE_DIR
from air.auth.jwt import get_jwt_token

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
_REFRESH_MARGIN: Final[int] = 60  # Seconds before "exp" when we proactively refresh
_TOKEN_DIR: Final[Path] = Path(CACHE_DIR) / "tokens"
_TOKEN_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------------- #
def _token_path(url: str, client_id: str) -> Path:
    """Return on-disk location for the given (url, client_id) pair."""
    digest = hashlib.sha256(f"{url}|{client_id}".encode("utf-8")).hexdigest()
    return _TOKEN_DIR / f"{digest}.json"


def _read(path: Path) -> tuple[str, int] | None:
    """Return (token, exp) from *path* or ``None`` if the file is missing/corrupt."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            obj = json.load(fp)
        return obj["token"], obj["exp"]
    except FileNotFoundError:
        return None
    except Exception:  # pylint: disable=broad-except
        # Corrupted file – delete so we can start fresh.
        logger.warning("Corrupted token cache at %s – deleting.", path)
        path.unlink(missing_ok=True)
        return None


def _write(path: Path, token: str, exp: int) -> None:
    """Atomically write *(token, exp)* to *path*."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump({"token": token, "exp": exp}, fp)
    tmp.replace(path)  # Atomic on POSIX


def _decode_exp(token: str) -> int:
    """Return the ``exp`` claim of *token* without verifying the signature.

    Args:
        token: A RFC 7519 JSON Web Token.

    Returns:
        The unix timestamp (`int`) found in the ``exp`` claim.

    Raises:
        ValueError: If the token is malformed or the claim is missing.
    """
    try:
        _header, payload, _sig = token.split(".")
        payload += "=" * (-len(payload) % 4)  # Restore Base-64 padding.
        body: dict[str, int] = json.loads(base64.urlsafe_b64decode(payload))
        return body["exp"]
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("Malformed JWT – cannot extract exp") from exc


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
class TokenProvider:
    """Thread/process/async-safe cached access token provider.

    Example:
        provider = TokenProvider(
            url="https://idp.example.com/oauth/token",
            client_id="my-client",
            client_secret="s3cr3t",
            extra_fields={"audience": "my-api"},
        )
        token = provider.token()               # synchronous
        token = await provider.token_async()   # asynchronous
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        url: str,
        client_id: str,
        client_secret: str,
        extra_fields: dict[str, str] | None = None,
    ) -> None:
        """Create a new provider instance.

        One instance can be created in each process; all of them share the same
        on-disk cache entry.

        Args:
            url: Auth-server token endpoint.
            client_id: OAuth client id.
            client_secret: OAuth client secret.
            extra_fields: Optional extra form fields forwarded to the IdP.
        """
        self._url = url
        self._client_id = client_id
        self._client_secret = client_secret
        self._extra_fields = extra_fields or {}

        self._path = _token_path(url, client_id)
        self._lock = FileLock(f"{self._path}.lock")

    # ------------------------------------------------------------------ #
    #  Public – synchronous
    # ------------------------------------------------------------------ #
    def token(self) -> str:
        """Return a valid bearer token, refreshing if necessary.

        The call:
            • Never blocks longer than the IdP round-trip (<1 s).
            • Is safe to call from multiple threads/processes.
        """
        token = self._fast_path()
        if token is not None:
            return token

        # Slow path – ensure only one worker hits the IdP at a time.
        with self._lock:
            token = self._fast_path()
            if token is not None:
                return token

            # Actual refresh
            token = get_jwt_token(
                url=self._url,
                client_id=self._client_id,
                client_secret=self._client_secret,
                extra_fields=self._extra_fields,
            )
            exp = _decode_exp(token)
            _write(self._path, token, exp)
            return token

    # ------------------------------------------------------------------ #
    #  Public – asynchronous
    # ------------------------------------------------------------------ #
    async def token_async(self) -> str:
        """Async wrapper around :py:meth:`token`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.token)

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #
    def _fast_path(self) -> str | None:
        """Return cached token if still valid, otherwise ``None``."""
        cached = _read(self._path)
        if not cached:
            return None

        token, exp = cached
        if exp - _REFRESH_MARGIN > time.time():
            return token
        return None

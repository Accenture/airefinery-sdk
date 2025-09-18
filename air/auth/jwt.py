"""Utility helpers that obtain a JWT access token via the OAuth-2
*client-credentials* grant.

Public API
----------

* ``get_jwt_token``        – synchronous (uses *requests*)
* ``get_jwt_token_async``  – asynchronous (uses *aiohttp*)

Only three inputs are strictly required:

    • ``url``           – fully-qualified token endpoint
    • ``client_id``
    • ``client_secret``

If a provider needs additional form fields (for instance Azure AD requires a
``scope``), supply them through the ``extra_fields`` mapping.

Typical endpoints
-----------------

Azure AD
    https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/token
    extra_fields = {"scope": "https://graph.microsoft.com/.default"}

AWS Cognito
    https://<user-pool>.amazoncognito.com/oauth2/token

Keycloak
    https://<kc-host>/realms/<realm>/protocol/openid-connect/token
"""

import json
import logging
from typing import Final

import aiohttp
import requests
from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)

_GRANT_TYPE: Final[str] = "client_credentials"


# --------------------------------------------------------------------------- #
#  Exceptions                                                                 #
# --------------------------------------------------------------------------- #
class AuthError(RuntimeError):
    """Raised when the token endpoint does not return an *access_token*."""


# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
def _extract_access_token(payload: dict[str, str]) -> str:
    token = payload.get("access_token")
    if not token:
        raise KeyError("access_token missing in response")
    return token


def _merge_fields(
    client_id: str,
    client_secret: str,
    extra: dict[str, str] | None,
) -> dict[str, str]:
    fields: dict[str, str] = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": _GRANT_TYPE,
    }
    if extra:
        fields.update(extra)
    return fields


# --------------------------------------------------------------------------- #
#  Synchronous helper                                                         #
# --------------------------------------------------------------------------- #
def get_jwt_token(
    *,
    url: str,
    client_id: str,
    client_secret: str,
    extra_fields: dict[str, str] | None = None,
    timeout: float | tuple[float, float] = 10,
) -> str:
    """Obtain a JWT access token (blocking).

    Args:
        url: Fully-qualified token endpoint.
        client_id: Confidential-client identifier.
        client_secret: Confidential-client secret.
        extra_fields: Additional ``x-www-form-urlencoded`` fields.
        timeout: Passed verbatim to :pyfunc:`requests.post`.

    Returns:
        The raw access token (usually a JWT).

    Raises:
        AuthError: If the request fails, the JSON cannot be decoded, or the
            response lacks an ``access_token`` field.
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = _merge_fields(client_id, client_secret, extra_fields)

    try:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return _extract_access_token(resp.json())

    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        logger.error("Unable to obtain access token: %s", exc)
        raise AuthError("get_jwt_token() failed") from exc


# --------------------------------------------------------------------------- #
#  Asynchronous helper (aiohttp)                                              #
# --------------------------------------------------------------------------- #
async def get_jwt_token_async(
    *,
    url: str,
    client_id: str,
    client_secret: str,
    extra_fields: dict[str, str] | None = None,
    timeout: float | ClientTimeout = 10,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Asynchronous counterpart to :func:`get_jwt_token` (uses *aiohttp*).

    Args:
        url: Fully-qualified token endpoint.
        client_id: Confidential-client identifier.
        client_secret: Confidential-client secret.
        extra_fields: Additional ``x-www-form-urlencoded`` fields.
        timeout: Either a single `float` (total seconds) or an
            :class:`aiohttp.ClientTimeout` instance.
        session: Optional :class:`aiohttp.ClientSession` to allow connection
            reuse.  If *None*, a temporary session is created and closed
            automatically.

    Returns:
        The raw access token (usually a JWT).

    Raises:
        AuthError: If the request fails, the JSON cannot be decoded, or the
            response lacks an ``access_token`` field.

    Example:
        ```python
        import asyncio
        from air.auth.jwt import get_jwt_token_async

        async def main() -> None:
            token = await get_jwt_token_async(
                url="https://my-pool.amazoncognito.com/oauth2/token",
                client_id="my-app-id",
                client_secret="***",
            )
            print(token)

        asyncio.run(main())
        ```
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = _merge_fields(client_id, client_secret, extra_fields)

    owns_session = session is None
    if owns_session:
        # Convert plain float to ClientTimeout for consistency
        timeout_obj = (
            timeout
            if isinstance(timeout, ClientTimeout)
            else ClientTimeout(total=timeout)  # type: ignore[arg-type]
        )
        session = aiohttp.ClientSession(timeout=timeout_obj)

    try:
        async with session.post(url, data=data, headers=headers) as resp:
            resp.raise_for_status()
            json_payload = await resp.json(loads=json.loads)

        return _extract_access_token(json_payload)

    except (aiohttp.ClientError, json.JSONDecodeError, KeyError) as exc:
        logger.error("Unable to obtain access token: %s", exc)
        raise AuthError("get_jwt_token_async() failed") from exc

    finally:
        if owns_session:
            await session.close()

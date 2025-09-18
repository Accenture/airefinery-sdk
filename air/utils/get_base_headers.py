"""Header–construction helpers.

The utilities in this module take either a raw API key (`str`) or a
:class:`air.auth.token_provider.TokenProvider` instance and return the HTTP
headers that must accompany **every** request made by the SDK.

Two entry-points are provided:

* ``get_base_headers`` – synchronous, suitable for regular/blocking code.
* ``get_base_headers_async`` – asynchronous, awaits the
  :pymeth:`TokenProvider.token_async` method so the surrounding event loop
  remains responsive.
"""

from typing import Mapping

from air import __version__
from air.auth.token_provider import TokenProvider

Headers = dict[str, str]


# --------------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------------- #
def _build_headers(token: str, extra: Mapping[str, str] | None) -> Headers:
    """Merge default headers with user-supplied overrides.

    Args:
        token: Bearer token that will be placed in the ``Authorization`` header.
        extra: Optional mapping of header overrides.  When a key exists in both
            *extra* and the default set, the value from *extra* wins.

    Returns:
        A new ``dict`` that contains at minimum the following keys:

        * ``Authorization``
        * ``Content-Type``
        * ``sdk_version``
    """
    headers: Headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "sdk_version": __version__,
    }
    if extra:
        headers.update(extra)
    return headers


# --------------------------------------------------------------------------- #
#  Public – synchronous
# --------------------------------------------------------------------------- #
def get_base_headers(
    api_key: str | TokenProvider,
    extra_headers: Mapping[str, str] | None = None,
) -> Headers:
    """Build the default request headers (synchronous version).

    Args:
        api_key: One of
            * ``str`` – a pre-fetched bearer token.
            * :class:`air.auth.token_provider.TokenProvider` – will be queried
              via :pymeth:`TokenProvider.token` for a fresh token.
        extra_headers: Optional mapping containing additional or overriding
            header fields (e.g. ``{"X-Client-Version": "1.2.3"}``).

    Returns:
        A freshly created dictionary ready to be passed to the ``headers=``
        argument of ``httpx.request``, ``requests.request`` or similar.
    """
    token = api_key.token() if isinstance(api_key, TokenProvider) else api_key
    return _build_headers(token, extra_headers)


# --------------------------------------------------------------------------- #
#  Public – asynchronous
# --------------------------------------------------------------------------- #
async def get_base_headers_async(
    api_key: str | TokenProvider,
    extra_headers: Mapping[str, str] | None = None,
) -> Headers:
    """Build the default request headers (asynchronous version).

    The logic is identical to :func:`get_base_headers` except that a
    :class:`TokenProvider` is awaited to avoid blocking the event loop.

    Args:
        api_key: Either a raw bearer token or a
            :class:`air.auth.token_provider.TokenProvider`.
        extra_headers: Optional mapping containing additional or overriding
            header fields.

    Returns:
        The same structure as returned by :func:`get_base_headers`.
    """
    if isinstance(api_key, TokenProvider):
        token = await api_key.token_async()
    else:
        token = api_key
    return _build_headers(token, extra_headers)

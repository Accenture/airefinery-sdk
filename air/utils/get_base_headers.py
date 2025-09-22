"""Helpers for constructing the default HTTP headers attached to every request.

Two entry points are exposed:

* get_base_headers          – synchronous, suitable for regular/blocking code.
* get_base_headers_async    – asynchronous, awaits ``TokenProvider.token_async``

Both functions accept either a raw bearer token or a
:class:`air.auth.token_provider.TokenProvider` instance.  They also allow
callers to add or override headers and to control the value of the
``Content-Type`` header (or suppress it entirely).
"""

from collections.abc import Mapping

from air import __version__
from air.auth.token_provider import TokenProvider

Headers = dict[str, str]  # Public alias


# --------------------------------------------------------------------------- #
# Private helpers                                                             #
# --------------------------------------------------------------------------- #
def _build_headers(
    token: str,
    extra: Mapping[str, str] | None = None,
    *,
    content_type: str | None = "application/json",
) -> Headers:
    """Merge default headers with optional user-supplied overrides.

    Args:
        token: Bearer token that will be placed in the ``Authorization`` header.
        extra: Optional mapping of additional or overriding header fields.
        content_type: Value for the ``Content-Type`` header or ``None`` to omit
            the header altogether.

    Returns:
        A **new** dictionary containing at minimum:

        * ``Authorization``
        * ``sdk_version``

        ``Content-Type`` is included if `content_type` is not ``None``.  Any
        keys present in `extra` override the defaults.
    """
    headers: Headers = {
        "Authorization": f"Bearer {token}",
        "sdk_version": __version__,
    }

    if content_type is not None:
        headers["Content-Type"] = content_type

    if extra:
        # `extra` wins over default values
        headers.update(extra)

    return headers


# --------------------------------------------------------------------------- #
# Public – synchronous                                                        #
# --------------------------------------------------------------------------- #
def get_base_headers(
    api_key: str | TokenProvider,
    extra_headers: Mapping[str, str] | None = None,
    *,
    content_type: str | None = "application/json",
) -> Headers:
    """Return the default request headers (synchronous version).

    Args:
        api_key: Either a raw bearer token (`str`) or a
            :class:`TokenProvider` instance.  When a ``TokenProvider`` is
            supplied, :pymeth:`TokenProvider.token` is invoked to obtain a
            fresh token.
        extra_headers: Optional mapping with additional or overriding headers.
        content_type: Value for ``Content-Type`` or ``None`` to omit it.

    Returns:
        A freshly-created dictionary ready to be passed to the ``headers=``
        argument of ``httpx.request``, ``requests.request``, etc.
    """
    token = api_key.token() if isinstance(api_key, TokenProvider) else api_key
    return _build_headers(token, extra_headers, content_type=content_type)


# --------------------------------------------------------------------------- #
# Public – asynchronous                                                       #
# --------------------------------------------------------------------------- #
async def get_base_headers_async(
    api_key: str | TokenProvider,
    extra_headers: Mapping[str, str] | None = None,
    *,
    content_type: str | None = "application/json",
) -> Headers:
    """Return the default request headers (asynchronous version).

    This variant awaits :pymeth:`TokenProvider.token_async` (if applicable) so
    as not to block the surrounding event loop.

    Args:
        api_key: Either a raw bearer token or a :class:`TokenProvider`.
        extra_headers: Optional mapping with additional or overriding headers.
        content_type: Value for ``Content-Type`` or ``None`` to omit it.

    Returns:
        The same structure as produced by :func:`get_base_headers`.
    """
    token = (
        await api_key.token_async() if isinstance(api_key, TokenProvider) else api_key
    )
    return _build_headers(token, extra_headers, content_type=content_type)

from typing import Any, Mapping

from air import __version__


def get_base_headers(
    api_key: str,
    extra_headers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the HTTP headers sent with every API request.

    Parameters
    ----------
    api_key:
        The API key used in the Authorization header.
    extra_headers:
        Optional mapping of additional headers.  If a key in
        `extra_headers` collides with a default header, it overrides
        the default value.

    Returns
    -------
    dict[str, Any]
        A freshly built dictionary containing the merged headers.
    """
    base_headers: dict[str, Any] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "sdk_version": __version__,
    }

    if extra_headers:
        # `extra_headers` overrides the defaults where keys clash.
        base_headers.update(extra_headers)

    return base_headers

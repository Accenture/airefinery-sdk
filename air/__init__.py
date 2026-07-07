"""
Top–level namespace for the *AIR* SDK.

Environment variables
---------------------
AIR_SILENT          If set to a truthy value (`1`, `true`, `yes`) suppresses the
                    legal compliance banner.
AIR_BASE_URL        Override the default API endpoint.
AIR_CACHE_DIR       Custom cache directory; defaults to ".air".
USE_LEGACY_API_URL  If set to a truthy value (`1`, `true`) routes requests to
                    legacy-api.airefinery.accenture.com (Azure Container Apps)
                    instead of api.airefinery.accenture.com (K8s cluster).
"""

import os
import pathlib
import warnings
from importlib import metadata as _metadata

# ------------------------------------------------------------------------------
# Public constants
# ------------------------------------------------------------------------------

# Try to obtain the version from installed metadata; fall back to the hard-coded
# value when running from source.
try:
    __version__: str = _metadata.version(__package__ or "airefinery-sdk")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "1.31.6"

# Decide the default base url
# - Default: api.airefinery.accenture.com (production K8s cluster)
# - USE_LEGACY_API_URL=True: legacy-api.airefinery.accenture.com (Azure Container Apps)
DEFAULT_BASE_URL = (
    "https://legacy-api.airefinery.accenture.com"
    if os.getenv("USE_LEGACY_API_URL", "").lower() in {"1", "true"}
    else "https://api.airefinery.accenture.com"
)

BASE_URL: str = os.getenv("AIR_BASE_URL", DEFAULT_BASE_URL)
CACHE_DIR: pathlib.Path = pathlib.Path(os.getenv("AIR_CACHE_DIR", ".air"))

# ------------------------------------------------------------------------------
# Public re-exports
# ------------------------------------------------------------------------------

# pylint: disable=wrong-import-position
from air.api import PostgresAPI  # noqa:  E402  (re-export)
from air.client import AIRefinery, AsyncAIRefinery  # noqa:  E402
from air.distiller.client import AsyncDistillerClient  # noqa:  E402
from air.distiller.realtime_client import AsyncRealtimeDistillerClient  # noqa:  E402
from air.document_analysis import (  # noqa:  E402
    AsyncDocumentAnalysisClient,
    DocumentAnalysisClient,
)
from air.governance import AsyncGovernanceClient, GovernanceClient  # noqa:  E402

# Backwards-compatibility alias
DistillerClient = AsyncDistillerClient

__all__ = [
    # Classes
    "PostgresAPI",
    "AIRefinery",
    "AsyncAIRefinery",
    "AsyncDistillerClient",
    "AsyncRealtimeDistillerClient",
    "AsyncDocumentAnalysisClient",
    "AsyncGovernanceClient",
    "DocumentAnalysisClient",
    "DistillerClient",
    "GovernanceClient",
    # Constants
    "BASE_URL",
    "CACHE_DIR",
    "__version__",
]

# ------------------------------------------------------------------------------
# Legal compliance banner (can be silenced via AIR_SILENT)
# ------------------------------------------------------------------------------

from air.utils import print_compliance_banner as _print_compliance_banner  # noqa:  E402

if os.getenv("AIR_SILENT", "").lower() not in {"1", "true", "yes"}:
    try:
        _print_compliance_banner()
    except Exception as exc:  # pragma: no cover
        # Never fail the import if the banner cannot be shown
        warnings.warn(
            f"Could not display the AIR compliance banner: {exc}",
            category=RuntimeWarning,
            stacklevel=2,
        )

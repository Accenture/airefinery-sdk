# pylint: disable=wrong-import-position,line-too-long,unnecessary-dunder-call
__version__ = "1.15.1"

__base_url__ = "https://api.airefinery.accenture.com"

__token_url__ = "https://login.microsoftonline.com/e0793d39-0939-496d-b129-198edd916feb/oauth2/v2.0/token"

CACHE_DIR = ".air"

from air.authenticator import Authenticator

auth = Authenticator()

from typing import Optional

from air.api import PostgresAPI  # Backward compatible
from air.api import PostgresAPI as DatabaseClient
from air.client import AIRefinery, AsyncAIRefinery
from air.distiller import AsyncDistillerClient as DistillerClient
from air.utils import compliance_banner

# AIR SDK Legal requirement
compliance_banner()


def login(
    account: str,
    api_key: str,
    base_url: Optional[str] = None,
    oauth_server: Optional[str] = None,
) -> Authenticator:
    """Helper function to instantiate the Authenticator and perform login.

    Args:
        account (str): The account name for authentication.
        api_key (str): The API key for authentication.
        base_url (Optional[str]): Base URL AIRefinery API; if not provided, defaults are used.
    """
    if oauth_server:
        print(
            "The oauth_server argument is going to be "
            "deprecated in future releases. "
            "AIRefinery authentication is updated and does not rely "
            "on access management services."
        )
    auth.__init__(account=account, api_key=api_key, base_url=base_url)
    return auth

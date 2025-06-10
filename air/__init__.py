# pylint: disable=wrong-import-position,line-too-long,unnecessary-dunder-call
__version__ = "1.10.2"

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
    account: str, api_key: str, oauth_server: Optional[str] = ""
) -> Authenticator:
    """Helper function to instantiate the Authenticator and perform login.

    Args:
        account (str): The account name for authentication.
        api_key (str): The API key for authentication.
        oauth_server (str): Base url to local oauth server for on prem only.
    """
    auth.__init__(account=account, api_key=api_key, oauth_server=oauth_server)
    return auth

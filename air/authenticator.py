import json
import time
from typing import Optional, Any
import logging

import requests

from air import __token_url__, __base_url__

# Set up logging
logger = logging.getLogger(__name__)


class Authenticator:
    """Handles authentication for the AI Refinery platform."""

    timeout = 3000  # Token validity duration in seconds

    def __init__(
        self,
        account: Optional[str] = None,
        api_key: Optional[str] = None,
        oauth_server: Optional[str] = "",
    ):
        """
        Initialize the Authenticator, attempting to log in using provided credentials.

        Args:
            account (Optional[str]): Account name for authentication.
            api_key (Optional[str]): API key for authentication.
        """
        if account is None:
            return

        self.account = account
        self.api_key = api_key
        self.oauth_server = "" if oauth_server is None else oauth_server.strip()
        self.access_token = self.login()
        self.time = time.time()

    def openai(self, base_url: Optional[str] = None) -> dict[str, Any]:
        """
        Prepare and return configuration for OpenAI API interaction.

        Args:
            base_url (Optional[str]): Base URL for the OpenAI API; if not provided, defaults are used.

        Returns:
            dict[str, Any]: Dictionary containing base_url, api_key, and default_headers.
        """
        if base_url is None or base_url == "":
            base_url = f"{__base_url__}/inference"
        else:
            base_url = f"{base_url}/inference"

        return {
            "base_url": base_url,
            "api_key": self.login(),
            "default_headers": {"airefinery_account": self.account},
        }

    def login(self) -> str:
        """
        Log in using the provided client id and client secret.

        Returns:
            str: The access token if login was successful, empty string otherwise.
        """
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = None
            oauth_url = __token_url__  # by default for azure cloud
            if not self.oauth_server:  # self.oauth_server is None or ""
                data = {
                    "client_id": self.account,
                    "scope": "https://graph.microsoft.com/.default",
                    "client_secret": self.api_key,
                    "grant_type": "client_credentials",
                }
            elif self.oauth_server.endswith(".amazoncognito.com"):  # AWS OAuth
                data = {
                    "client_id": self.account,
                    "client_secret": self.api_key,
                    "grant_type": "client_credentials",
                }
                oauth_url = f"{self.oauth_server}/oauth2/token"
            else:
                data = {
                    "client_id": self.account,
                    "client_secret": self.api_key,
                    "grant_type": "client_credentials",
                }
                oauth_url = f"{self.oauth_server}/realms/airefinery-realm/protocol/openid-connect/token"
            response = requests.post(
                oauth_url,
                headers=headers,
                data=data,
                timeout=10,  # Specify an appropriate timeout
            )
            response.raise_for_status()
            response_json = response.json()
            access_token = response_json["access_token"]
            return access_token
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to login: %s", e)
            return ""

    def get_access_token(self) -> str:
        """
        Retrieve a valid access token, refreshing it if necessary.

        Returns:
            str: The valid access token.
        """
        if time.time() - self.time < self.timeout:
            assert self.access_token is not None
            return self.access_token
        self.access_token = self.login()
        self.time = time.time()
        return self.access_token

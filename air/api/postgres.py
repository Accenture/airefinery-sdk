import logging
import traceback
from typing import Any, Optional

import asyncpg

# Set up logging
logger = logging.getLogger(__name__)


class PostgresAPI:
    # pylint: disable=too-few-public-methods
    """
    A class to manage PostgreSQL database interactions using asynchronous connections.

    Attributes:
        config (dict): Configuration settings for the database connection, including
            "host", "port", "database", "user", and "password".
    """

    def __init__(self, db_config: dict) -> None:
        """
        Initialize the PostgresAPI object with database configuration settings.

        Args:
            db_config (dict): Configuration object containing database settings.
        """
        self.config = db_config

    async def execute_query(
        self, query: str, params: Optional[Any] = None
    ) -> tuple[list[Any] | str, bool]:
        """
        Execute a SQL query using an asynchronous PostgreSQL connection.

        Args:
            query (str): The SQL query to execute.
            params (Any, optional): The parameters to substitute into the query. Defaults to None.

        Returns:
            tuple[Optional[list[Any]], bool]:
                - If successful, returns (query_results, True), where query_results is a list of results.
                - If failed, returns (None, False).
        """
        query_results = []
        try:
            # Establish an asynchronous connection to the PostgreSQL database
            conn = await asyncpg.connect(
                host=self.config["host"],
                port=self.config["port"],
                database=self.config["database"],
                user=self.config["user"],
                password=self.config["password"],
            )
            try:
                # Execute the dynamic query
                if params is None:
                    query_results = await conn.fetch(query)
                else:
                    query_results = await conn.fetch(query, *params)  # type: ignore

                formatted_results = [tuple(record.values()) for record in query_results]
                return formatted_results, True
            except Exception as query_error:
                # Handle exceptions specifically related to query execution
                logger.error("Query execution failed: %s", query_error)
                query_traceback = traceback.format_exc()
                logger.debug("Query traceback:\n%s", query_traceback)
                return query_traceback, False
            finally:
                # Ensure the connection is closed after use
                await conn.close()
        except Exception as connection_error:
            # Handle exceptions related to establishing the connection
            logger.error("Failed to connect to the database: %s", connection_error)
            connection_traceback = traceback.format_exc()
            logger.debug("Connection traceback:\n%s", connection_traceback)
            return connection_traceback, False

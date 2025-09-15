"""
Azure AI Search module, supports upload and vector search
"""

import json
import logging
from typing import List

import requests
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from air.api.vector_db.base_vectordb import BaseVectorDB, VectorDBConfig
from air.embeddings import EmbeddingsClient

logger = logging.getLogger(__name__)


class AzureAISearch(BaseVectorDB):
    """
    Class to upload data to vector DB, inherits from BaseVectorDB
    """

    def __init__(self, vectordb_config: VectorDBConfig):
        super().__init__(vectordb_config)
        self.fields = vectordb_config.embedding_column
        self.k = vectordb_config.top_k
        self.select = ", ".join(vectordb_config.content_column)
        self.timeout = vectordb_config.timeout
        self.headers = {"Content-Type": "application/json", "api-key": self.api_key}
        self.search_url = f"{self.url}/indexes/{self.index}/docs/search?api-version={self.api_version}"
        self.index_url = (
            f"{self.url}/indexes/{self.index}/docs/index?api-version={self.api_version}"
        )

    @retry(
        retry=retry_if_exception_type(requests.exceptions.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=2, min=2, max=6),
    )
    def upload(self, rows: List[dict]) -> bool:
        """
        Function to upload list of document data to vector DB

        Args:
            rows (List[dict]): List of row dictionaries to be uploaded to the vector DB

        Returns:
            bool: Status of vector DB upload, False if failure, True if success
        """
        try:
            rows = [dict(row, **{"@search.action": "upload"}) for row in rows]
            data = {"value": rows}
            response = requests.post(
                self.index_url, headers=self.headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()
            return True
        except HTTPError as http_err:
            logger.error(
                "VectorDB upload request failed due to HTTP error: %s", http_err
            )
            return False

    @retry(
        retry=retry_if_exception_type(requests.exceptions.HTTPError),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=2, min=2, max=6),
    )
    def vector_search(
        self, query: str, embedding_client: EmbeddingsClient, embedding_model: str
    ) -> List[dict]:
        """
        Function to perform vector search over the index
        using the given query.

        Args:
            query (str): Query string which will be used to
            create a search vector to search over the vector DB index

        Returns:
            List[dict]: List of k vector db row dictionaries
            that were retrieved by the vector search
        """
        try:
            vector = (
                embedding_client.create(
                    input=[query],
                    model=embedding_model,
                    encoding_format="float",
                    extra_body={"input_type": "query"},
                )
                .data[0]
                .embedding
            )
        except HTTPError as http_err:
            logger.error(
                "Embedding generation request failed due to HTTP error: %s",
                http_err,
            )
            return []
        if not vector:
            logger.error("Embedding client did not return a response for the query.")
            return []

        try:
            search_vectors = [
                {
                    "kind": "vector",
                    "vector": vector,
                    "exhaustive": True,
                    "fields": self.fields,
                    "k": self.k,
                }
            ]
            data = {
                "count": True,
                "select": self.select,
                "vectorQueries": search_vectors,
            }
            response = requests.post(
                url=self.search_url,
                headers=self.headers,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response = json.loads(response.text)
            result = response["value"]
            return result
        except HTTPError as http_err:
            logger.error(
                "Azure AI Search DB search request failed due to HTTP error: %s",
                http_err,
            )
            logger.error("Failed to retrieve from Azure AI search API")
            return []

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

from air import auth
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
        rows = [dict(row, **{"@search.action": "upload"}) for row in rows]
        data = {"value": rows}
        response = requests.post(
            self.index_url, headers=self.headers, json=data, timeout=self.timeout
        )
        if response.status_code != 200:
            logger.error(
                "VectorDB upload request failed with status code: %s",
                response.status_code,
            )
            logger.error(response.reason)
            return False
        return True

    def get_query_embedding(
        self, query: str, embedding_client: EmbeddingsClient, embedding_model: str
    ) -> List:
        """
        Function to generate the embedding vector for a given query string
        """
        try:
            response = embedding_client.create(
                input=[query],
                model=embedding_model,
                encoding_format="float",
                extra_body={"input_type": "query"},
                extra_headers={"airefinery_account": auth.account},
            )
        except HTTPError as http_err:
            logger.error(
                "Embedding generation request failed due to HTTP error: %s",
                http_err,
            )
            return []
        embedding = response.data[0].embedding
        return embedding

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
        vector = self.get_query_embedding(query, embedding_client, embedding_model)
        if not vector:
            raise Exception("Embedding client did not return a response for the query.")

        search_vectors = [
            {
                "kind": "vector",
                "vector": vector,
                "exhaustive": True,
                "fields": self.fields,
                "k": self.k,
            }
        ]
        data = {"count": True, "select": self.select, "vectorQueries": search_vectors}
        response = requests.post(
            url=self.search_url, headers=self.headers, json=data, timeout=self.timeout
        )
        if not response.status_code == 200:
            logger.error(
                "VectorDB search request failed with status code: %s",
                response.status_code,
            )
            logger.error(response.reason)
        try:
            response = json.loads(response.text)
            result = response["value"]
            return result
        except Exception as e:
            logger.error(
                "An exception of type %s occurred: %s", type(e).__name__, str(e)
            )
            logger.error("Failed to retrieve from AI search API")
            return []

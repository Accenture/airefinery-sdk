import concurrent.futures
import logging
from typing import List, Tuple

from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
)
from tqdm import tqdm

from air import __base_url__, auth
from air.embeddings.client import EmbeddingsClient
from air.types import ClientConfig, Document, EmbeddingConfig

logger = logging.getLogger(__name__)


def is_unauthorized_error(exception):
    """
    Method to verify if exception is a 401 HTTP error
    """
    return isinstance(exception, HTTPError) and exception.response.status_code == 401


class Embedding:
    """
    Extends Executor to support data embedding functions.
    """

    def __init__(self, embedding_config: EmbeddingConfig, base_url: str):
        self.model = embedding_config.model
        self.batch_size = embedding_config.batch_size
        self.max_workers = embedding_config.max_workers
        self.base_url = base_url
        self.client_config = ClientConfig(**auth.openai(base_url=self.base_url))
        self.client = EmbeddingsClient(**dict(self.client_config))

    def refresh_client_access_token(self):
        """
        Refresh the access token for the OpenAI client.
        """
        self.client_config.api_key = auth.get_access_token()
        self.client = EmbeddingsClient(**dict(self.client_config))

    @retry(
        retry=retry_if_exception(is_unauthorized_error),
        stop=stop_after_attempt(2),
    )
    def generate_embeddings(self, data: List[Document]) -> Tuple[List[Document], bool]:
        """Function to upload data to Azure"""
        self.refresh_client_access_token()
        embeddings = []
        status = True
        texts = [doc.elements[0].text for doc in data]
        try:
            response = self.client.create(
                input=texts,
                model=self.model,
                encoding_format="float",
                extra_body={"input_type": "passage"},
                extra_headers={"airefinery_account": auth.account},
            )
            embeddings = [data.embedding for data in response.data]
            if None in embeddings:
                status = False
            for idx, doc in enumerate(data):
                doc.elements[0].text_vector = embeddings[idx]
        except HTTPError as http_err:
            logger.error(
                "Embedding generation request failed due to HTTP error: %s",
                http_err,
            )
            status = False
        except Exception as e:
            logger.error(
                "An exception of type %s occurred: %s", type(e).__name__, str(e)
            )
            status = False

        return data, status

    def run(self, document_list: List[Document]) -> Tuple[List[Document], bool]:
        """
        Function to create the embeddings for the given list of documents

        Args:
            document_list (List[Document]): List of Document class objects
            each containing one element which corresponds to one row in the Vector DB

        Returns:
            Tuple[List[Document], bool]: List of Document class objects with
            embeddings, and status of embedding generation
        """
        embedded_documents = []
        status = True
        batch_status = []
        batch_data = [
            document_list[idx : idx + self.batch_size]
            for idx in range(0, len(document_list), self.batch_size)
        ]
        logger.info("Generating embeddings...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            for embedded_rows, status in tqdm(
                executor.map(self.generate_embeddings, batch_data),
                total=len(batch_data),
            ):
                embedded_documents.extend(embedded_rows)
                batch_status.append(status)
        if False in batch_status:
            err_msg = "Some embeddings failed to generate"
            logger.error(err_msg)
            status = False
        return embedded_documents, status

import logging
import concurrent.futures
from typing import List, Tuple

from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI, AuthenticationError
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
)

from air import auth, __base_url__
from air.knowledge.schema import Document

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """
    Embedding configuration class
    """

    model: str = Field(..., description="Name of the model to use for embedding")
    batch_size: int = Field(
        default=50, description="Number of rows in a batch per embedding request"
    )
    max_workers: int = Field(
        default=8,
        description="Number of parallel threads to spawn while creating embeddings",
    )


class ClientConfig(BaseModel):
    """
    Configuration for the OpenAI client.
    """

    base_url: str = Field(
        default=__base_url__, description="Base URL for the OpenAI API"
    )
    api_key: str = Field(..., description="API key for authentication")
    default_headers: dict = Field(
        default_factory=dict, description="Default headers for API requests"
    )


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
        self.client = OpenAI(**dict(self.client_config))

    def refresh_client_access_token(self):
        """
        Refresh the access token for the OpenAI client.
        """
        self.client_config.api_key = auth.get_access_token()
        self.client = OpenAI(**dict(self.client_config))

    @retry(
        retry=retry_if_exception_type(AuthenticationError),
        stop=stop_after_attempt(2),
    )
    def generate_embeddings(self, data: List[Document]) -> Tuple[List[Document], bool]:
        """Function to upload data to Azure"""
        self.refresh_client_access_token()
        embeddings = []
        status = True
        texts = [doc.elements[0].text for doc in data]
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            if not getattr(response, "status_code", 200) == 200:
                logger.error(
                    "Embedding generation request failed with status code: %s",
                    getattr(response, "status_code"),
                )
                return data, False
            embeddings = [data.embedding for data in response.data]
            if None in embeddings:
                status = False
            for idx, doc in enumerate(data):
                doc.elements[0].text_vector = embeddings[idx]
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

"""
VectorDBUpload and its config class
"""

import concurrent.futures
import logging
from typing import List, Tuple

from tqdm import tqdm

from air.api.vector_db import VectorDBConfig, VectorDBRegistry
from air.types import Document, VectorDBUploadConfig

logger = logging.getLogger(__name__)


class VectorDBUpload:
    """
    Class to upload data to vector DB
    """

    def __init__(
        self, upload_config: VectorDBUploadConfig, vectordb_config: VectorDBConfig
    ):
        self.element_columns = ["id", "text", "text_vector"]
        self.metadata_columns = list(
            set(vectordb_config.content_column) - set(self.element_columns)
        )
        self.batch_size = upload_config.batch_size
        self.max_workers = upload_config.max_workers
        vectordb_class = VectorDBRegistry.get(vectordb_config.type)
        if not vectordb_class:
            raise KeyError("Vector DB type not found in the VectorDB registry")
        self.vectordb_client = vectordb_class(vectordb_config)

    def run(self, document_list: List[Document]) -> Tuple[None, bool]:
        """
        Function to upload list of document data to Vector DB

        Args:
            document_list (List[Document]): List of Document objects to be uploaded to the vector DB

        Returns:
            Tuple[None, bool]: Tuple containing None and status of vector DB upload,
                                False if failure, True if success
        """
        rows = []
        batch_results = []
        status = True
        for document in document_list:
            row = {
                column: getattr(document.elements[0], column)
                for column in self.element_columns
            }
            for column in self.metadata_columns:
                row[column] = document.metadata.get(column, "")
            rows.append(row)
        batch_data = [
            rows[idx : idx + self.batch_size]
            for idx in range(0, len(rows), self.batch_size)
        ]
        logger.info("Uploading documents to Vector DB...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            for result in tqdm(
                executor.map(self.vectordb_client.upload, batch_data),
                total=len(batch_data),
            ):
                batch_results.append(result)
        if False in batch_results:
            logger.error("Some rows were not uploaded to the Vector DB!")
            status = False
        return None, status

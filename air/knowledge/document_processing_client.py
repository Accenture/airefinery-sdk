"""
Document processing client
"""

import base64
import functools
import json
import logging
import os
from typing import Dict, List, Union

import requests

from air import __base_url__, auth
from air.api.vector_db import VectorDBRegistry
from air.knowledge.pipeline import (
    ChunkingRegistry,
    Embedding,
    VectorDBUpload,
)
from air.types import Document, DocumentProcessingConfig

logger = logging.getLogger(__name__)


class DocumentProcessingClient:
    """
    Interface for interacting with the AI Refinery's knowledge extraction service,
    allowing users to extract knowledge from input documents.
    """

    # Define API endpoint for extraction
    parse_document_suffix = "inference/document-extract"

    def __init__(
        self,
        *,
        base_url: str = "",
    ) -> None:
        """
        Initialize the DocumentProcessingClient with authentication details.

        Args:
            base_url (str, optional): Base URL for the API. Defaults to "".
        """
        # Use the provided base URL or the default one
        self.base_url = __base_url__ if base_url == "" else base_url

    def create_project(self, doc_process_config: DocumentProcessingConfig):
        """
        Initializes and sets up a document proccesing project based on the provided configuration.

        Args:
            doc_process_config (DocumentProcessingConfig):  Configuration for the document processing

        Raises:
            ValueError: If the specified vector db type is not registered.
        """

        # set up pipeline
        self.predifined_task_order = ["chunk", "embed", "upload"]
        self.doc_process_config = doc_process_config
        # init chunker
        selected_chunker = ChunkingRegistry.get(
            doc_process_config.chunking_config.algorithm
        )
        if not selected_chunker:
            raise KeyError(
                f"Chunking algorithm '{doc_process_config.chunking_config.algorithm}' \
                    not found in the Chunking registry"
            )
        self.chunker = selected_chunker(doc_process_config.chunking_config)
        # init embedder
        self.embedder = Embedding(
            embedding_config=doc_process_config.embedding_config,
            base_url=self.base_url,
        )
        # init uploader
        if not VectorDBRegistry.get(doc_process_config.vectordb_config.type):
            raise KeyError(
                f"Vector DB type '{doc_process_config.vectordb_config.type}' \
                    not found in the VectorDB registry"
            )
        self.uploader = VectorDBUpload(
            upload_config=doc_process_config.upload_config,
            vectordb_config=doc_process_config.vectordb_config,
        )
        # init executor dictionary
        self.exec_dict = {
            "chunk": functools.partial(self.chunker.run),
            "embed": functools.partial(self.embedder.run),
            "upload": functools.partial(self.uploader.run),
        }

    def parse_document(
        self, *, file_path: str, model: str, timeout: Union[int, None] = None
    ) -> dict:
        """
        Extract text/(multimedia) from the given document using the specified
        knowledge-extraction model.

        Args:
            file_path (str): local path of input file
            model (str): name of the knowledge extraction model to be used
            timeout Union[int, None] (defaults to None): timeout of the document extraction call
            in seconds, if set as None, default timeout configured for the model will be used.

        Returns:
            If success, return a dictionary of extracted document elements
            Otherwise, return a dictionary with key "error" and value being the detail reason
        """

        try:
            file_name = os.path.basename(file_path)
            file_type = os.path.splitext(file_name)[1][1:].upper()
            if model == "knowledge-brain/knowledge-brain":
                assert file_type in [
                    "PDF",
                    "PPTX",
                    "DOCX",
                    "DOC",
                    "PPT",
                ]
            elif model == "nv-ingest/nv-ingest":
                assert file_type in [
                    "PDF",
                    "PPTX",
                    "DOCX",
                ]
            else:
                err_msg = f"Not supported model {model}"
                logger.error(err_msg)
                raise ValueError
            with open(file_path, "rb") as fp:
                content = fp.read()
            file_base64 = base64.b64encode(content).decode("utf-8")
        except Exception:
            err_msg = f"{model} cannot handle the input file {file_path}"
            logger.error(err_msg)
            return {"error": err_msg}

        # Prepare the payload for the request
        payload = {
            "file": file_base64,
            "model": model,
            "file_type": file_type,
            "file_name": file_name,
            "timeout": timeout,
        }

        # Prepare the headers with the API key for authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth.get_access_token()}",
            "airefinery_account": auth.account,
        }

        # Determine the base URL
        base_url = f"{self.base_url}/{self.parse_document_suffix}"
        # Send a POST request to parse document
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        # Check the response status and return the result
        if response.status_code != 200:
            error_msg = (
                f"Failed to extract knowledge from {file_name}. {response.status_code}"
            )
            logger.error(error_msg)
            return {"error": error_msg}
        return json.loads(response.text)

    def pipeline(
        self, doc_list: List[Document], task_list: List[str]
    ) -> Dict[str, bool]:
        """
        RAG pipeline

        Perform a list of tasks specified by the user on a list of documents.

        Args:
            doc_list (List(Doument)):
                A list of Document to be processed

            task_list (List[str]):
                A list of tasks that user want to perform.
                Currently supported tasks are: "chunk", "embed", "upload"
                To be supported: "de-id", "translate"

        Returns:
            A dictionary indicating whether each task successfully done on all documents.
            True: completed successfully on all documents.
            False: otherwise

        """
        exec_status = {key: False for key in self.predifined_task_order}
        # check orders before execution
        if len(task_list) < 1 or len(doc_list) < 1:
            error_msg = "No task to perform!"
            logger.error(error_msg)
            return exec_status
        if not all(s in self.predifined_task_order for s in task_list):
            error_msg = f"At least one task not in the supported list {self.predifined_task_order}"
            logger.error(error_msg)
            return exec_status
        task_series_num = [self.predifined_task_order.index(t) for t in set(task_list)]
        sorted_task_num = sorted(task_series_num)
        if not all(
            sorted_task_num[i] + 1 == sorted_task_num[i + 1]
            for i in range(len(sorted_task_num) - 1)
        ):
            error_msg = "The list of tasks can not be executed."
            logger.error(error_msg)
            return exec_status

        # execute the user specified tasks in order
        proc_doc_list = doc_list
        for i in sorted_task_num:
            t = self.predifined_task_order[i]
            proc_doc_list, status = self.exec_dict[t](proc_doc_list)
            exec_status[t] = status
            if not status:
                error_msg = "Failed to complete all tasks."
                logger.error(error_msg)
                break

        return exec_status

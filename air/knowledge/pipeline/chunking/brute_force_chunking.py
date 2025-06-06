"""Brute force chunking"""

import logging
import uuid
from typing import List, Tuple

from tqdm import tqdm

from air.knowledge.pipeline.chunking.base_chunking import BaseChunking
from air.types import Document, TextElement

logger = logging.getLogger(__name__)


class BruteForceChunking(BaseChunking):
    """
    BruteForce Chunking strategy class

    Split text into fixed-length chunks with optional overlap.
    Instead of allowing nested document structures, each document
    will be split into multiple smaller documents, each containing
    only one text element.
    """

    def chunk_one_text_element(self, element: TextElement) -> List[TextElement]:
        """
        Given a long text element, split it into a list of smaller text elements,
        while retaining parent attributes such as page number and element type.
        """
        chunk_size = self.chunk_size
        overlap = self.overlap_size
        text = element.text

        chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)
        ]

        chunked_elements = [
            TextElement(
                id=str(uuid.uuid4()),  # Generate a unique ID for each chunk
                text=chunk,
                page_number=element.page_number,  # Retain original page number
                element_type=element.element_type,  # Retain original element type
            )
            for chunk in chunks
        ]
        return chunked_elements

    def chunk_one_document(
        self, document: Document, retain_metadata=True
    ) -> List[Document]:
        """
        Given a document with multiple text elements, split each text element
        into chunks and return a list of smaller documents, each with a single element.

        Args:
            document: The input document to be chunked.
            retain_metadata (bool): Whether to retain document-level metadata.

        Returns:
            List[Document]: A list of smaller documents, each containing one chunked element.
        """
        chunked_documents = []
        for text_element in document.elements:
            chunked_elements = self.chunk_one_text_element(text_element)

            for chunk in chunked_elements:
                new_doc = Document(
                    filename=document.filename,
                    elements=[chunk],  # Each new document has only one element
                    file_type=document.file_type,
                    metadata=(
                        document.metadata if retain_metadata else {}
                    ),  # Preserve metadata if enabled
                )
                chunked_documents.append(new_doc)

        return chunked_documents

    def run(
        self, documents: List[Document], retain_metadata=True
    ) -> Tuple[List[Document], bool]:
        """
        Process a list of documents and return a flat list of chunked documents.

        Args:
            documents (List[Document]): A list of input documents where some may have long content.
            retain_metadata (bool): Whether to retain document-level metadata.

        Returns:
            List[Document]: A larger list of smaller documents where each document contains only
            one chunked text element.
        """
        chunked_documents = []
        logger.info("Chunking documents...")
        for doc in tqdm(documents):
            chunked_documents.extend(self.chunk_one_document(doc, retain_metadata))

        return chunked_documents, True

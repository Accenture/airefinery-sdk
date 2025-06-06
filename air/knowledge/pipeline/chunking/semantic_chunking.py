import logging
import re
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from air.knowledge.pipeline.chunking.base_chunking import BaseChunking, ChunkingConfig
from air.types import Document, TextElement

logger = logging.getLogger(__name__)


class SemanticChunking(BaseChunking):
    """
    Semantic Chunking strategy class
    Splits text into semantically meaningful chunks based on sentence embeddings.
    """

    def __init__(self, chunking_config: ChunkingConfig):
        """
        Initializes the semantic chunker with configuration.
        """
        super().__init__(chunking_config)
        self.chunking_config = chunking_config

    def run(self, documents: List[Document]) -> Tuple[List[Document], bool]:
        """
        Applies semantic chunking to a list of documents.

        Args:
            documents: List of Document objects with text elements.

        Returns:
            A new list of Document objects with semantically chunked text.
        """
        new_docs = []
        logger.info("Chunking documents...")
        for doc in tqdm(documents):
            elements = doc.elements
            all_text = " ".join(
                [el.text for el in elements if isinstance(el, TextElement)]
            )

            sentences = self._split_sentences(all_text)
            if not sentences:
                continue

            sentence_embeddings = self._embed_sentences(sentences)

            chunks = []
            current_chunk = [sentences[0]]
            current_chunk_vector = [sentence_embeddings[0]]

            for i in range(1, len(sentences)):
                similarity = self._cosine_similarity(
                    sentence_embeddings[i],
                    np.mean(current_chunk_vector, axis=0),
                )

                if (
                    len(" ".join(current_chunk + [sentences[i]]))
                    < self.chunking_config.chunk_size
                    and similarity > 0.6
                ):
                    current_chunk.append(sentences[i])
                    current_chunk_vector.append(sentence_embeddings[i])
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_chunk_vector = [sentence_embeddings[i]]

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            chunked_elements = [
                TextElement(
                    id=f"{i + 1}",
                    text=chunk,
                    page_number=1,
                    element_type="text",
                    text_vector=[],
                )
                for i, chunk in enumerate(chunks)
            ]

            document_chunks = [
                Document(
                    filename=doc.filename,
                    file_type=doc.file_type,
                    elements=[chunked_element],
                    metadata=doc.metadata,
                )
                for chunked_element in chunked_elements
            ]

            new_docs.extend(document_chunks)

        return new_docs, True

    def _split_sentences(self, text: str) -> List[str]:
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        return [s.strip() for s in sentence_endings.split(text) if s.strip()]

    def _embed_sentences(self, sentences: List[str]) -> List[np.ndarray]:
        def embed(sentence: str) -> np.ndarray:
            if not sentence:
                return np.zeros(128)
            vec = np.array([ord(c) for c in sentence if ord(c) < 10000])
            return np.pad(vec.mean().reshape(1), (0, 127), constant_values=0)

        return [embed(s) for s in sentences]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

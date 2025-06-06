from typing import List, Literal

from pydantic import BaseModel, Field

from air import __base_url__
from air.api.vector_db.base_vectordb import VectorDBConfig


# knowledge graph config
class KnowledgeGraphConfig(BaseModel):
    """
    KnowledgeGraph configuration class
    """

    type: str = Field(default="GraphRAG", description="Type of the Knowledge Graph")
    work_dir: str = Field(
        default="graph_dir", description="Workspace directory for the knowledge graph"
    )
    api_type: Literal["openai", "azure"] = Field(
        default="openai",
        description="API type of deployed LLM",
    )
    chunk_size: int = Field(default=1200, description="Size of text chunks")
    chunk_overlap: int = Field(default=100, description="Overlap between text chunks")
    llm_model: str = Field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        description="LLM model to use for knowledge graph tasks",
    )
    embedding_model: str = Field(
        default="intfloat/e5-mistral-7b-instruct",
        description="Embedding model to use for knowledge graph tasks",
    )


# chunk config
class ChunkingConfig(BaseModel):
    """
    Chunking configuration class
    """

    algorithm: str = Field(
        default="BruteForceChunking", description="Type of Chunking Algorithm"
    )
    chunk_size: int = Field(..., description="Max length per chunk")
    overlap_size: int = Field(
        default=0, description="Overlap between two neighboring chunks"
    )


# embedding config
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


# upload config
class VectorDBUploadConfig(BaseModel):
    """
    VectorDB upload configuration class
    """

    batch_size: int = Field(
        default=50, description="Number of rows in a batch per upload request"
    )
    max_workers: int = Field(
        default=8,
        description="Number of parallel threads to spawn while uploading rows to vector DB",
    )


# document processing client
class DocumentProcessingConfig(BaseModel):
    """
    Configuration for document processing
    """

    upload_config: VectorDBUploadConfig = Field(
        default=VectorDBUploadConfig(), description="Vector DB upload configuration"
    )
    vectordb_config: VectorDBConfig = Field(..., description="Vector DB configuration")
    embedding_config: EmbeddingConfig = Field(
        ..., description="Embedding configuration"
    )
    chunking_config: ChunkingConfig = Field(
        ..., description="Chunking parameter configuration"
    )


# document objects
class TextElement(BaseModel):
    """
    Document element data config

    Attributes:
        id (str): Unique identifier for the element
        text (str): Text of the element
        page_number (int): Document page number from which element was extracted
        element_type (str): Type of element, one of (text, table, figure)
        text_vector (list): Embedding Vector for the element text
    """

    id: str = Field(..., description="Unique identifier for the element")
    text: str = Field(..., description="Text from the element")
    page_number: int = Field(
        ..., description="Document page number from which element was extracted"
    )
    element_type: Literal["text", "table", "figure"] = Field(
        ..., description="Type of element"
    )
    text_vector: List = Field(
        default=[], description="Embedding Vector for the element text"
    )


class Document(BaseModel):
    """
    Document Object data class.

    Attributes:
        filename (str): Name of the file
        file_type (str): File type/extension
        elements (list): List of file elements
        metadata (dict): Metadata related to the document
    """

    filename: str = Field(..., description="Name of the file")
    file_type: str = Field(..., description="File type/extension")
    elements: List[TextElement] = Field(..., description="List of document elements")
    metadata: dict = Field(default={}, description="Metadata related to the document")

"""Definition to document"""

from typing import List, Literal
from pydantic import BaseModel, Field


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

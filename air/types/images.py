"""
Pydantic models for the OpenAI-style Images **generate** response.

Provides:

- Image: a single generated image record.
- Usage: token-usage statistics for an image generation request.
- ImagesResponse: the top-level container for an Images.generate call.
"""

from typing import List, Dict, Optional
from air.types.base import CustomBaseModel


class Image(CustomBaseModel):
    """Represents one generated image and its metadata.

    Attributes:
        b64_json: Base64-encoded image data (only present when `response_format="b64_json"`).
        revised_prompt: The final prompt string the model actually used for image generation,
                        which may be None if no revision was applied.
        url: Publicly accessible URL of the generated image (only present when `response_format="url"`).
    """

    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    url: Optional[str] = None


class Usage(CustomBaseModel):
    """Represents token-usage statistics for an image request.

    Attributes:
        input_tokens: The number of tokens (images and text) in the input prompt.
        input_tokens_details: The input tokens detailed information for the image generation.
        output_tokens: The number of image tokens in the output image.
        total_tokens: The total number of tokens (images and text) used for the image generation.
    """

    input_tokens: int
    input_tokens_details: Dict[str, int]
    output_tokens: int
    total_tokens: int


class ImagesResponse(CustomBaseModel):
    """Represents the full response returned by the Images *generate* endpoint.

    Attributes:
        created: The Unix timestamp (in seconds) of when the images were created.
        data: A list of generated images.
        usage: Aggregate token-usage information for the request (optional).
    """

    created: int
    data: List[Image]
    usage: Optional[Usage] = None

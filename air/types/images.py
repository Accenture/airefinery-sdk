"""
Pydantic models for the OpenAI-style Images **generate** response
and for the point-prompt based SAM-HQ segmentation response

Provides:

- Image: a single generated image record
- Usage: token-usage statistics for an image generation/segmentation request
- ImagesResponse: the top-level container for an Images.generate call
- Mask: a single generated segmentation mask
- SegmentationResponse: the top-level container for an Images.segment call

"""

from typing import Dict, List, Optional

from air.types.base import CustomBaseModel


class Image(CustomBaseModel):
    """Represents one generated image and its metadata.

    Attributes:
        b64_json: Base64-encoded image data (only present when `response_format="b64_json"`)
        revised_prompt: The final prompt string the model actually used for image generation,
                        which may be None if no revision was applied
        url: Publicly accessible URL of the generated image (only present when
            `response_format="url"`)
    """

    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    url: Optional[str] = None


class Usage(CustomBaseModel):
    """Represents token-usage statistics for image related requests.

    Attributes:
        input_tokens: The number of tokens (images and text) in the input prompt
        input_tokens_details: The input tokens detailed information for the image generation
        output_tokens: The number of image tokens in the output image
        total_tokens: The total number of tokens (images and text) used for the image generation
    """

    input_tokens: int
    input_tokens_details: Dict[str, int]
    output_tokens: int
    total_tokens: int


class ImagesResponse(CustomBaseModel):
    """Represents the full response returned by the Images *generate* endpoint.

    Attributes:
        created: The Unix timestamp (in seconds) of when the images were created
        data: A list of generated images
        usage: Aggregate token-usage information for the request (optional)
    """

    created: int
    data: List[Image]
    usage: Optional[Usage] = None


class Mask(CustomBaseModel):
    """Represents one segmentation mask and its metadata.

    Attributes:
        b64_json: Base64-encoded categorical mask image
        label: Optional semantic class label, if provided
        score: Optional confidence score from the model, if provided
    """

    b64_json: Optional[str] = None
    label: Optional[str] = None
    score: Optional[float] = None


class SegmentationResponse(CustomBaseModel):
    """Represents the full response returned by the point-prompt segmentation
    endpoint

    Attributes:
        created: The Unix timestamp (in seconds) when the masks were created
        data: A list of generated masks
        usage: Aggregate token-usage information for the request (optional)
    """

    created: int
    data: List[Mask]
    usage: Optional[Usage] = None

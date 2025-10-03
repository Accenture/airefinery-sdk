"""
Pydantic models for Moderation responses and content safety analysis.

This module provides:
  • Categories                    – Per-category violation flags (harassment, hate, violence, etc.)
  • CategoryAppliedInputTypes     – Input types flagged for each category (text, image)
  • CategoryScores               – Confidence scores for policy violations (0.0-1.0)
  • Moderation                   – Complete moderation result for a single input
  • ModerationCreateResponse     – Top-level response from the moderations endpoint
"""

from typing import List, Optional

from pydantic import Field
from typing_extensions import Literal

from air.types.base import CustomBaseModel


class Categories(CustomBaseModel):
    """
    Contains a dictionary of per-category violation flags.
    For each category, the value is true if the model
    flags the corresponding category as violated, false otherwise.
    """

    harassment: bool = Field(
        description="Content that expresses, incites, \
                    or promotes harassing language towards any target."
    )

    harassment_threatening: bool = Field(
        alias="harassment/threatening",
        description="Harassment content that also includes \
                    violence or serious harm towards any target.",
    )

    hate: bool = Field(
        description="Content that expresses, incites, or promotes hate \
                    based on race, gender, ethnicity, religion, \
                    nationality, sexual orientation, disability status, \
                    or caste. Hateful content aimed at non-protected \
                    groups (e.g., chess players) is harassment."
    )

    hate_threatening: bool = Field(
        alias="hate/threatening",
        description="Hateful content that also includes violence \
                    or serious harm towards the targeted group based on \
                    race, gender, ethnicity, religion, nationality, \
                    sexual orientation, disability status, or caste.",
    )

    illicit: Optional[bool] = Field(
        default=None,
        description="Content that includes instructions or advice \
                    that facilitate the planning or execution of wrongdoing, \
                    or that gives advice or instruction on how to commit illicit acts. \
                    For example, 'how to shoplift' would fit this category.",
    )

    illicit_violent: Optional[bool] = Field(
        alias="illicit/violent",
        default=None,
        description="Content that includes instructions or advice \
                    that facilitate the planning or execution of wrongdoing \
                    that also includes violence, or that gives advice \
                    or instruction on the procurement of any weapon.",
    )

    self_harm: bool = Field(
        alias="self-harm",
        description="Content that promotes, encourages, or depicts acts \
                    of self-harm, such as suicide, cutting, and eating disorders.",
    )

    self_harm_instructions: bool = Field(
        alias="self-harm/instructions",
        description="Content that encourages performing acts of self-harm, \
                    such as suicide, cutting, and eating disorders, \
                    or that gives instructions or advice on how to commit such acts.",
    )

    self_harm_intent: bool = Field(
        alias="self-harm/intent",
        description="Content where the speaker expresses that \
                    they are engaging or intend to engage in acts of self-harm, \
                    such as suicide, cutting, and eating disorders.",
    )

    sexual: bool = Field(
        description="Content meant to arouse sexual excitement, \
                    such as the description of sexual activity, \
                    or that promotes sexual services \
                    (excluding sex education and wellness)."
    )

    sexual_minors: bool = Field(
        alias="sexual/minors",
        description="Sexual content that includes an individual \
                    who is under 18 years old.",
    )

    violence: bool = Field(
        description="Content that depicts death, violence, or physical injury."
    )

    violence_graphic: bool = Field(
        alias="violence/graphic",
        description="Content that depicts death, violence, or physical injury in graphic detail.",
    )


class CategoryAppliedInputTypes(CustomBaseModel):
    """
    This property contains information on which input types were flagged in the response,
    for each category. For example, if the both the image and text inputs to the model are flagged for "violence/graphic",
    the violence/graphic property will be set to ["image", "text"].
    """

    harassment: List[Literal["text"]] = Field(
        description="The applied input type(s) for the category 'harassment'."
    )

    harassment_threatening: List[Literal["text"]] = Field(
        alias="harassment/threatening",
        description="The applied input type(s) for the category 'harassment/threatening'.",
    )

    hate: List[Literal["text"]] = Field(
        description="The applied input type(s) for the category 'hate'."
    )

    hate_threatening: List[Literal["text"]] = Field(
        alias="hate/threatening",
        description="The applied input type(s) for the category 'hate/threatening'.",
    )

    illicit: List[Literal["text"]] = Field(
        description="The applied input type(s) for the category 'illicit'."
    )

    illicit_violent: List[Literal["text"]] = Field(
        alias="illicit/violent",
        description="The applied input type(s) for the category 'illicit/violent'.",
    )

    self_harm: List[Literal["text", "image"]] = Field(
        alias="self-harm",
        description="The applied input type(s) for the category 'self-harm'.",
    )

    self_harm_instructions: List[Literal["text", "image"]] = Field(
        alias="self-harm/instructions",
        description="The applied input type(s) for the category 'self-harm/instructions'.",
    )

    self_harm_intent: List[Literal["text", "image"]] = Field(
        alias="self-harm/intent",
        description="The applied input type(s) for the category 'self-harm/intent'.",
    )

    sexual: List[Literal["text", "image"]] = Field(
        description="The applied input type(s) for the category 'sexual'."
    )

    sexual_minors: List[Literal["text"]] = Field(
        alias="sexual/minors",
        description="The applied input type(s) for the category 'sexual/minors'.",
    )

    violence: List[Literal["text", "image"]] = Field(
        description="The applied input type(s) for the category 'violence'."
    )

    violence_graphic: List[Literal["text", "image"]] = Field(
        alias="violence/graphic",
        description="The applied input type(s) for the category 'violence/graphic'.",
    )


class CategoryScores(CustomBaseModel):
    """
    Contains a dictionary of per-category scores output by the model,
    denoting the model's confidence that the input violates the OpenAI's policy for the category.
    The value is between 0 and 1, where higher values denote higher confidence.
    """

    harassment: float = Field(description="The score for the category 'harassment'.")

    harassment_threatening: float = Field(
        alias="harassment/threatening",
        description="The score for the category 'harassment/threatening'.",
    )

    hate: float = Field(description="The score for the category 'hate'.")

    hate_threatening: float = Field(
        alias="hate/threatening",
        description="The score for the category 'hate/threatening'.",
    )

    illicit: float = Field(description="The score for the category 'illicit'.")

    illicit_violent: float = Field(
        alias="illicit/violent",
        description="The score for the category 'illicit/violent'.",
    )

    self_harm: float = Field(
        alias="self-harm", description="The score for the category 'self-harm'."
    )

    self_harm_instructions: float = Field(
        alias="self-harm/instructions",
        description="The score for the category 'self-harm/instructions'.",
    )

    self_harm_intent: float = Field(
        alias="self-harm/intent",
        description="The score for the category 'self-harm/intent'.",
    )

    sexual: float = Field(description="The score for the category 'sexual'.")

    sexual_minors: float = Field(
        alias="sexual/minors", description="The score for the category 'sexual/minors'."
    )

    violence: float = Field(description="The score for the category 'violence'.")

    violence_graphic: float = Field(
        alias="violence/graphic",
        description="The score for the category 'violence/graphic'.",
    )


class Moderation(CustomBaseModel):
    """Represents a single moderation from a Moderations response.

    Attributes:
        categories: A list of the categories, and whether they are flagged or not.
        category_applied_input_types: A list of the categories along with the input type(s) that the score applies to.
        category_scores: A list of the categories along with their scores as predicted by model.
        flagged: Whether any of the below categories are flagged.
    """

    categories: Categories = Field(
        description="A list of the categories, and whether they are flagged or not."
    )

    category_applied_input_types: CategoryAppliedInputTypes = Field(
        description="A list of the categories along with the input type(s) that the score applies to."
    )

    category_scores: CategoryScores = Field(
        description="A list of the categories along with their scores as predicted by model."
    )

    flagged: bool = Field(
        description="Whether any of the below categories are flagged."
    )


class ModerationCreateResponse(CustomBaseModel):
    """Top-level Moderation response returned by the API.

    Attributes:
        id: Unique identifier for this Moderation request.
        model: The language model used.
        results: A list of moderation objects.
    """

    id: str = Field(description="The unique identifier for the moderation request.")

    model: str = Field(description="The model used to generate the moderation results.")

    results: List[Moderation] = Field(description="A list of moderation objects.")

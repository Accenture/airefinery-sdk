from typing import Dict

from pydantic import BaseModel, Field

USER_INPUT_METHOD_DEFAULT = "Terminal"


class FeedbackFieldConfig(BaseModel):
    """Schema field configuration for expected user feedback.

    Attributes:
        type (str): The type of the field (e.g., "str", "int").
        description (str): Description of what the field represents.
        required (bool): Whether the field is mandatory in the feedback.
    """

    type: str = Field(
        default="",
        description="The type of the field.",
    )
    description: str = Field(
        default="",
        description="The description of the field.",
    )
    required: bool = Field(
        default=True,
        description="Whether the field is mandatory in the feedback",
    )


class HumanAgentConfig(BaseModel):
    """
    Configuration settings for the Human Agent.
    """

    wait_time: int = Field(
        default=300,
        description="Maximum wait time (in seconds) for human input before timing out.",
    )
    user_input_method: str = Field(
        default=USER_INPUT_METHOD_DEFAULT,
        description="Method used to collect human input: 'Terminal' or 'Custom'.",
    )
    feedback_interpreter: bool = Field(
        default=True,
        description="Whether to convert structured feedback defined in the schema into natural language.",
    )
    feedback_schema: Dict[str, FeedbackFieldConfig] = Field(
        default_factory=dict,
        description="Schema defining the expected feedback fields from user input.",
    )

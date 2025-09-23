from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from air.types.base import CustomBaseModel


class FineTuningRequest(CustomBaseModel):
    """Represents a fine-tuning job and its metadata.

    Attributes:
        job_id: The object identifier, which can be referenced in the API endpoints.
        job_description: Description of the job for user's recognition.
        user_id: Unique identifier for the user.
        method: The method used for fine-tuning, either 'supervised' or 'dpo'.
        created_at: Unix timestamp (in seconds) for when the fine-tuning job was created.
        error: For failed jobs, contains more information on the cause of failure.
        fine_tuned_model: Name of the fine-tuned model being created;
                          null if the job is still running.
        finished_at: Unix timestamp (in seconds) for when the job was finished;
                     null if the job is still running.
        train_config: The hyperparameters used for supervised fine-tuning jobs.
        model: The base model that is being fine-tuned.
        status: Current status of the job: 'queued', 'running', 'succeeded',
                'failed', or 'cancelled'.
        training_file: The file ID used for training.
        validation_file: The file ID used for validation, if provided.
        seed: The seed used for the fine-tuning job.
    """

    job_id: str = Field(
        ...,
        description="Unique identifier for the job. "
        "(e.g., 'supervised-b9872adb-61a3-4aec-8f30-341f84cd545c-1755827328')",
    )
    job_description: str = Field(
        ...,
        description="Description of the job.",
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user. " "(e.g., 'test_user')",
    )
    method: Literal["supervised", "dpo"] = Field(
        ...,
        description="The method used for fine-tuning.",
    )
    created_at: str = Field(
        ...,
        description="Unix timestamp (in seconds) for when the fine-tuning job was created. "
        "(e.g., August 22, 2025 at 1:48:48 AM UTC)",
    )
    error: Optional[str] = Field(
        None,
        description="For failed jobs, contains more information on the cause of failure.",
    )
    fine_tuned_model: Optional[str] = Field(
        None,
        description="Name of the fine-tuned model being created.",
    )
    finished_at: Optional[str] = Field(
        None,
        description="Unix timestamp (in seconds) for when the job was finished.",
    )
    train_config: Dict[str, Any] = Field(
        ..., description="The hyperparameters used for supervised fine-tuning jobs."
    )
    model: str = Field(..., description="The base model that is being fine-tuned.")
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = Field(
        ...,
        description="Current status of the job. (only valid in these five status)",
    )
    training_file: str = Field(..., description="The file ID used for training.")
    validation_file: Optional[str] = Field(
        None, description="The file ID used for validation, if provided."
    )
    seed: int = Field(..., description="The seed used for the fine-tuning job.")


class FineTuningJobConfig(BaseModel):
    """
    Configuration for fine-tuning job submitted by users..
    """

    description: Optional[str] = Field(
        ...,  # required
        description="Description of the fine-tuning job    .",
    )
    method: Optional[str] = Field(
        ...,
        description="Method used for fine-tuning (e.g., 'supervised', 'dpo').",  # required
    )
    train_config: Optional[Dict[str, Any]] = Field(
        ..., description="The hyperparameters used for fine-tuning jobs."
    )

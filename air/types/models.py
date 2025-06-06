"""
Typed Pydantic schemas for the List Models endpoint (`client.models.list()`).

Public classes:
  • ModelPermission
  • Model
"""

from typing import List, Optional

from air.types.base import CustomBaseModel


class ModelPermission(CustomBaseModel):
    """Represents a fine-grained permission set attached to a model.

    Attributes:
        id: Unique identifier of this permission object.
        object: Always "model_permission".
        created: Unix timestamp indicating when this permission was created.
        allow_create_engine: Whether creating an engine is allowed.
        allow_sampling: Whether sampling is allowed.
        allow_logprobs: Whether viewing log probabilities is allowed.
        allow_search_indices: Whether search indices are allowed.
        allow_view: Whether viewing the model is allowed.
        allow_fine_tuning: Whether fine-tuning is allowed.
        organization: The owning organization.
        group: An optional group name.
        is_blocking: True if this permission blocks access.
    """

    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: Optional[str]
    is_blocking: bool


class Model(CustomBaseModel):
    """Represents metadata describing a single model.

    Attributes:
        id: The model identifier (e.g., "gpt-4o-mini").
        created: The Unix timestamp when the model was created.
        object: Always "model".
        owned_by: The organization that owns this model.
        root: The root model identifier.
        parent: The parent model identifier, or None if not applicable.
        permission: A list of permission objects for this model.
    """

    id: str
    created: int
    object: str
    owned_by: str
    root: str
    parent: Optional[str]
    permission: List[ModelPermission]

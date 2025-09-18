"""
This module defines the schema used for handling output from
DeepResearchAgent (DRA) sub-agents.

This module provides Pydantic models and enums that standardize how intermediate
outputs and final results are structured and communicated back to the client.

It defines:
- Core Enums
  - `DeepResearchStep`:
        String mappings for main API (step function) calls to `OutputHandler`
  - `DeepResearchStatus`:
        High-level status types (e.g., pipeline steps, progress, references).
- Payload Models
  - `DeepResearchPipelineStepPayload`:
        Represents a major pipeline stage and its display message.
  - `DeepResearchIRProgressPayload`:
        Tracks progress across iterative research tasks.
  - `DeepResearchResearchQuestionsPayload`:
        Outputs generated research questions from the Research Planner.
  - `DeepResearchThoughtStatusPayload`:
        Provides concise updates on intermediate reasoning (in Iterative Researcher).
  - `DeepResearchReferencePayload`:
        Streams discovered references with associated question IDs (in Iterative Researcher).
  - `DeepResearchSummaryStatisticsPayload`:
        Reports overall runtime statistics and resource usage.
"""

from enum import Enum
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field


class DeepResearchStep(Enum):
    """
    Enumeration of all pipeline steps used in the Deep Research Agent.

    This enum standardizes the step identifiers used when calling OutputHandler.step()
    to prevent typos and improve type safety.
    """

    # Follow-up steps
    START_FOLLOW_UP = "start_follow_up"
    END_FOLLOW_UP_POS = "end_follow_up_pos"
    END_FOLLOW_UP_NEG = "end_follow_up_neg"
    FAIL_CLARIFICATION = "fail_clarification"

    # Query rewriter steps
    START_QUERY_REWRITER = "start_query_rewriter"
    END_QUERY_REWRITER = "end_query_rewriter"
    END_QUERY_REWRITER_NO_FEEDBACK = "end_query_rewriter_no_feedback"

    # Background search steps
    START_SEARCH_BACKGROUND = "start_search_background"
    END_SEARCH_BACKGROUND = "end_search_background"
    FAIL_SEARCH_BACKGROUND = "fail_search_background"

    # Research planner steps
    START_RESEARCH_PLANNER = "start_research_planner"
    END_RESEARCH_PLANNER = "end_research_planner"
    FAIL_RESEARCH_PLANNER = "fail_research_planner"

    # Iterative research steps
    START_ITERATIVE_RESEARCH = "start_iterative_research"
    ITERATIVE_RESEARCH_TASK_FAILED = "iterative_research_task_failed"
    ITERATIVE_RESEARCH_PIPELINE_ABORTED = "iterative_research_pipeline_aborted"

    # Author steps
    START_AUTHOR = "start_author"
    END_AUTHOR = "end_author"
    FAIL_AUTHOR = "fail_author"

    # Audio steps
    START_AUDIO = "start_audio"
    END_AUDIO = "end_audio"
    FAIL_AUDIO = "fail_audio"

    # Dedicated handler steps
    THOUGHT_STATUS = "thought_status"
    LOG_REFS = "log_refs"
    PRINT_TASK = "print_task"
    END_PIPELINE = "end_pipeline"


class DeepResearchThoughtStatusOutput(BaseModel):
    """
    Represents a concise status summary of a single thought from the Iterative Researcher.
    """

    status: str = Field(
        description="A short, human-readable summary of the current thought state."
    )


class DeepResearchStatus(str, Enum):
    """
    Enum of high-level status types used by the Deep Research Agent (DRA)
    to structure intermediate outputs and client-side display.

    Attributes:
        PIPELINE_STEP: A major stage in the research pipeline (e.g., "FollowUpAgent").
        IR_PROGRESS: Iteration progress within the Iterative Researcher.
        RESEARCH_QUESTIONS: A list of generated research questions.
        THOUGHT_STATUS: Status of a specific thought or reasoning step.
        REFERENCE: Sources or references retrieved during the search phase.
        SUMMARY_STATISTICS: Runtime and usage metrics for the overall pipeline.
    """

    PIPELINE_STEP = "pipeline_step"
    IR_PROGRESS = "ir_progress"
    RESEARCH_QUESTIONS = "research_questions"
    THOUGHT_STATUS = "thought_status"
    REFERENCE = "reference"
    SUMMARY_STATISTICS = "summary_statistics"


class DeepResearchPipelineStepPayload(BaseModel):
    """
    Payload representing a high-level pipeline step and its description.
    """

    type: Literal["pipeline_step"] = Field(
        description="Constant indicating that this payload represents a pipeline step update."
    )
    step_key: DeepResearchStep = Field(
        description="Unique identifier for the pipeline step (e.g., 'start_follow_up', 'end_author')."
    )
    info: str = Field(description="Human-readable message describing the current step.")


class DeepResearchIRProgressPayload(BaseModel):
    """
    Payload representing progress through the Iterative Researcher steps.
    """

    type: Literal["ir_progress"] = Field(
        description="Constant indicating that this payload reports iterative research progress."
    )
    processed_tasks: int = Field(
        description="Number of research tasks or subtasks that have been processed so far."
    )
    total_task: int = Field(description="Total number of research tasks planned.")


class DeepResearchResearchQuestionsPayload(BaseModel):
    """
    Payload containing research questions generated during planning.
    """

    type: Literal["research_questions"] = Field(
        description="Constant indicating that this payload contains research questions output from the planning step."
    )
    questions: List[str] = Field(
        description="List of generated research questions to guide subsequent iterative research."
    )


class DeepResearchThoughtStatusPayload(BaseModel):
    """
    Payload describing the status of a specific thought.
    """

    type: Literal["thought_status"] = Field(
        description="Constant indicating that this payload contains an intermediate thought status update."
    )
    question_id: int = Field(
        description="Unique identifier of the research question this thought status relates to."
    )
    thought: str = Field(
        description="Human-readable summary of the current reasoning or thought process for the question."
    )


class DeepResearchReferencePayload(BaseModel):
    """
    Payload containing references retrieved during the search process.
    """

    type: Literal["reference"] = Field(
        description="Constant indicating that this payload contains search reference data."
    )
    question_id: int = Field(
        description="Unique identifier of the research question associated with these references."
    )
    references: dict[str, str] = Field(
        description="Mapping of reference URLs to their concise descriptions or titles."
    )


class DeepResearchSummaryStatisticsPayload(BaseModel):
    """
    Payload containing summary statistics for the overall deep research session.
    """

    type: Literal["summary_statistics"] = Field(
        description="Constant indicating that this payload contains overall session summary statistics."
    )
    used_time: float = Field(
        description="Total time spent on the deep research session in minutes."
    )
    website_num: int = Field(
        description="Total number of unique websites visited during the session."
    )


DeepResearchPayloadType = Annotated[
    Union[
        DeepResearchPipelineStepPayload,
        DeepResearchIRProgressPayload,
        DeepResearchResearchQuestionsPayload,
        DeepResearchThoughtStatusPayload,
        DeepResearchReferencePayload,
        DeepResearchSummaryStatisticsPayload,
    ],
    Field(discriminator="type"),
]

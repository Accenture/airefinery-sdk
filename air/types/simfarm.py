"""Pydantic models for SimFarm operations."""

from enum import Enum
from pydantic import Field
from typing import List, Optional

from air.types.base import CustomBaseModel


class SimfarmStatusEnum(str, Enum):
    """Status enumeration for simulation jobs and tasks."""

    PENDING = "Pending"
    RUNNING = "Running"
    SUCCESS = "Success"
    SUBMITTED = "Submitted"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class SimfarmJobInfo(CustomBaseModel):
    """Job status information and metadata."""

    job_id: Optional[str] = Field(default=None, description="Job ID")
    status: Optional[str] = Field(default=None, description="Current status")
    created_at: Optional[str] = Field(default=None, description="Creation time")
    updated_at: Optional[str] = Field(default=None, description="Update time")
    input_file_names: Optional[List[str]] = Field(
        default=None, description="List of input file names"
    )


class SimfarmTaskDetail(CustomBaseModel):
    """Individual task details within a simulation job."""

    input_file_name: str = Field(description="Input file name for this task")
    sim_task_id: str = Field(description="Simulation task ID")
    status: SimfarmStatusEnum = Field(description="Task status")
    created_at: str = Field(description="Creation time")
    updated_at: str = Field(description="Update time")


class SimfarmJobDetailsResponse(CustomBaseModel):
    """Response from getting detailed job information."""

    job_info: Optional[SimfarmJobInfo] = Field(
        default=None, description="Job information"
    )
    status: Optional[str] = Field(default=None, description="Job status")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Update timestamp")
    total_tasks: Optional[int] = Field(default=None, description="Total task count")
    task_details: Optional[List[SimfarmTaskDetail]] = Field(
        default=None, description="Task details list"
    )


class SimfarmJobListResponse(CustomBaseModel):
    """Response from listing all simulation jobs."""

    jobs: Optional[List[SimfarmJobInfo]] = Field(
        default=None, description="List of jobs"
    )
    count: Optional[int] = Field(default=None, description="Total count of jobs")

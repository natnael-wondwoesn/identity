from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ResearchRequest(BaseModel):
    query: str = Field(..., description="Market research query")
    industry: Optional[str] = Field(None, description="Industry focus")
    timeframe: Optional[str] = Field("current", description="Time period for analysis")
    priority: Optional[str] = Field("normal", description="Task priority")

class ResearchResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    created_at: str
    updated_at: str

class AnalysisResult(BaseModel):
    task_id: str
    research_query: str
    industry: Optional[str]
    sources_found: int
    analysis_results: Optional[Dict[str, Any]]
    strategy_results: Optional[Dict[str, Any]]
    final_report: Optional[str]
    compliance_status: Optional[Dict[str, Any]]
    status: str
    created_at: str
    completed_at: str

class ComplianceCheck(BaseModel):
    source_url: str
    compliant: bool
    issues: List[str] = []
    recommendations: List[str] = []

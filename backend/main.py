from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
import uuid
import logging
import sys
import os
from contextlib import asynccontextmanager

# Add parent directory to path to access graph module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import create_workflow, MarketResearchWorkflow
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global workflow instance
workflow_instance: Optional[MarketResearchWorkflow] = None
active_tasks: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global workflow_instance

    # Startup
    try:
        settings = get_settings()
        llm_config = {
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL,
            "temperature": settings.LLM_TEMPERATURE,
        }
        workflow_instance = create_workflow(llm_config)
        logger.info("Market Research Workflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize workflow: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Market Research API")


# Initialize FastAPI app
app = FastAPI(
    title="Market Research Analysis API",
    description="Compliant market research system using LangGraph and AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
    ],  # React and Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
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


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """API health check"""
    return {
        "message": "Market Research Analysis API",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Detailed health check"""
    global workflow_instance

    health_status = {
        "api": "healthy",
        "workflow": "healthy" if workflow_instance else "unhealthy",
        "agents": workflow_instance.get_agent_status() if workflow_instance else {},
        "active_tasks": len(active_tasks),
        "timestamp": datetime.now().isoformat(),
    }

    status_code = 200 if workflow_instance else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.post("/research/start", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new market research task"""
    global workflow_instance, active_tasks

    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Workflow not initialized")

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task tracking
    active_tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "request": request.dict(),
        "progress": {"stage": "initialization", "percentage": 0},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # Start background task
    background_tasks.add_task(
        execute_research_workflow,
        task_id,
        request.query,
        request.industry,
        request.timeframe,
    )

    logger.info(f"Started research task {task_id}: {request.query}")

    return ResearchResponse(
        task_id=task_id,
        status="queued",
        message="Research task queued successfully",
        estimated_completion="5-10 minutes",
    )


@app.get("/research/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a research task"""
    global active_tasks

    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = active_tasks[task_id]

    return TaskStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info.get("progress"),
        results=task_info.get("results"),
        errors=task_info.get("errors"),
        created_at=task_info["created_at"],
        updated_at=task_info["updated_at"],
    )


@app.get("/research/results/{task_id}", response_model=AnalysisResult)
async def get_research_results(task_id: str):
    """Get complete results of a research task"""
    global active_tasks

    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = active_tasks[task_id]

    if task_info["status"] not in ["completed", "completed_with_errors"]:
        raise HTTPException(status_code=202, detail="Task not yet completed")

    results = task_info.get("results", {})

    return AnalysisResult(
        task_id=task_id,
        research_query=results.get("research_query", ""),
        industry=results.get("industry"),
        sources_found=results.get("sources_found", 0),
        analysis_results=results.get("analysis_results"),
        strategy_results=results.get("strategy_results"),
        final_report=results.get("final_report"),
        compliance_status=results.get("compliance_status"),
        status=results.get("status", "unknown"),
        created_at=results.get("created_at", task_info["created_at"]),
        completed_at=results.get("completed_at", task_info["updated_at"]),
    )


@app.get("/research/tasks", response_model=List[TaskStatus])
async def list_tasks(limit: int = 10, status: Optional[str] = None):
    """List recent research tasks"""
    global active_tasks

    tasks = []
    for task_info in list(active_tasks.values())[-limit:]:
        if status and task_info["status"] != status:
            continue

        tasks.append(
            TaskStatus(
                task_id=task_info["task_id"],
                status=task_info["status"],
                progress=task_info.get("progress"),
                results=task_info.get("results"),
                errors=task_info.get("errors"),
                created_at=task_info["created_at"],
                updated_at=task_info["updated_at"],
            )
        )

    return tasks


@app.delete("/research/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a research task"""
    global active_tasks

    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = active_tasks[task_id]

    if task_info["status"] in ["completed", "failed", "cancelled"]:
        return {"message": "Task already finished", "status": task_info["status"]}

    # Mark as cancelled
    active_tasks[task_id]["status"] = "cancelled"
    active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

    return {"message": "Task cancelled successfully", "task_id": task_id}


@app.get("/workflow/info")
async def get_workflow_info():
    """Get workflow information and visualization"""
    global workflow_instance

    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Workflow not initialized")

    return {
        "visualization": workflow_instance.get_workflow_visualization(),
        "agents": workflow_instance.get_agent_status(),
        "configuration": {
            "llm_provider": workflow_instance.llm_config.get("provider", "openai"),
            "llm_model": workflow_instance.llm_config.get("model", "gpt-3.5-turbo"),
            "temperature": workflow_instance.llm_config.get("temperature", 0.1),
        },
    }


# Background task execution
async def execute_research_workflow(
    task_id: str, query: str, industry: str, timeframe: str
):
    """Execute research workflow in background"""
    global workflow_instance, active_tasks

    try:
        # Update status to running
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["progress"] = {"stage": "research", "percentage": 20}
        active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        # Execute workflow
        results = await workflow_instance.run_research(query, industry, timeframe)

        # Update with results
        active_tasks[task_id]["status"] = results["status"]
        active_tasks[task_id]["results"] = results
        active_tasks[task_id]["progress"] = {"stage": "completed", "percentage": 100}
        active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        if results.get("errors"):
            active_tasks[task_id]["errors"] = results["errors"]

        logger.info(f"Completed research task {task_id}")

    except Exception as e:
        # Update with error
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["errors"] = [str(e)]
        active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        logger.error(f"Research task {task_id} failed: {str(e)}")


# Cleanup old tasks periodically
@app.on_event("startup")
async def setup_periodic_tasks():
    """Setup periodic cleanup tasks"""
    asyncio.create_task(cleanup_old_tasks())


async def cleanup_old_tasks():
    """Clean up old completed tasks"""
    global active_tasks

    while True:
        try:
            # Sleep for 1 hour
            await asyncio.sleep(3600)

            # Remove tasks older than 24 hours
            cutoff_time = datetime.now().timestamp() - (24 * 3600)

            tasks_to_remove = []
            for task_id, task_info in active_tasks.items():
                task_time = datetime.fromisoformat(task_info["created_at"]).timestamp()
                if task_time < cutoff_time and task_info["status"] in [
                    "completed",
                    "failed",
                    "cancelled",
                ]:
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del active_tasks[task_id]

            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

        except Exception as e:
            logger.error(f"Task cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

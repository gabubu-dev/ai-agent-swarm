"""FastAPI server for AI Agent Swarm Manager."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agents.agent import AgentConfig
from ..aggregator.aggregator import AggregationStrategy
from ..swarm.manager import SwarmManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global swarm manager instance
swarm_manager: Optional[SwarmManager] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    
    logger.warning("config.json not found, using defaults")
    return {
        "redis": {"host": "localhost", "port": 6379},
        "providers": {},
        "swarm": {"max_agents": 10},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global swarm_manager
    
    # Startup
    logger.info("Starting AI Agent Swarm API...")
    config = load_config()
    
    swarm_manager = SwarmManager(
        redis_config=config.get("redis", {}),
        provider_config=config.get("providers", {}),
        max_swarms=config.get("swarm", {}).get("max_swarms", 10),
    )
    await swarm_manager.start()
    logger.info("Swarm manager started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Agent Swarm API...")
    if swarm_manager:
        await swarm_manager.stop()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AI Agent Swarm Manager",
    description="API for managing and coordinating multiple AI agents",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response Models

class CreateSwarmRequest(BaseModel):
    """Request to create a new swarm."""
    name: str
    num_agents: int = Field(default=3, ge=1, le=100)
    provider: str = Field(default="anthropic")
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=100000)
    aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE


class CreateSwarmResponse(BaseModel):
    """Response for swarm creation."""
    swarm_id: str
    name: str
    num_agents: int
    status: str


class SubmitTaskRequest(BaseModel):
    """Request to submit a task."""
    swarm_id: str
    task_type: str
    description: str
    subtasks: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=0, ge=0, le=100)


class SubmitTaskResponse(BaseModel):
    """Response for task submission."""
    task_id: str
    swarm_id: str
    status: str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Agent Swarm Manager API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        stats = await swarm_manager.get_stats()
        return {
            "status": "healthy",
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/swarms", response_model=CreateSwarmResponse)
async def create_swarm(request: CreateSwarmRequest):
    """Create a new swarm of agents."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        agent_config = AgentConfig(
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        swarm_id = await swarm_manager.create_swarm(
            name=request.name,
            num_agents=request.num_agents,
            agent_config=agent_config,
            aggregation_strategy=request.aggregation_strategy,
        )
        
        swarm = swarm_manager.get_swarm(swarm_id)
        
        return CreateSwarmResponse(
            swarm_id=swarm_id,
            name=request.name,
            num_agents=request.num_agents,
            status=swarm.status.value if swarm else "unknown",
        )
        
    except Exception as e:
        logger.error(f"Failed to create swarm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarms")
async def list_swarms():
    """List all swarms."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        swarms = swarm_manager.list_swarms()
        return {"swarms": swarms}
    except Exception as e:
        logger.error(f"Failed to list swarms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/swarms/{swarm_id}")
async def get_swarm(swarm_id: str):
    """Get information about a specific swarm."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    swarm = swarm_manager.get_swarm(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail=f"Swarm {swarm_id} not found")
    
    pool = swarm_manager.agent_pools.get(swarm_id)
    pool_stats = pool.get_pool_stats() if pool else {}
    
    return {
        **swarm.model_dump(),
        "pool_stats": pool_stats,
    }


@app.delete("/swarms/{swarm_id}")
async def terminate_swarm(swarm_id: str):
    """Terminate a swarm."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        success = await swarm_manager.terminate_swarm(swarm_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Swarm {swarm_id} not found")
        
        return {"message": f"Swarm {swarm_id} terminated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to terminate swarm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks", response_model=SubmitTaskResponse)
async def submit_task(request: SubmitTaskRequest):
    """Submit a task to a swarm."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        task_id = await swarm_manager.submit_task(
            swarm_id=request.swarm_id,
            task_type=request.task_type,
            description=request.description,
            subtasks=request.subtasks,
            context=request.context,
            priority=request.priority,
        )
        
        return SubmitTaskResponse(
            task_id=task_id,
            swarm_id=request.swarm_id,
            status="pending",
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and result."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        task = await swarm_manager.get_task_result(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return task.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        success = await swarm_manager.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {"message": f"Task {task_id} cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get overall system statistics."""
    if not swarm_manager:
        raise HTTPException(status_code=503, detail="Swarm manager not initialized")
    
    try:
        stats = await swarm_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    api_config = config.get("api", {})
    
    uvicorn.run(
        "src.api.server:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("debug", False),
    )

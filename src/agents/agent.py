"""Core Agent implementation."""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .provider import AIProvider, OpenAIProvider, AnthropicProvider

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent status states."""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentConfig(BaseModel):
    """Configuration for an AI agent."""
    provider: str = "anthropic"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 300
    retry_attempts: int = 3


class AgentMetrics(BaseModel):
    """Metrics for tracking agent performance."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0
    total_processing_time: float = 0.0
    average_response_time: float = 0.0


class Agent:
    """AI Agent that can execute tasks using various AI providers."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        provider_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Agent configuration
            provider_config: Configuration for the AI provider
        """
        self.agent_id = agent_id or f"agent-{uuid4().hex[:8]}"
        self.config = config or AgentConfig()
        self.status = AgentStatus.IDLE
        self.current_task: Optional[str] = None
        self.metrics = AgentMetrics()
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)
        
        # Initialize AI provider
        self.provider: AIProvider = self._init_provider(provider_config or {})
        
        logger.info(f"Agent {self.agent_id} initialized with {self.config.provider} provider")

    def _init_provider(self, provider_config: Dict[str, Any]) -> AIProvider:
        """Initialize the AI provider based on configuration."""
        provider_name = self.config.provider.lower()
        
        if provider_name == "openai":
            return OpenAIProvider(
                api_key=provider_config.get("api_key", ""),
                model=self.config.model or provider_config.get("model", "gpt-4"),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif provider_name == "anthropic":
            return AnthropicProvider(
                api_key=provider_config.get("api_key", ""),
                model=self.config.model or provider_config.get("model", "claude-3-5-sonnet-20241022"),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    async def execute_task(
        self,
        task_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a task using the AI provider.
        
        Args:
            task_id: Unique task identifier
            task_description: Description of the task to execute
            context: Additional context for the task
            
        Returns:
            Dictionary containing task result and metadata
        """
        if self.status == AgentStatus.TERMINATED:
            raise RuntimeError(f"Agent {self.agent_id} is terminated")
        
        if self.status == AgentStatus.PAUSED:
            raise RuntimeError(f"Agent {self.agent_id} is paused")
        
        self.status = AgentStatus.BUSY
        self.current_task = task_id
        self.last_activity = datetime.now(timezone.utc)
        
        start_time = datetime.now(timezone.utc)
        result = {
            "task_id": task_id,
            "agent_id": self.agent_id,
            "status": "success",
            "output": None,
            "error": None,
            "tokens_used": 0,
            "processing_time": 0.0,
        }
        
        try:
            logger.info(f"Agent {self.agent_id} executing task {task_id}")
            
            # Execute with retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    response = await asyncio.wait_for(
                        self.provider.generate(task_description, context),
                        timeout=self.config.timeout,
                    )
                    
                    result["output"] = response["content"]
                    result["tokens_used"] = response.get("tokens_used", 0)
                    break
                    
                except asyncio.TimeoutError:
                    if attempt == self.config.retry_attempts - 1:
                        raise
                    logger.warning(
                        f"Agent {self.agent_id} timeout on attempt {attempt + 1}, retrying..."
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise
                    logger.warning(
                        f"Agent {self.agent_id} error on attempt {attempt + 1}: {e}, retrying..."
                    )
                    await asyncio.sleep(2 ** attempt)
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.total_tokens_used += result["tokens_used"]
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed task {task_id}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            self.metrics.tasks_failed += 1
            self.status = AgentStatus.ERROR
        
        finally:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            result["processing_time"] = processing_time
            
            # Update metrics
            self.metrics.total_processing_time += processing_time
            total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
            if total_tasks > 0:
                self.metrics.average_response_time = (
                    self.metrics.total_processing_time / total_tasks
                )
            
            self.current_task = None
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.IDLE
            self.last_activity = datetime.now(timezone.utc)
        
        return result

    def pause(self) -> None:
        """Pause the agent."""
        if self.status not in [AgentStatus.IDLE, AgentStatus.ERROR]:
            logger.warning(f"Cannot pause agent {self.agent_id} in status {self.status}")
            return
        
        self.status = AgentStatus.PAUSED
        logger.info(f"Agent {self.agent_id} paused")

    def resume(self) -> None:
        """Resume a paused agent."""
        if self.status != AgentStatus.PAUSED:
            logger.warning(f"Cannot resume agent {self.agent_id} not in paused state")
            return
        
        self.status = AgentStatus.IDLE
        logger.info(f"Agent {self.agent_id} resumed")

    def terminate(self) -> None:
        """Terminate the agent."""
        self.status = AgentStatus.TERMINATED
        self.current_task = None
        logger.info(f"Agent {self.agent_id} terminated")

    def get_info(self) -> Dict[str, Any]:
        """Get agent information and metrics."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "current_task": self.current_task,
            "provider": self.config.provider,
            "model": self.provider.model,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "total_tokens_used": self.metrics.total_tokens_used,
                "average_response_time": self.metrics.average_response_time,
            },
        }

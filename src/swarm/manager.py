"""Swarm manager for orchestrating multiple agents."""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from ..agents.agent import AgentConfig
from ..aggregator.aggregator import ResultAggregator, AggregationStrategy
from ..queue.redis_queue import RedisQueue, Task, TaskStatus
from .pool import AgentPool

logger = logging.getLogger(__name__)


class SwarmStatus(str, Enum):
    """Swarm status states."""
    ACTIVE = "active"
    IDLE = "idle"
    PAUSED = "paused"
    TERMINATED = "terminated"


class SwarmConfig(BaseModel):
    """Configuration for a swarm."""
    name: str
    num_agents: int = 3
    agent_config: Optional[AgentConfig] = None
    aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE
    auto_scale: bool = False
    min_agents: int = 1
    max_agents: int = 10


class Swarm(BaseModel):
    """Swarm model."""
    swarm_id: str = Field(default_factory=lambda: f"swarm-{uuid4().hex[:8]}")
    name: str
    status: SwarmStatus = SwarmStatus.IDLE
    agent_ids: List[str] = Field(default_factory=list)
    task_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: SwarmConfig


class SwarmManager:
    """Manages multiple swarms of agents."""

    def __init__(
        self,
        redis_config: Optional[Dict[str, Any]] = None,
        provider_config: Optional[Dict[str, Any]] = None,
        max_swarms: int = 10,
    ):
        """Initialize the swarm manager.
        
        Args:
            redis_config: Redis configuration
            provider_config: AI provider configuration
            max_swarms: Maximum number of concurrent swarms
        """
        self.max_swarms = max_swarms
        self.provider_config = provider_config or {}
        self.swarms: Dict[str, Swarm] = {}
        self.agent_pools: Dict[str, AgentPool] = {}
        self.aggregators: Dict[str, ResultAggregator] = {}
        
        # Initialize Redis queue
        redis_config = redis_config or {}
        self.queue = RedisQueue(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
        )
        
        self._worker_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
        logger.info("Swarm manager initialized")

    async def start(self) -> None:
        """Start the swarm manager."""
        await self.queue.connect()
        self._running = True
        logger.info("Swarm manager started")

    async def stop(self) -> None:
        """Stop the swarm manager."""
        self._running = False
        
        # Cancel all worker tasks
        for task in self._worker_tasks.values():
            task.cancel()
        
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks.values(), return_exceptions=True)
        
        # Cleanup agent pools
        for pool in self.agent_pools.values():
            pool.cleanup()
        
        await self.queue.disconnect()
        logger.info("Swarm manager stopped")

    async def create_swarm(
        self,
        name: str,
        num_agents: int = 3,
        agent_config: Optional[AgentConfig] = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.CONCATENATE,
    ) -> str:
        """Create a new swarm.
        
        Args:
            name: Name for the swarm
            num_agents: Number of agents in the swarm
            agent_config: Configuration for agents
            aggregation_strategy: Strategy for aggregating results
            
        Returns:
            Swarm ID
        """
        if len(self.swarms) >= self.max_swarms:
            raise RuntimeError(f"Maximum number of swarms ({self.max_swarms}) reached")
        
        swarm_config = SwarmConfig(
            name=name,
            num_agents=num_agents,
            agent_config=agent_config,
            aggregation_strategy=aggregation_strategy,
        )
        
        swarm = Swarm(
            name=name,
            config=swarm_config,
        )
        
        # Create agent pool for this swarm
        pool = AgentPool(
            max_agents=swarm_config.max_agents,
            agent_config=agent_config or AgentConfig(),
            provider_config=self.provider_config,
        )
        
        # Create agents
        for _ in range(num_agents):
            agent = await pool.create_agent()
            swarm.agent_ids.append(agent.agent_id)
        
        # Create result aggregator
        aggregator = ResultAggregator(strategy=aggregation_strategy)
        
        # Store swarm components
        self.swarms[swarm.swarm_id] = swarm
        self.agent_pools[swarm.swarm_id] = pool
        self.aggregators[swarm.swarm_id] = aggregator
        
        # Start worker for this swarm
        worker_task = asyncio.create_task(self._swarm_worker(swarm.swarm_id))
        self._worker_tasks[swarm.swarm_id] = worker_task
        
        logger.info(f"Created swarm {swarm.swarm_id} ({name}) with {num_agents} agents")
        return swarm.swarm_id

    async def submit_task(
        self,
        swarm_id: str,
        task_type: str,
        description: str,
        subtasks: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """Submit a task to a swarm.
        
        Args:
            swarm_id: ID of the swarm
            task_type: Type of task
            description: Task description
            subtasks: Optional list of subtasks
            context: Optional context for the task
            priority: Task priority (higher = more urgent)
            
        Returns:
            Task ID
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm:
            raise ValueError(f"Swarm {swarm_id} not found")
        
        task = Task(
            swarm_id=swarm_id,
            task_type=task_type,
            description=description,
            subtasks=subtasks or [],
            context=context or {},
            priority=priority,
        )
        
        task_id = await self.queue.enqueue_task(task)
        swarm.task_ids.append(task_id)
        swarm.status = SwarmStatus.ACTIVE
        
        logger.info(f"Submitted task {task_id} to swarm {swarm_id}")
        return task_id

    async def _swarm_worker(self, swarm_id: str) -> None:
        """Worker loop for processing tasks in a swarm."""
        logger.info(f"Starting worker for swarm {swarm_id}")
        
        while self._running:
            try:
                swarm = self.swarms.get(swarm_id)
                if not swarm:
                    break
                
                pool = self.agent_pools.get(swarm_id)
                if not pool:
                    break
                
                # Get pending tasks for this swarm
                tasks = await self.queue.get_swarm_tasks(swarm_id)
                pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
                
                if not pending_tasks:
                    swarm.status = SwarmStatus.IDLE
                    await asyncio.sleep(1)
                    continue
                
                swarm.status = SwarmStatus.ACTIVE
                
                # Process tasks with available agents
                for task in pending_tasks:
                    # Check if we have subtasks to distribute
                    if task.subtasks:
                        await self._process_distributed_task(swarm_id, task)
                    else:
                        await self._process_single_task(swarm_id, task)
                
            except asyncio.CancelledError:
                logger.info(f"Worker for swarm {swarm_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in swarm {swarm_id} worker: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_single_task(self, swarm_id: str, task: Task) -> None:
        """Process a single task with one agent."""
        pool = self.agent_pools[swarm_id]
        agent = pool.get_idle_agent()
        
        if not agent:
            return  # No agents available, try again later
        
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = agent.agent_id
        task.started_at = datetime.now(timezone.utc)
        await self.queue.update_task(task)
        
        try:
            result = await agent.execute_task(
                task_id=task.task_id,
                task_description=task.description,
                context=task.context,
            )
            
            await self.queue.complete_task(
                task_id=task.task_id,
                result=result,
                success=(result["status"] == "success"),
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            await self.queue.complete_task(
                task_id=task.task_id,
                result={"error": str(e)},
                success=False,
            )

    async def _process_distributed_task(self, swarm_id: str, task: Task) -> None:
        """Process a task distributed across multiple agents."""
        pool = self.agent_pools[swarm_id]
        aggregator = self.aggregators[swarm_id]
        
        # Get available agents
        num_subtasks = len(task.subtasks)
        idle_agents = pool.get_idle_agents(num_subtasks)
        
        if not idle_agents:
            return  # No agents available
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)
        await self.queue.update_task(task)
        
        # Distribute subtasks to agents
        agent_tasks = []
        for agent, subtask in zip(idle_agents, task.subtasks):
            agent_task = agent.execute_task(
                task_id=f"{task.task_id}-{agent.agent_id}",
                task_description=f"{task.description}\n\nSubtask: {subtask}",
                context=task.context,
            )
            agent_tasks.append(agent_task)
        
        # Wait for all agents to complete
        try:
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            # Aggregate results
            aggregated = await aggregator.aggregate(
                results=valid_results,
                task_description=task.description,
            )
            
            await self.queue.complete_task(
                task_id=task.task_id,
                result=aggregated,
                success=bool(valid_results),
            )
            
        except Exception as e:
            logger.error(f"Distributed task {task.task_id} failed: {e}")
            await self.queue.complete_task(
                task_id=task.task_id,
                result={"error": str(e)},
                success=False,
            )

    async def get_task_result(self, task_id: str) -> Optional[Task]:
        """Get the result of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task with result or None if not found
        """
        return await self.queue.get_task(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False if not found
        """
        return await self.queue.cancel_task(task_id)

    def get_swarm(self, swarm_id: str) -> Optional[Swarm]:
        """Get swarm information.
        
        Args:
            swarm_id: Swarm ID
            
        Returns:
            Swarm or None if not found
        """
        return self.swarms.get(swarm_id)

    def list_swarms(self) -> List[Dict[str, Any]]:
        """List all swarms.
        
        Returns:
            List of swarm information
        """
        return [
            {
                **swarm.model_dump(),
                "pool_stats": self.agent_pools[swarm.swarm_id].get_pool_stats(),
            }
            for swarm in self.swarms.values()
        ]

    async def terminate_swarm(self, swarm_id: str) -> bool:
        """Terminate a swarm.
        
        Args:
            swarm_id: Swarm ID
            
        Returns:
            True if terminated, False if not found
        """
        swarm = self.swarms.get(swarm_id)
        if not swarm:
            return False
        
        # Cancel worker task
        worker_task = self._worker_tasks.get(swarm_id)
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            del self._worker_tasks[swarm_id]
        
        # Cleanup pool
        pool = self.agent_pools.get(swarm_id)
        if pool:
            pool.cleanup()
            del self.agent_pools[swarm_id]
        
        # Remove aggregator
        if swarm_id in self.aggregators:
            del self.aggregators[swarm_id]
        
        # Update swarm status
        swarm.status = SwarmStatus.TERMINATED
        
        logger.info(f"Terminated swarm {swarm_id}")
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics.
        
        Returns:
            Statistics dictionary
        """
        queue_stats = await self.queue.get_queue_stats()
        
        total_agents = sum(
            pool.get_pool_stats()["total_agents"]
            for pool in self.agent_pools.values()
        )
        
        swarm_statuses = {}
        for status in SwarmStatus:
            swarm_statuses[status.value] = sum(
                1 for s in self.swarms.values() if s.status == status
            )
        
        return {
            "total_swarms": len(self.swarms),
            "swarm_statuses": swarm_statuses,
            "total_agents": total_agents,
            **queue_stats,
        }

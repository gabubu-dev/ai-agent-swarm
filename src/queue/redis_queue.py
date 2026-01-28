"""Redis-backed task queue for distributed task management."""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import redis.asyncio as aioredis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status states."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Task model."""
    task_id: str = Field(default_factory=lambda: f"task-{uuid4().hex[:12]}")
    swarm_id: str
    task_type: str
    description: str
    subtasks: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RedisQueue:
    """Redis-backed task queue."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        """Initialize Redis queue.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis: Optional[aioredis.Redis] = None
        
        # Key prefixes
        self.task_key_prefix = "task:"
        self.queue_key = "task_queue"
        self.active_tasks_key = "active_tasks"
        self.swarm_tasks_prefix = "swarm_tasks:"

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password,
                encoding="utf-8",
                decode_responses=True,
            )
            await self.redis.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    async def enqueue_task(self, task: Task) -> str:
        """Add a task to the queue.
        
        Args:
            task: Task to enqueue
            
        Returns:
            Task ID
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        task_key = f"{self.task_key_prefix}{task.task_id}"
        task_data = task.model_dump_json()
        
        # Store task data
        await self.redis.set(task_key, task_data)
        
        # Add to priority queue (using sorted set with priority as score)
        await self.redis.zadd(
            self.queue_key,
            {task.task_id: -task.priority}  # Negative for high priority first
        )
        
        # Track task for swarm
        swarm_tasks_key = f"{self.swarm_tasks_prefix}{task.swarm_id}"
        await self.redis.sadd(swarm_tasks_key, task.task_id)
        
        logger.info(f"Enqueued task {task.task_id} for swarm {task.swarm_id}")
        return task.task_id

    async def dequeue_task(self) -> Optional[Task]:
        """Get the next task from the queue.
        
        Returns:
            Next task or None if queue is empty
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Get highest priority task (lowest score)
        result = await self.redis.zpopmin(self.queue_key, count=1)
        
        if not result:
            return None
        
        task_id, _ = result[0]
        task_key = f"{self.task_key_prefix}{task_id}"
        
        # Get task data
        task_data = await self.redis.get(task_key)
        if not task_data:
            logger.warning(f"Task {task_id} not found in Redis")
            return None
        
        task = Task.model_validate_json(task_data)
        
        # Move to active tasks
        await self.redis.sadd(self.active_tasks_key, task_id)
        
        logger.info(f"Dequeued task {task_id}")
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        task_key = f"{self.task_key_prefix}{task_id}"
        task_data = await self.redis.get(task_key)
        
        if not task_data:
            return None
        
        return Task.model_validate_json(task_data)

    async def update_task(self, task: Task) -> None:
        """Update a task.
        
        Args:
            task: Updated task
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        task_key = f"{self.task_key_prefix}{task.task_id}"
        task_data = task.model_dump_json()
        await self.redis.set(task_key, task_data)
        
        logger.debug(f"Updated task {task.task_id}")

    async def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
        success: bool = True,
    ) -> None:
        """Mark a task as completed.
        
        Args:
            task_id: Task ID
            result: Task result
            success: Whether task succeeded
        """
        task = await self.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found for completion")
            return
        
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.result = result
        task.completed_at = datetime.utcnow()
        
        await self.update_task(task)
        
        # Remove from active tasks
        if self.redis:
            await self.redis.srem(self.active_tasks_key, task_id)
        
        logger.info(f"Task {task_id} marked as {task.status.value}")

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False if not found
        """
        task = await self.get_task(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.CANCELLED
        await self.update_task(task)
        
        # Remove from queue if pending
        if self.redis:
            await self.redis.zrem(self.queue_key, task_id)
            await self.redis.srem(self.active_tasks_key, task_id)
        
        logger.info(f"Task {task_id} cancelled")
        return True

    async def get_swarm_tasks(self, swarm_id: str) -> List[Task]:
        """Get all tasks for a swarm.
        
        Args:
            swarm_id: Swarm ID
            
        Returns:
            List of tasks
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        swarm_tasks_key = f"{self.swarm_tasks_prefix}{swarm_id}"
        task_ids = await self.redis.smembers(swarm_tasks_key)
        
        tasks = []
        for task_id in task_ids:
            task = await self.get_task(task_id)
            if task:
                tasks.append(task)
        
        return tasks

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Queue statistics
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        pending_count = await self.redis.zcard(self.queue_key)
        active_count = await self.redis.scard(self.active_tasks_key)
        
        return {
            "pending_tasks": pending_count,
            "active_tasks": active_count,
            "total_tasks": pending_count + active_count,
        }

    async def clear_all(self) -> None:
        """Clear all tasks from the queue (use with caution!)."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        # Get all task IDs
        all_task_ids = await self.redis.zrange(self.queue_key, 0, -1)
        all_task_ids.extend(await self.redis.smembers(self.active_tasks_key))
        
        # Delete all task keys
        if all_task_ids:
            task_keys = [f"{self.task_key_prefix}{tid}" for tid in all_task_ids]
            await self.redis.delete(*task_keys)
        
        # Clear queue and active tasks
        await self.redis.delete(self.queue_key)
        await self.redis.delete(self.active_tasks_key)
        
        # Clear swarm task sets
        swarm_keys = await self.redis.keys(f"{self.swarm_tasks_prefix}*")
        if swarm_keys:
            await self.redis.delete(*swarm_keys)
        
        logger.info("Cleared all tasks from queue")

"""Tests for RedisQueue."""

import pytest
from datetime import datetime

from src.queue.redis_queue import Task, TaskStatus


@pytest.mark.asyncio
async def test_queue_enqueue_task(redis_queue):
    """Test enqueueing a task."""
    task = Task(
        swarm_id="swarm-1",
        task_type="analysis",
        description="Test task",
        priority=5,
    )
    
    task_id = await redis_queue.enqueue_task(task)
    
    assert task_id == task.task_id
    redis_queue.redis.set.assert_called()
    redis_queue.redis.zadd.assert_called()


@pytest.mark.asyncio
async def test_queue_get_task(redis_queue):
    """Test getting a task."""
    task = Task(
        task_id="task-1",
        swarm_id="swarm-1",
        task_type="analysis",
        description="Test task",
    )
    
    # Mock Redis get to return task data
    redis_queue.redis.get.return_value = task.model_dump_json()
    
    retrieved = await redis_queue.get_task("task-1")
    
    assert retrieved is not None
    assert retrieved.task_id == "task-1"


@pytest.mark.asyncio
async def test_queue_update_task(redis_queue):
    """Test updating a task."""
    task = Task(
        task_id="task-1",
        swarm_id="swarm-1",
        task_type="analysis",
        description="Test task",
    )
    
    task.status = TaskStatus.COMPLETED
    await redis_queue.update_task(task)
    
    redis_queue.redis.set.assert_called()


@pytest.mark.asyncio
async def test_queue_complete_task(redis_queue):
    """Test completing a task."""
    task = Task(
        task_id="task-1",
        swarm_id="swarm-1",
        task_type="analysis",
        description="Test task",
    )
    
    redis_queue.redis.get.return_value = task.model_dump_json()
    
    result = {"output": "Task result"}
    await redis_queue.complete_task("task-1", result, success=True)
    
    redis_queue.redis.set.assert_called()
    redis_queue.redis.srem.assert_called()


@pytest.mark.asyncio
async def test_queue_cancel_task(redis_queue):
    """Test cancelling a task."""
    task = Task(
        task_id="task-1",
        swarm_id="swarm-1",
        task_type="analysis",
        description="Test task",
    )
    
    redis_queue.redis.get.return_value = task.model_dump_json()
    
    success = await redis_queue.cancel_task("task-1")
    
    assert success is True
    redis_queue.redis.zrem.assert_called()


@pytest.mark.asyncio
async def test_queue_get_stats(redis_queue):
    """Test getting queue statistics."""
    redis_queue.redis.zcard.return_value = 5
    redis_queue.redis.scard.return_value = 3
    
    stats = await redis_queue.get_queue_stats()
    
    assert stats["pending_tasks"] == 5
    assert stats["active_tasks"] == 3
    assert stats["total_tasks"] == 8

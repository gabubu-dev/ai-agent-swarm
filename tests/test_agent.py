"""Tests for Agent implementation."""

import pytest
from unittest.mock import patch, AsyncMock

from src.agents.agent import Agent, AgentConfig, AgentStatus


@pytest.mark.asyncio
async def test_agent_initialization(mock_agent):
    """Test agent initialization."""
    assert mock_agent.agent_id == "test-agent"
    assert mock_agent.status == AgentStatus.IDLE
    assert mock_agent.current_task is None
    assert mock_agent.metrics.tasks_completed == 0


@pytest.mark.asyncio
async def test_agent_execute_task(mock_agent):
    """Test agent executing a task."""
    result = await mock_agent.execute_task(
        task_id="task-1",
        task_description="Test task",
        context={"key": "value"},
    )
    
    assert result["task_id"] == "task-1"
    assert result["agent_id"] == "test-agent"
    assert result["status"] == "success"
    assert result["output"] is not None
    assert result["tokens_used"] > 0
    assert mock_agent.status == AgentStatus.IDLE
    assert mock_agent.metrics.tasks_completed == 1


@pytest.mark.asyncio
async def test_agent_pause_resume(mock_agent):
    """Test pausing and resuming agent."""
    mock_agent.pause()
    assert mock_agent.status == AgentStatus.PAUSED
    
    mock_agent.resume()
    assert mock_agent.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_agent_terminate(mock_agent):
    """Test terminating agent."""
    mock_agent.terminate()
    assert mock_agent.status == AgentStatus.TERMINATED
    
    with pytest.raises(RuntimeError):
        await mock_agent.execute_task("task-1", "Test task")


@pytest.mark.asyncio
async def test_agent_get_info(mock_agent):
    """Test getting agent information."""
    info = mock_agent.get_info()
    
    assert info["agent_id"] == "test-agent"
    assert info["status"] == AgentStatus.IDLE.value
    assert "metrics" in info
    assert "created_at" in info


@pytest.mark.asyncio
async def test_agent_retry_on_failure(mock_agent):
    """Test agent retry logic on failure."""
    mock_agent.provider.generate = AsyncMock(
        side_effect=[Exception("Error 1"), Exception("Error 2"), {"content": "Success", "tokens_used": 50}]
    )
    
    result = await mock_agent.execute_task("task-1", "Test task")
    
    assert result["status"] == "success"
    assert mock_agent.provider.generate.call_count == 3


@pytest.mark.asyncio
async def test_agent_task_failure(mock_agent):
    """Test agent handling task failure."""
    mock_agent.provider.generate = AsyncMock(side_effect=Exception("Test error"))
    
    result = await mock_agent.execute_task("task-1", "Test task")
    
    assert result["status"] == "failed"
    assert "error" in result
    assert mock_agent.status == AgentStatus.ERROR
    assert mock_agent.metrics.tasks_failed == 1

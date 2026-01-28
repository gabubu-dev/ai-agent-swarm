"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.agent import Agent, AgentConfig
from src.agents.provider import AIProvider
from src.queue.redis_queue import RedisQueue
from src.swarm.manager import SwarmManager
from src.swarm.pool import AgentPool


# Set test environment variables
os.environ["TESTING"] = "true"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class MockAIProvider(AIProvider):
    """Mock AI provider for testing."""
    
    async def generate(self, prompt: str, context=None):
        """Mock generate method."""
        return {
            "content": f"Mock response to: {prompt[:50]}...",
            "tokens_used": 100,
        }


@pytest.fixture
def mock_provider():
    """Mock AI provider."""
    return MockAIProvider(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        max_tokens=2000,
    )


@pytest.fixture
def agent_config():
    """Default agent configuration."""
    return AgentConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=2000,
    )


@pytest_asyncio.fixture
async def mock_agent(agent_config, mock_provider):
    """Create a mock agent for testing."""
    with patch("src.agents.agent.Agent._init_provider", return_value=mock_provider):
        agent = Agent(agent_id="test-agent", config=agent_config)
        yield agent


@pytest_asyncio.fixture
async def agent_pool(agent_config):
    """Create an agent pool for testing."""
    with patch("src.agents.agent.Agent._init_provider") as mock_init:
        mock_init.return_value = MockAIProvider(
            api_key="test", model="test", temperature=0.7, max_tokens=2000
        )
        pool = AgentPool(max_agents=5, agent_config=agent_config)
        yield pool
        pool.cleanup()


@pytest_asyncio.fixture
async def redis_queue():
    """Create a Redis queue for testing (mock)."""
    queue = RedisQueue(host="localhost", port=6379, db=0)
    
    # Mock Redis connection
    queue.redis = AsyncMock()
    queue.redis.ping = AsyncMock()
    queue.redis.set = AsyncMock()
    queue.redis.get = AsyncMock()
    queue.redis.zadd = AsyncMock()
    queue.redis.zpopmin = AsyncMock(return_value=[])
    queue.redis.sadd = AsyncMock()
    queue.redis.srem = AsyncMock()
    queue.redis.smembers = AsyncMock(return_value=set())
    queue.redis.zcard = AsyncMock(return_value=0)
    queue.redis.scard = AsyncMock(return_value=0)
    
    yield queue


@pytest_asyncio.fixture
async def swarm_manager(redis_queue):
    """Create a swarm manager for testing."""
    with patch("src.swarm.manager.RedisQueue", return_value=redis_queue):
        with patch("src.agents.agent.Agent._init_provider") as mock_init:
            mock_init.return_value = MockAIProvider(
                api_key="test", model="test", temperature=0.7, max_tokens=2000
            )
            
            manager = SwarmManager(
                redis_config={"host": "localhost", "port": 6379},
                provider_config={},
                max_swarms=10,
            )
            await manager.start()
            yield manager
            await manager.stop()

"""Tests for AgentPool."""

import pytest

from src.agents.agent import AgentStatus


@pytest.mark.asyncio
async def test_pool_create_agent(agent_pool):
    """Test creating agents in the pool."""
    agent = await agent_pool.create_agent()
    
    assert agent.agent_id in agent_pool.agents
    assert agent.status == AgentStatus.IDLE


@pytest.mark.asyncio
async def test_pool_max_agents(agent_pool):
    """Test max agents limit."""
    # Create agents up to max
    for i in range(agent_pool.max_agents):
        await agent_pool.create_agent()
    
    # Should fail when creating beyond max
    with pytest.raises(RuntimeError):
        await agent_pool.create_agent()


@pytest.mark.asyncio
async def test_pool_get_agent(agent_pool):
    """Test getting an agent by ID."""
    agent = await agent_pool.create_agent(agent_id="test-agent-1")
    
    retrieved = agent_pool.get_agent("test-agent-1")
    assert retrieved == agent
    
    not_found = agent_pool.get_agent("nonexistent")
    assert not_found is None


@pytest.mark.asyncio
async def test_pool_get_idle_agent(agent_pool):
    """Test getting an idle agent."""
    agent1 = await agent_pool.create_agent()
    agent2 = await agent_pool.create_agent()
    
    # Both should be idle
    idle = agent_pool.get_idle_agent()
    assert idle in [agent1, agent2]
    
    # Mark one as busy
    agent1.status = AgentStatus.BUSY
    idle = agent_pool.get_idle_agent()
    assert idle == agent2


@pytest.mark.asyncio
async def test_pool_get_idle_agents(agent_pool):
    """Test getting multiple idle agents."""
    await agent_pool.create_agent()
    await agent_pool.create_agent()
    await agent_pool.create_agent()
    
    idle_agents = agent_pool.get_idle_agents(2)
    assert len(idle_agents) == 2
    
    all_idle = agent_pool.get_idle_agents()
    assert len(all_idle) == 3


@pytest.mark.asyncio
async def test_pool_remove_agent(agent_pool):
    """Test removing an agent."""
    agent = await agent_pool.create_agent(agent_id="test-remove")
    
    success = agent_pool.remove_agent("test-remove")
    assert success is True
    assert "test-remove" not in agent_pool.agents
    assert agent.status == AgentStatus.TERMINATED
    
    # Removing non-existent agent
    success = agent_pool.remove_agent("nonexistent")
    assert success is False


@pytest.mark.asyncio
async def test_pool_get_stats(agent_pool):
    """Test getting pool statistics."""
    await agent_pool.create_agent()
    await agent_pool.create_agent()
    
    stats = agent_pool.get_pool_stats()
    
    assert stats["total_agents"] == 2
    assert stats["max_agents"] == agent_pool.max_agents
    assert "status_counts" in stats
    assert stats["status_counts"]["idle"] == 2


@pytest.mark.asyncio
async def test_pool_cleanup(agent_pool):
    """Test cleaning up the pool."""
    await agent_pool.create_agent()
    await agent_pool.create_agent()
    
    agent_pool.cleanup()
    
    assert len(agent_pool.agents) == 0

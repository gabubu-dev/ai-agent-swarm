"""Agent pool for managing multiple agents."""

import logging
from typing import Dict, List, Optional
from uuid import uuid4

from ..agents.agent import Agent, AgentConfig, AgentStatus

logger = logging.getLogger(__name__)


class AgentPool:
    """Pool of agents for task execution."""

    def __init__(
        self,
        max_agents: int = 10,
        agent_config: Optional[AgentConfig] = None,
        provider_config: Optional[Dict] = None,
    ):
        """Initialize the agent pool.
        
        Args:
            max_agents: Maximum number of agents in the pool
            agent_config: Default configuration for new agents
            provider_config: Configuration for AI provider
        """
        self.max_agents = max_agents
        self.agent_config = agent_config or AgentConfig()
        self.provider_config = provider_config or {}
        self.agents: Dict[str, Agent] = {}
        
        logger.info(f"Agent pool initialized with max {max_agents} agents")

    async def create_agent(
        self,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
    ) -> Agent:
        """Create a new agent in the pool.
        
        Args:
            agent_id: Optional custom agent ID
            config: Optional agent configuration
            
        Returns:
            Created agent instance
        """
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Agent pool is full (max: {self.max_agents})")
        
        agent = Agent(
            agent_id=agent_id,
            config=config or self.agent_config,
            provider_config=self.provider_config,
        )
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Created agent {agent.agent_id} in pool")
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_idle_agent(self) -> Optional[Agent]:
        """Get an idle agent from the pool."""
        for agent in self.agents.values():
            if agent.status == AgentStatus.IDLE:
                return agent
        return None

    def get_idle_agents(self, count: int = -1) -> List[Agent]:
        """Get multiple idle agents.
        
        Args:
            count: Number of agents to get (-1 for all idle agents)
            
        Returns:
            List of idle agents
        """
        idle_agents = [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE
        ]
        
        if count == -1:
            return idle_agents
        return idle_agents[:count]

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the pool.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        agent = self.agents.get(agent_id)
        if agent:
            agent.terminate()
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id} from pool")
            return True
        return False

    def get_pool_stats(self) -> Dict:
        """Get statistics about the pool."""
        status_counts = {status.value: 0 for status in AgentStatus}
        total_tasks_completed = 0
        total_tasks_failed = 0
        total_tokens_used = 0
        
        for agent in self.agents.values():
            status_counts[agent.status.value] += 1
            total_tasks_completed += agent.metrics.tasks_completed
            total_tasks_failed += agent.metrics.tasks_failed
            total_tokens_used += agent.metrics.total_tokens_used
        
        return {
            "total_agents": len(self.agents),
            "max_agents": self.max_agents,
            "status_counts": status_counts,
            "total_tasks_completed": total_tasks_completed,
            "total_tasks_failed": total_tasks_failed,
            "total_tokens_used": total_tokens_used,
        }

    def cleanup(self) -> None:
        """Terminate all agents and clear the pool."""
        for agent in self.agents.values():
            agent.terminate()
        self.agents.clear()
        logger.info("Agent pool cleaned up")

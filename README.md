# AI Agent Swarm Manager

A system for managing and coordinating multiple AI agents working together on complex tasks. Distribute work across multiple AI agents, aggregate results, and manage agent lifecycles efficiently.

## Features

- **Task Distribution**: Intelligently split complex tasks across multiple AI agents
- **Result Aggregation**: Combine and synthesize results from multiple agents
- **Agent Lifecycle Management**: Create, monitor, pause, resume, and terminate agents
- **Multi-Provider Support**: Works with OpenAI and Anthropic Claude APIs
- **Redis-Backed Queue**: Reliable task queue and state management
- **REST API**: FastAPI-based API for managing swarms and tasks
- **Async Operations**: Built on asyncio for high concurrency
- **Real-time Monitoring**: Track agent status, task progress, and results

## Architecture

The system consists of several core components:

- **Swarm Manager**: Orchestrates multiple agents and distributes tasks
- **Agent Pool**: Manages a pool of AI agents with different capabilities
- **Task Queue**: Redis-backed queue for task distribution
- **Result Aggregator**: Combines results from multiple agents
- **API Server**: FastAPI REST API for external interaction

## Installation

```bash
# Clone the repository
git clone https://github.com/gabubu-dev/ai-agent-swarm.git
cd ai-agent-swarm

# Install dependencies
pip install -r requirements.txt

# Set up Redis
docker run -d -p 6379:6379 redis:latest
# Or install Redis locally based on your OS

# Configure environment
cp config.example.json config.json
# Edit config.json with your API keys and settings
```

## Configuration

Create a `config.json` file with your settings:

```json
{
  "redis": {
    "host": "localhost",
    "port": 6379
  },
  "providers": {
    "openai": {
      "api_key": "your-openai-key",
      "model": "gpt-4"
    },
    "anthropic": {
      "api_key": "your-anthropic-key",
      "model": "claude-3-5-sonnet-20241022"
    }
  },
  "swarm": {
    "max_agents": 10,
    "default_provider": "anthropic"
  }
}
```

## Usage

### Start the API Server

```bash
python -m src.api.server
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Basic Example

```python
from src.swarm.manager import SwarmManager
from src.agents.agent import AgentConfig

# Initialize the swarm manager
manager = SwarmManager()

# Create a swarm with 3 agents
swarm_id = await manager.create_swarm(
    name="research-swarm",
    num_agents=3,
    agent_config=AgentConfig(provider="anthropic")
)

# Submit a task to the swarm
task_id = await manager.submit_task(
    swarm_id=swarm_id,
    task_type="research",
    description="Research the top 10 AI trends in 2024",
    subtasks=[
        "Find recent AI research papers",
        "Analyze industry reports",
        "Summarize key trends"
    ]
)

# Get results
result = await manager.get_task_result(task_id)
print(result.aggregated_output)
```

### API Examples

Create a swarm:
```bash
curl -X POST http://localhost:8000/swarms \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analysis-swarm",
    "num_agents": 5,
    "provider": "anthropic"
  }'
```

Submit a task:
```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "swarm_id": "swarm-123",
    "task_type": "analysis",
    "description": "Analyze market trends",
    "subtasks": ["Gather data", "Process data", "Generate report"]
  }'
```

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
ai-agent-swarm/
├── src/
│   ├── agents/           # Agent implementations
│   ├── swarm/            # Swarm management
│   ├── queue/            # Task queue (Redis)
│   ├── aggregator/       # Result aggregation
│   └── api/              # FastAPI server
├── tests/                # Test suite
├── config.example.json   # Configuration template
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

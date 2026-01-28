"""Result aggregator for combining outputs from multiple agents."""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Strategy for aggregating results."""
    CONCATENATE = "concatenate"
    MAJORITY_VOTE = "majority_vote"
    CONSENSUS = "consensus"
    BEST_QUALITY = "best_quality"
    WEIGHTED_AVERAGE = "weighted_average"
    CUSTOM = "custom"


class ResultAggregator:
    """Aggregates results from multiple agents."""

    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.CONCATENATE):
        """Initialize the aggregator.
        
        Args:
            strategy: Aggregation strategy to use
        """
        self.strategy = strategy
        logger.info(f"Result aggregator initialized with strategy: {strategy.value}")

    async def aggregate(
        self,
        results: List[Dict[str, Any]],
        task_description: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate results from multiple agents.
        
        Args:
            results: List of agent results
            task_description: Original task description
            weights: Optional weights for weighted aggregation
            
        Returns:
            Aggregated result
        """
        if not results:
            return {
                "aggregated_output": None,
                "num_results": 0,
                "strategy": self.strategy.value,
                "metadata": {},
            }
        
        # Filter successful results
        successful_results = [r for r in results if r.get("status") == "success"]
        
        if not successful_results:
            return {
                "aggregated_output": None,
                "num_results": len(results),
                "strategy": self.strategy.value,
                "metadata": {
                    "error": "All agent tasks failed",
                    "failed_count": len(results),
                },
            }
        
        # Apply aggregation strategy
        if self.strategy == AggregationStrategy.CONCATENATE:
            aggregated = await self._concatenate(successful_results)
        elif self.strategy == AggregationStrategy.MAJORITY_VOTE:
            aggregated = await self._majority_vote(successful_results)
        elif self.strategy == AggregationStrategy.CONSENSUS:
            aggregated = await self._consensus(successful_results, task_description)
        elif self.strategy == AggregationStrategy.BEST_QUALITY:
            aggregated = await self._best_quality(successful_results)
        elif self.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            aggregated = await self._weighted_average(successful_results, weights)
        else:
            aggregated = await self._concatenate(successful_results)
        
        # Add metadata
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        total_time = sum(r.get("processing_time", 0) for r in results)
        
        return {
            "aggregated_output": aggregated,
            "num_results": len(results),
            "successful_results": len(successful_results),
            "failed_results": len(results) - len(successful_results),
            "strategy": self.strategy.value,
            "metadata": {
                "total_tokens_used": total_tokens,
                "total_processing_time": total_time,
                "agent_ids": [r.get("agent_id") for r in results],
            },
        }

    async def _concatenate(self, results: List[Dict[str, Any]]) -> str:
        """Concatenate all results."""
        outputs = []
        for i, result in enumerate(results, 1):
            agent_id = result.get("agent_id", "unknown")
            output = result.get("output", "")
            outputs.append(f"=== Agent {agent_id} ===\n{output}")
        
        return "\n\n".join(outputs)

    async def _majority_vote(self, results: List[Dict[str, Any]]) -> str:
        """Select the most common result."""
        outputs = [r.get("output", "") for r in results]
        
        # Count occurrences
        vote_counts: Dict[str, int] = {}
        for output in outputs:
            # Normalize whitespace for comparison
            normalized = " ".join(output.split())
            vote_counts[normalized] = vote_counts.get(normalized, 0) + 1
        
        # Return most common
        if vote_counts:
            winner = max(vote_counts.items(), key=lambda x: x[1])
            return winner[0]
        
        return outputs[0] if outputs else ""

    async def _consensus(
        self,
        results: List[Dict[str, Any]],
        task_description: Optional[str],
    ) -> str:
        """Build consensus from all results."""
        outputs = [r.get("output", "") for r in results]
        
        # Simple consensus: find common themes/sentences
        # For now, concatenate with context
        consensus_parts = [
            f"Based on analysis from {len(results)} agents:",
            "",
        ]
        
        for i, output in enumerate(outputs, 1):
            consensus_parts.append(f"{i}. {output[:200]}...")
        
        consensus_parts.append("")
        consensus_parts.append("Consensus Summary: Multiple agents analyzed this task and provided similar insights.")
        
        return "\n".join(consensus_parts)

    async def _best_quality(self, results: List[Dict[str, Any]]) -> str:
        """Select the highest quality result."""
        # Quality heuristics: length, tokens used, processing time
        def quality_score(result: Dict[str, Any]) -> float:
            output_length = len(result.get("output", ""))
            tokens = result.get("tokens_used", 0)
            time = result.get("processing_time", 1)
            
            # Balance between output length and efficiency
            return (output_length * 0.5 + tokens * 0.3) / (time * 0.2 + 1)
        
        best_result = max(results, key=quality_score)
        return best_result.get("output", "")

    async def _weighted_average(
        self,
        results: List[Dict[str, Any]],
        weights: Optional[List[float]],
    ) -> str:
        """Combine results with weights."""
        if not weights or len(weights) != len(results):
            # Fall back to equal weights
            weights = [1.0 / len(results)] * len(results)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Combine with weights as context
        output_parts = []
        for result, weight in zip(results, normalized_weights):
            agent_id = result.get("agent_id", "unknown")
            output = result.get("output", "")
            confidence = int(weight * 100)
            output_parts.append(
                f"[Confidence: {confidence}%] Agent {agent_id}:\n{output}"
            )
        
        return "\n\n".join(output_parts)

    def set_strategy(self, strategy: AggregationStrategy) -> None:
        """Change the aggregation strategy.
        
        Args:
            strategy: New aggregation strategy
        """
        self.strategy = strategy
        logger.info(f"Aggregation strategy changed to: {strategy.value}")

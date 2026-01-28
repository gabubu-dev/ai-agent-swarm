"""AI Provider implementations for different APIs."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import anthropic
import openai

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """Initialize the provider.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the AI provider.
        
        Args:
            prompt: The input prompt
            context: Additional context for the generation
            
        Returns:
            Dictionary with 'content' and 'tokens_used' keys
        """
        pass


class OpenAIProvider(AIProvider):
    """OpenAI API provider."""

    def __init__(self, *args, **kwargs):
        """Initialize OpenAI provider."""
        super().__init__(*args, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        try:
            messages = []
            
            # Add system message if context provided
            if context and "system" in context:
                messages.append({
                    "role": "system",
                    "content": context["system"]
                })
            
            # Add the main prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Add conversation history if provided
            if context and "history" in context:
                for msg in context["history"]:
                    messages.append(msg)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "content": content,
                "tokens_used": tokens_used,
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(AIProvider):
    """Anthropic Claude API provider."""

    def __init__(self, *args, **kwargs):
        """Initialize Anthropic provider."""
        super().__init__(*args, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate using Anthropic Claude API."""
        try:
            messages = []
            system_prompt = None
            
            # Extract system message if context provided
            if context and "system" in context:
                system_prompt = context["system"]
            
            # Add conversation history if provided
            if context and "history" in context:
                for msg in context["history"]:
                    messages.append(msg)
            
            # Add the main prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = await self.client.messages.create(**kwargs)
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return {
                "content": content,
                "tokens_used": tokens_used,
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

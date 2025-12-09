"""
OpenRouter API Client for easy response generation.

This module provides a simple interface to interact with OpenRouter's API
for generating responses from various LLM models.
"""

import os
import requests
from typing import Optional, Dict, List, Union

import dotenv
dotenv.load_dotenv()

class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API.

    Usage:
        client = OpenRouterClient(api_key="your-api-key")
        response = client.generate("What is the capital of France?")
        print(response)
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-haiku-4.5",
        app_name: Optional[str] = None,
        site_url: Optional[str] = None
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var.
            model: Default model to use for generations. Default is Claude 3.5 Sonnet.
            app_name: Optional application name for attribution.
            site_url: Optional site URL for attribution.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided either as argument or "
                "through OPENROUTER_API_KEY environment variable"
            )

        self.model = model
        self.app_name = app_name
        self.site_url = site_url

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.app_name:
            headers["X-Title"] = self.app_name
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url

        return headers

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the OpenRouter API.

        Args:
            prompt: The user prompt/question.
            model: Model to use (overrides default).
                   Examples: "anthropic/claude-3.5-sonnet", "openai/gpt-4", "meta-llama/llama-3.1-70b-instruct"
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0-2). Higher = more random.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated response text.
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Make a chat completion request to OpenRouter API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello"}]
            model: Model to use (overrides default).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the response (not yet implemented).
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated response text.
        """
        print(type(model))
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )

        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts to generate responses for.
            model: Model to use (overrides default).
            system_prompt: Optional system prompt to set context.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            List of generated response texts.
        """
        responses = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            responses.append(response)
        return responses

    def list_models(self) -> List[Dict]:
        """
        List available models from OpenRouter.

        Returns:
            List of model information dicts.
        """
        response = requests.get(
            f"{self.BASE_URL}/models",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()["data"]


# Convenience function for quick usage
def quick_generate(
    prompt: str,
    model: str = "anthropic/claude-3.5-sonnet",
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick generation without instantiating a client.

    Args:
        prompt: The prompt to generate a response for.
        model: Model to use.
        api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var).
        **kwargs: Additional parameters for generation.

    Returns:
        The generated response text.
    """
    client = OpenRouterClient(api_key=api_key, model=model)
    return client.generate(prompt, **kwargs)


if __name__ == "__main__":
    # Example usage
    client = OpenRouterClient()

    # Simple generation
    response = client.generate("What is the capital of France?")
    print(f"Response: {response}")

    # With system prompt
    response = client.generate(
        prompt="Explain quantum computing",
        system_prompt="You are a helpful physics teacher. Explain concepts simply.",
        temperature=0.7,
        max_tokens=500
    )
    print(f"\nWith system prompt: {response}")

    # Chat with conversation history
    messages = [
        {"role": "user", "content": "Hello! Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help with Python."},
        {"role": "user", "content": "How do I read a CSV file?"}
    ]
    response = client.chat(messages)
    print(f"\nChat response: {response}")

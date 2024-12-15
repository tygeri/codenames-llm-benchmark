# llm_providers.py

from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional
import openai
import google.generativeai as genai
from anthropic import Anthropic

class BaseLLM(ABC):
    def __init__(self, config: Dict):
        self.temperature = config.get('temperature', 0.7)
        self.last_request_time = 0
        self.min_delay = 0.5

    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()

    @abstractmethod
    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        """Generate a response from the model"""
        pass

class OpenAILLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config['api_key'])
        self.model_name = config['model_name']

    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        self._rate_limit()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

class GeminiLLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        genai.configure(api_key=config['api_key'])
        self.client = genai.GenerativeModel(config['model_name'])

    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        self._rate_limit()
        # Convert OpenAI-style messages to Gemini format
        prompt = self._convert_messages(messages)
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text.strip()

    def _convert_messages(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt += f"Instructions: {msg['content']}\n\n"
            elif msg['role'] == 'user':
                prompt += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"Assistant: {msg['content']}\n"
        return prompt.strip()

class ClaudeLLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.client = Anthropic(api_key=config['api_key'])
        self.model_name = config['model_name']

    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        self._rate_limit()
        # Convert OpenAI-style messages to Claude format
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")
        conversation = [m for m in messages if m['role'] != 'system']
        
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{
                'role': 'user' if m['role'] == 'user' else 'assistant',
                'content': m['content']
            } for m in conversation],
            max_tokens=max_tokens,
            temperature=self.temperature
        )
        return response.content[0].text.strip()

# Factory function to create LLM instances
def create_llm(config: Dict) -> BaseLLM:
    llm_type = config['type'].lower()
    if llm_type == 'openai':
        return OpenAILLM(config)
    elif llm_type == 'gemini':
        return GeminiLLM(config)
    elif llm_type == 'claude':
        return ClaudeLLM(config)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
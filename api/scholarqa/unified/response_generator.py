import os
import httpx
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

class ResponseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class ModalResponseGenerator(ResponseGenerator):
    def __init__(self, endpoint: str, api_key: str = None, model: str = "allenai/sqa_basicsftdpo"):
        self.endpoint = f"{endpoint.rstrip('/')}/v1/chat/completions"
        self.api_key = api_key or os.environ.get("MODAL_PLAYGROUND_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("MODAL_PLAYGROUND_API_KEY is required for ModalResponseGenerator")

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096, # High limit for one-shot generation
        }

        with httpx.Client(timeout=600.0) as client:
            response = client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
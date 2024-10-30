# src/llm/client.py

"""
LLM Client module for the little-matrix simulation.

This module defines the LLMClient class, which manages interactions with the Large Language Model (LLM)
via the Hugging Face Inference API. It handles prompt construction, response generation, and parsing
of the LLM's outputs.

Classes:
    LLMClient: Manages interactions with the LLM.
"""

import logging
import re
from typing import Callable, Dict, Any, Optional
from threading import Thread
from huggingface_hub import InferenceApi
from huggingface_hub.utils import build_hf_headers
from ..llm.prompts import PromptManager
from ..utils.config import Config

class LLMClient:
    """
    Manages interactions with the LLM via Hugging Face Inference API.

    Attributes:
        logger (logging.Logger): Logger for the LLMClient.
        client (InferenceApi): Hugging Face Inference API Client.
        model (str): The model identifier from Hugging Face Hub.
        prompt_manager (PromptManager): Manages prompt templates.
        config (LLMConfig): Configuration settings for the LLM.
    """

    def __init__(self, config: Config):
        """
        Initializes the LLMClient.

        Args:
            config (Config): The configuration object loaded from config.yaml.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.llm  # Access LLM-specific configuration
        self.model = self.config.model
        self.api_key = self.config.api_key

        if not self.api_key or not self.model:
            self.logger.error("API key and model must be provided in the configuration.")
            raise ValueError("API key and model must be provided in the configuration.")

        self.client = InferenceApi(
            repo_id=self.model,
            token=self.api_key,
            task='text-generation'
        )
        self.client.api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        
        # Pass config to PromptManager
        self.prompt_manager = PromptManager(config=config)
        self.logger.info(f"LLMClient initialized with model '{self.model}'.")

    def generate_prompted_response(self, prompt_name: str, prompt_kwargs: Dict[str, Any], max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generates a response from the LLM based on a named prompt template.

        Args:
            prompt_name (str): The name of the prompt template to use.
            prompt_kwargs (Dict[str, Any]): Variables to format the prompt template.
            max_tokens (int, optional): The maximum number of tokens in the response.

        Returns:
            Dict[str, Any]: A dictionary containing the processed response and additional details.
        """
        # Get prompt text from PromptManager
        prompt = self.prompt_manager.get_prompt(prompt_name, **prompt_kwargs)
        return self.generate_response(prompt, max_tokens)

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generates a response from the LLM based on the provided prompt.

        Args:
            prompt (str): The prompt string to send to the LLM.
            max_tokens (int, optional): The maximum number of tokens in the response.

        Returns:
            Dict[str, Any]: A dictionary containing the processed response and additional details.
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        self.logger.debug(f"Generating LLM response for prompt: {prompt}")

        try:
            response = self.client(inputs=prompt, params={
                "max_new_tokens": max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": self.config.do_sample
            })
            self.logger.debug(f"Received raw LLM response: {response}")

            generated_text = self._extract_generated_text(response)
            action = self._extract_action(generated_text)
            details = self._extract_details(generated_text)

            return {
                "action": action,
                "details": details,
                "full_response": generated_text
            }
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return {"action": "rest", "details": {}, "full_response": ""}

    def generate_response_async(
        self,
        prompt_name: str,
        prompt_kwargs: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None],
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Generates a response from the LLM asynchronously based on the provided prompt name.

        Args:
            prompt_name (str): The prompt template name.
            prompt_kwargs (Dict[str, Any]): Variables for formatting the prompt.
            callback (Callable[[Dict[str, Any]], None]): The function to call with the response once it's ready.
            max_tokens (int, optional): The maximum number of tokens in the response.

        Returns:
            None
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        def run():
            self.logger.debug(f"Starting async LLM generation for prompt: {prompt_name}")
            response = self.generate_prompted_response(prompt_name, prompt_kwargs, max_tokens)
            callback(response)

        thread = Thread(target=run)
        thread.start()

    def _extract_generated_text(self, response: Any) -> str:
        """
        Extracts the generated text from the LLM response.

        Args:
            response (Any): The raw response from the LLM.

        Returns:
            str: The extracted generated text.
        """
        if isinstance(response, list) and len(response) > 0:
            first_response = response[0]
            if isinstance(first_response, dict) and 'generated_text' in first_response:
                return first_response['generated_text']
        elif isinstance(response, dict):
            if 'generated_text' in response:
                return response['generated_text']
            elif 'text' in response:
                return response['text']

        self.logger.error(f"Unexpected response format: {response}")
        return ""

    def _extract_action(self, text: str) -> str:
        """
        Extracts the action from the generated text.

        Args:
            text (str): The generated text from the LLM.

        Returns:
            str: The extracted action, or 'rest' if no valid action is found.
        """
        action_match = re.search(r"action:\s*'(\w+)'", text)
        if action_match:
            return action_match.group(1)
        else:
            self.logger.debug("No valid action found in the generated text.")
            return 'rest'

    def _extract_details(self, text: str) -> Dict[str, Any]:
        """
        Extracts additional details from the generated text.

        Args:
            text (str): The generated text from the LLM.

        Returns:
            Dict[str, Any]: A dictionary of extracted details.
        """
        details = {}
        # Extract key-value pairs in the format key: 'value'
        matches = re.findall(r"(\w+):\s*'([^']*)'", text)
        for key, value in matches:
            details[key] = value

        self.logger.debug(f"Extracted details: {details}")
        return details

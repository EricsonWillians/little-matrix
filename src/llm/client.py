# llm/client.py

import logging
import re
from typing import Callable, Dict, Any
from huggingface_hub import InferenceApi
from threading import Thread
from huggingface_hub.utils import build_hf_headers
from llm.prompts import PromptManager

class LLMClient:
    """
    Manages interactions with the LLM via Hugging Face Inference API.

    Attributes:
        logger (logging.Logger): Logger for the LLMClient.
        client (InferenceApi): Hugging Face Inference API Client.
        model (str): The model identifier from Hugging Face Hub.
        prompt_manager (PromptManager): Manages prompt templates.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initializes the LLMClient.

        Args:
            api_key (str): The Hugging Face API key for authentication.
            model (str): The identifier of the model to use from Hugging Face Hub.
        """
        self.logger = logging.getLogger(__name__)
        headers = build_hf_headers(token=api_key)  # Manually setting API key
        self.client = InferenceApi(
            repo_id=model,
            token=api_key,
            task='text-generation'  # Explicitly set the task
        )
        # Override the api_url as per forum suggestion
        self.client.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.model = model
        self.prompt_manager = PromptManager()
        self.logger.info(f"LLMClient initialized with model '{self.model}'.")

    def generate_response(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generates a response from the LLM based on the provided prompt.

        Args:
            prompt (str): The prompt string to send to the LLM.
            max_tokens (int, optional): The maximum number of tokens in the response.

        Returns:
            Dict[str, Any]: A dictionary containing the processed response and additional details.
        """
        self.logger.debug(f"Generating LLM response for prompt: {prompt}")

        try:
            response = self.client(inputs=prompt, params={"max_new_tokens": max_tokens})
            self.logger.debug(f"Received raw LLM response: {response}")
            print(f"Received raw LLM response: {response}")

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

    def generate_response_async(self, prompt: str, callback: Callable[[Dict[str, Any]], None], max_tokens: int = 500):
        """
        Generates a response from the LLM asynchronously based on the provided prompt.

        Args:
            prompt (str): The prompt string to send to the LLM.
            callback (Callable[[Dict[str, Any]], None]): The function to call with the response once it's ready.
            max_tokens (int, optional): The maximum number of tokens in the response.

        Returns:
            None
        """
        def run():
            self.logger.debug(f"Starting async LLM generation for prompt: {prompt}")
            response = self.generate_response(prompt, max_tokens)
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
        action_match = re.search(r"'(move|collect|communicate|rest)'", text)
        return action_match.group(1) if action_match else 'rest'

    def _extract_details(self, text: str) -> Dict[str, Any]:
        """
        Extracts additional details from the generated text.

        Args:
            text (str): The generated text from the LLM.

        Returns:
            Dict[str, Any]: A dictionary of extracted details.
        """
        # This is a placeholder. Implement more sophisticated parsing as needed.
        return {}
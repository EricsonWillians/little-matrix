"""
LLM client for interacting with Hugging Face endpoints with robust error handling and token management.
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, Union
from datetime import datetime

from huggingface_hub import InferenceClient, InferenceTimeoutError
from dotenv import load_dotenv
from transformers import AutoTokenizer

from ..llm.prompts import PromptManager


class LLMClient:
    """
    LLMClient is responsible for interacting with the Hugging Face Inference API or dedicated endpoints,
    handling retries, rate limiting, token length management, and error handling to ensure robust communication
    with the language model.
    """
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 1.0
    MIN_REQUEST_INTERVAL = 0.1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_environment()
        self.prompt_manager = PromptManager(config)
        self.client = InferenceClient(
            model=self.endpoint,
            token=self.api_key,
            timeout=self.request_timeout
        )
        self.last_request_time = 0.0
        self._initialize_metrics()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_total_tokens = self._get_model_max_tokens()
        self.logger.info(f"LLMClient initialized with model '{self.model_name}' and max tokens {self.max_total_tokens}.")

    def _setup_environment(self) -> None:
        load_dotenv()
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.endpoint = os.getenv('HUGGINGFACE_ENDPOINT')
        self.model_name = os.getenv('HUGGINGFACE_MODEL', 'gpt2')
        self.request_timeout = int(os.getenv('HUGGINGFACE_REQUEST_TIMEOUT', '60'))

        if not self.api_key:
            self.logger.error("Missing HUGGINGFACE_API_KEY")
            raise ValueError("Missing HUGGINGFACE_API_KEY")
        if not self.endpoint:
            self.logger.error("Missing HUGGINGFACE_ENDPOINT")
            raise ValueError("Missing HUGGINGFACE_ENDPOINT")

    def _initialize_metrics(self) -> None:
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency': 0.0,
            'last_success': None,
        }

    def _get_model_max_tokens(self) -> int:
        # Define model-specific max token limits
        model_max_tokens = {
            'Qwen/Qwen2.5-7B-Instruct': 8192,
            # Add other models and their max token limits here
        }
        return model_max_tokens.get(self.model_name, 2048)  # Default to 2048 if not specified

    def generate_response(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        self.metrics['total_requests'] += 1
        start_time = time.time()

        input_ids = self.tokenizer.encode(prompt, truncation=False)
        self.logger.debug(f"Prompt length (tokens): {len(input_ids)}")

        # Ensure total tokens do not exceed model's maximum allowed tokens
        if len(input_ids) + max_tokens > self.max_total_tokens:
            max_allowed_input_tokens = self.max_total_tokens - max_tokens
            if max_allowed_input_tokens <= 0:
                error_msg = "max_tokens is too high for the model's token limit."
                self.logger.error(error_msg)
                return {"action": "rest", "details": {}, "error": error_msg}
            # Truncate the input_ids from the beginning to fit the max allowed input tokens
            input_ids = input_ids[-max_allowed_input_tokens:]
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            self.logger.warning("Prompt truncated to fit model's token limit.")

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

        try:
            self._enforce_rate_limit()
            response_text = self._retry_request(payload)
            result = self._parse_response(response_text)
            self._update_metrics(success=True, start_time=start_time)
            return result

        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            self._update_metrics(success=False, start_time=start_time)
            return {"action": "rest", "details": {}, "error": str(e)}

    def _retry_request(self, payload: Dict[str, Any]) -> str:
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = self.client.text_generation(
                    prompt=payload["prompt"],
                    max_new_tokens=payload["max_new_tokens"],
                    temperature=payload["temperature"],
                    top_p=payload["top_p"],
                    do_sample=payload["do_sample"],
                )
                if not response:
                    raise ValueError("Empty response from LLM.")
                return response

            except (InferenceTimeoutError, ValueError) as e:
                retries += 1
                self.logger.warning(f"Request failed: {e}. Retrying {retries}/{self.MAX_RETRIES}...")
                time.sleep(self.BACKOFF_FACTOR * retries)

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise

        raise Exception("Max retries exceeded for the request.")


    def generate_prompted_response(
        self,
        prompt_name: str,
        prompt_kwargs: Dict[str, Any],
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        try:
            prompt = self.prompt_manager.get_prompt(prompt_name, **prompt_kwargs)
            return self.generate_response(prompt, max_tokens)
        except Exception as e:
            self.logger.error(f"Failed to generate prompted response: {e}")
            return {"action": "rest", "details": {}, "error": str(e)}

    def _enforce_rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _update_metrics(self, success: bool, start_time: float) -> None:
        elapsed_time = time.time() - start_time
        self.metrics['total_latency'] += elapsed_time

        if success:
            self.metrics['successful_requests'] += 1
            self.metrics['last_success'] = datetime.now()
        else:
            self.metrics['failed_requests'] += 1

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parses the generated text from the LLM and extracts action and details.

        Returns:
            Dict[str, Any]: A dictionary containing the action, details, and full response.
        """
        self.logger.debug(f"Raw LLM response: {repr(text)}")
        try:
            # Extract JSON object using a stack-based approach
            import json
            start_idx = text.find('{')
            if start_idx == -1:
                self.logger.error("No '{' found in LLM response.")
                raise ValueError("No '{' found in LLM response.")

            brace_count = 0
            in_string = False
            for idx in range(start_idx, len(text)):
                char = text[idx]
                if char == '"' and text[idx - 1] != '\\':
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # We found the matching closing brace
                            json_str = text[start_idx:idx+1]
                            self.logger.debug(f"Extracted JSON string: {json_str}")
                            response = json.loads(json_str)
                            action = response.get('action', 'rest')
                            details = response.get('details', {})
                            return {
                                "action": action,
                                "details": details,
                                "full_response": text
                            }
            # If we reach here, no complete JSON object was found
            self.logger.error("No complete JSON object found in LLM response.")
            return {
                "action": "rest",
                "details": {},
                "full_response": text,
                "error": "No complete JSON object found in LLM response."
            }
        except Exception as e:
            self.logger.error(f"Error parsing response text: {e}")
            # Return default action with the raw response
            return {
                "action": "rest",
                "details": {},
                "full_response": text,
                "error": str(e)
            }


    def get_metrics(self) -> Dict[str, Union[int, float, str]]:
        """Return current performance metrics."""
        successful_requests = self.metrics['successful_requests']
        total_requests = self.metrics['total_requests']
        avg_latency = (
            self.metrics['total_latency'] / successful_requests
            if successful_requests > 0 else 0.0
        )
        success_rate = (
            successful_requests / total_requests
            if total_requests > 0 else 0.0
        )
        return {
            **self.metrics,
            'average_latency': avg_latency,
            'success_rate': success_rate
        }

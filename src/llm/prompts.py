# llm/prompts.py

"""
Prompts module for the little-matrix simulation.

This module defines the PromptManager class, which stores and manages prompt templates
for agent communication and other interactions with the LLM. It provides a centralized
location for all prompt templates used by agents for decision-making, communication,
and querying the environment.

Classes:
    PromptManager: Manages prompt templates.
"""

import logging
from typing import Dict


class PromptManager:
    """
    Manages prompt templates for the LLM.

    Attributes:
        prompts (Dict[str, str]): A dictionary of prompt templates.
        logger (logging.Logger): Logger for the PromptManager.
    """

    def __init__(self):
        """
        Initializes the PromptManager with default prompts.
        """
        self.logger = logging.getLogger(__name__)
        self.prompts: Dict[str, str] = {
            "agent_decision": (
                "<|system|>\n"
                "You are a highly intelligent and autonomous agent named {agent_name}. "
                "Your current state is {state}, and your knowledge base includes {knowledge_base}. "
                "You perceive the following agents: {perceived_agents} and objects: {perceived_objects}.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Based on this information, decide on the best action to take from the options: "
                "'move', 'collect', 'communicate', 'rest', or specify a custom action.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Please respond with only the action (e.g., 'move', 'collect', 'communicate', 'rest').\n"
            ),
            "agent_communication": (
                "<|system|>\n"
                "You are an autonomous agent named {sender_name}. You are communicating with {recipient_name}. "
                "Your message is: \"{message_content}\". The recipient's state is {recipient_state}, and their "
                "knowledge base includes {recipient_knowledge_base}.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Craft an appropriate response to this message.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Please respond with only the message content.\n"
            ),
            "environment_query": (
                "<|system|>\n"
                "You are an agent named {agent_name}. You observe the environment and have the following query: \"{query}\".\n"
                "<|end|>\n"
                "<|user|>\n"
                "Provide information relevant to the query based on the current world state.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Please respond with only the relevant information.\n"
            ),
            "resource_request": (
                "<|system|>\n"
                "You are an agent named {agent_name}. You request resources of type '{resource_type}'.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Determine if the request can be fulfilled based on the available resources.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Please respond with only 'Yes' or 'No'.\n"
            ),
            "status_report": (
                "<|system|>\n"
                "You are an agent named {agent_name}. Your current status is: health={health}, energy={energy}, mood='{mood}', "
                "inventory={inventory}.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Summarize your status for logging or display purposes.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Please respond with a concise summary of your status.\n"
            ),
            # Additional prompt templates can be added here.
        }
        self.logger.info("PromptManager initialized with default prompts.")

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Retrieves and formats a prompt template.

        Args:
            prompt_name (str): The name of the prompt template to retrieve.
            **kwargs: Variables to format the prompt template.

        Returns:
            str: The formatted prompt ready to be sent to the LLM.

        Raises:
            ValueError: If the prompt_name does not exist in the prompts dictionary or if required keys are missing.
        """
        template = self.prompts.get(prompt_name)
        if not template:
            error_message = f"Prompt '{prompt_name}' not found in PromptManager."
            self.logger.error(error_message)
            raise ValueError(error_message)
        try:
            prompt = template.format(**kwargs)
            self.logger.debug(f"Generated prompt for '{prompt_name}': {prompt}")
            return prompt
        except KeyError as e:
            missing_key = e.args[0]
            error_message = f"Missing key '{missing_key}' in prompt variables for '{prompt_name}'."
            self.logger.error(error_message)
            raise ValueError(error_message)

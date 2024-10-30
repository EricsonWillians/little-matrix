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
from typing import Dict, Any, Optional
import random
from ..utils.config import Config  # Import Config for type hinting if needed

class PromptManager:
    """
    Manages prompt templates for the LLM.

    Attributes:
        prompts (Dict[str, str]): A dictionary of prompt templates.
        logger (logging.Logger): Logger for the PromptManager.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initializes the PromptManager with default prompts, optionally using configuration.

        Args:
            config (Config, optional): Configuration object to customize prompt behavior.
        """
        self.logger = logging.getLogger(__name__)
        self.prompts: Dict[str, str] = {
            "agent_decision": (
                "<|system|>\n"
                "You are a highly intelligent and autonomous agent named {agent_name} in a dynamic, evolving world. "
                "Your current state is {state}, and your knowledge base includes {knowledge_base}. "
                "Your inventory contains: {inventory}. Your skills are: {skills}. "
                "Your current goals are: {goals}. Your relationships with other agents: {relationships}. "
                "You perceive the following agents: {perceived_agents} and objects: {perceived_objects}.\n"
                "The world around you is constantly changing, and your actions can have far-reaching consequences. "
                "Consider the potential for exploration, resource gathering, skill improvement, and social interactions.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Based on this information, decide on the best action to take. Your options include, but are not limited to:\n"
                "1. 'move': Explore new areas or approach objects/agents of interest.\n"
                "2. 'collect': Gather resources or items that could be useful.\n"
                "3. 'communicate': Interact with other agents to share information or build relationships.\n"
                "4. 'use': Utilize items in your inventory or interact with the environment.\n"
                "5. 'craft': Create new items from resources you've gathered.\n"
                "6. 'rest': Recover energy, but consider if this is the best use of your time.\n"
                "7. 'analyze': Study your surroundings or an object in detail.\n"
                "8. 'trade': Exchange resources or items with other agents.\n"
                "9. 'train': Improve one of your skills.\n"
                "10. 'plan': Formulate a strategy for achieving your goals.\n"
                "Choose an action that best advances your goals, improves your situation, or contributes to the world around you.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Respond with your chosen action and a brief explanation. Format: 'action: explanation'\n"
            ),
            "agent_communication": (
                "<|system|>\n"
                "You are an autonomous agent named {sender_name} in a complex, interconnected world. "
                "You are communicating with {recipient_name}. Your message is: \"{message_content}\". "
                "The recipient's state is {recipient_state}, and their knowledge base includes {recipient_knowledge_base}. "
                "Your relationship with {recipient_name} is: {relationship}. "
                "Consider the potential for collaboration, information exchange, or conflict.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Craft an appropriate response to this message. Your response should further your goals, "
                "maintain or improve relationships, and potentially lead to mutually beneficial outcomes.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Respond with your message content, keeping in mind the context and potential consequences.\n"
            ),
            "environment_query": (
                "<|system|>\n"
                "You are an agent named {agent_name} in a rich, detailed world. You observe the environment "
                "and have the following query: \"{query}\". Your current knowledge of the world includes: {world_knowledge}. "
                "Consider both obvious and subtle aspects of the environment.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Provide detailed information relevant to the query based on the current world state. "
                "Include any potential implications or hidden factors that might not be immediately obvious.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Respond with relevant information, analysis, and potential implications.\n"
            ),
            "resource_request": (
                "<|system|>\n"
                "You are an agent named {agent_name} in a world of limited resources. "
                "You request resources of type '{resource_type}'. The current resource availability is: {resource_availability}. "
                "Consider the broader implications of resource allocation and potential alternatives.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Determine if the request can be fulfilled based on the available resources. "
                "If not, suggest alternatives or compromises.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Respond with 'Yes', 'No', or a suggested alternative, along with a brief explanation.\n"
            ),
            "status_report": (
                "<|system|>\n"
                "You are an agent named {agent_name} in a dynamic, evolving world. Your current status is: "
                "health={health}, energy={energy}, mood='{mood}', inventory={inventory}, skills={skills}, "
                "goals={goals}, relationships={relationships}.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Provide a comprehensive status report. Include your current condition, progress towards goals, "
                "notable changes in your environment or relationships, and any concerns or opportunities you've identified.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Respond with a detailed status report, highlighting key information and potential future actions.\n"
            ),
            "world_event": (
                "<|system|>\n"
                "You are the narrator of a complex, evolving world. A significant event is occurring: {event_description}. "
                "This event has the potential to affect multiple agents and aspects of the world.\n"
                "<|end|>\n"
                "<|user|>\n"
                "Describe the immediate and potential long-term consequences of this event on the world and its inhabitants. "
                "Consider environmental, social, and individual impacts.\n"
                "<|end|>\n"
                "<|assistant|>\n"
                "Provide a vivid description of the event's impact and potential consequences.\n"
            ),
        }
        self.config = config
        self.logger.info("PromptManager initialized with enhanced prompts.")

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Retrieves and formats a prompt template, with additional randomization for variety.

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
            # Add some randomization to the prompt to encourage variety in responses
            kwargs = self._add_prompt_variety(prompt_name, kwargs)
            prompt = template.format(**kwargs)
            self.logger.debug(f"Generated prompt for '{prompt_name}': {prompt}")
            return prompt
        except KeyError as e:
            missing_key = e.args[0]
            error_message = f"Missing key '{missing_key}' in prompt variables for '{prompt_name }'."
            self.logger.error(error_message)
            raise ValueError(error_message)

    def _add_prompt_variety(self, prompt_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds some randomization to the prompt to encourage variety in responses.

        Args:
            prompt_name (str): The name of the prompt template.
            kwargs (Dict[str, Any]): The variables to format the prompt template.

        Returns:
            Dict[str, Any]: The updated variables with added randomization.
        """
        if prompt_name == "agent_decision":
            # Randomly select a goal or skill to focus on
            if "goals" in kwargs:
                goal_focus = random.choice(kwargs["goals"])
                kwargs["goal_focus"] = goal_focus
            if "skills" in kwargs:
                skill_focus = random.choice(kwargs["skills"])
                kwargs["skill_focus"] = skill_focus
        elif prompt_name == "agent_communication":
            # Randomly select a tone or topic for the message
            tone_options = ["friendly", "cautious", "assertive"]
            topic_options = ["resource sharing", "goal alignment", "social bonding"]
            kwargs["tone"] = random.choice(tone_options)
            kwargs["topic"] = random.choice(topic_options)
        # Add more randomization options for other prompts as needed
        return kwargs

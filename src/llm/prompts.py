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
from typing import Dict, Any, Optional, List
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
        Initializes the PromptManager with optimized prompts, optionally using configuration.

        Args:
            config (Config, optional): Configuration object to customize prompt behavior.
        """
        self.logger = logging.getLogger(__name__)
        self.prompts: Dict[str, str] = {
            "agent_decision": (
                "You are {agent_name}, an agent in a survival world.\n"
                "Your current stats are:\n"
                "- Health: {state[health]}\n"
                "- Energy: {state[energy]}\n"
                "- Hunger: {state[hunger]} (0 means full, 100 means starving)\n"
                "- Thirst: {state[thirst]} (0 means hydrated, 100 means dehydrated)\n"
                "Inventory: {inventory_summary}\n"
                "Nearby items: {nearby_summary}\n\n"
                "**Based on your current needs, choose the most appropriate action.**\n"
                "Only seek resources if you need them. If you don't need any resources, consider other actions like exploring or resting.\n"
                "Respond ONLY with your decision in JSON format, including any necessary details, without any additional text or explanation.\n"
                "\nActions and required details:\n"
                "- seek_resource (requires 'resource_type': 'food', 'water', or 'medical_supplies')\n"
                "- collect (requires 'object_type')\n"
                "- use (requires 'item_type')\n"
                "- move (requires 'direction': 'north', 'south', 'east', or 'west')\n"
                "- rest\n\n"
                "Format:\n"
                "{{\"action\": \"action_name\", \"details\": {{...}}}}\n"
                "Example:\n"
                "{{\"action\": \"rest\", \"details\": {{}}}}\n"
                "Do not include any extra text."
            ),
            "agent_communication": (
                "You are {sender_name}, communicating with {recipient_name}. Message: \"{message_content}\".\n"
                "Your relationship: {relationship}.\n"
                "Respond appropriately to further your goals.\n"
            ),
            "environment_query": (
                "Agent {agent_name} observes: \"{query}\".\n"
                "Based on your knowledge: {knowledge_summary}, provide relevant information.\n"
            ),
            "resource_request": (
                "Agent {agent_name} requests '{resource_type}'. Available resources: {resource_summary}.\n"
                "Determine if the request can be fulfilled and respond accordingly.\n"
            ),
            "status_report": (
                "Agent {agent_name} status:\n"
                "Health {health}, Energy {energy}, Mood '{mood}'.\n"
                "Inventory: {inventory_summary}\n"
                "Goals: {goals_summary}\n"
                "Provide a brief status report.\n"
            ),
            "world_event": (
                "A significant event occurs: {event_description}.\n"
                "Describe its immediate and potential long-term consequences.\n"
            ),
        }
        self.config = config
        self.logger.info("PromptManager initialized with optimized prompts.")

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Retrieves and formats a prompt template, optimizing variable content to limit length.

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
            kwargs = self._prepare_prompt_variables(prompt_name, kwargs)
            prompt = template.format(**kwargs)
            self.logger.debug(f"Generated prompt for '{prompt_name}': {prompt}")
            return prompt
        except KeyError as e:
            missing_key = e.args[0]
            error_message = f"Missing key '{missing_key}' in prompt variables for '{prompt_name}'."
            self.logger.error(error_message)
            raise ValueError(error_message)

    def _prepare_prompt_variables(self, prompt_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares and optimizes the prompt variables to prevent excessive length.

        Args:
            prompt_name (str): The name of the prompt template.
            kwargs (Dict[str, Any]): The variables to format the prompt template.

        Returns:
            Dict[str, Any]: The updated variables with optimized content.
        """
        # Limit the number of items in lists to reduce prompt length
        max_items = 2  # Adjust as needed to fit within token limits

        # Shorten inventory
        if 'inventory' in kwargs:
            inventory = kwargs['inventory']
            if isinstance(inventory, list):
                limited_inventory = inventory[:max_items]
                kwargs['inventory_summary'] = ', '.join(map(str, limited_inventory))
                if len(inventory) > max_items:
                    kwargs['inventory_summary'] += ', ...'
            else:
                kwargs['inventory_summary'] = str(inventory)
        else:
            kwargs['inventory_summary'] = 'Empty'

        # Shorten perceived objects
        if 'perceived_objects' in kwargs:
            perceived_objects = kwargs['perceived_objects']
            if isinstance(perceived_objects, list):
                limited_perceived = perceived_objects[:max_items]
                kwargs['nearby_summary'] = ', '.join(map(str, limited_perceived))
                if len(perceived_objects) > max_items:
                    kwargs['nearby_summary'] += ', ...'
            else:
                kwargs['nearby_summary'] = str(perceived_objects)
        else:
            kwargs['nearby_summary'] = 'Nothing notable'

        # Shorten goals
        if 'goals' in kwargs:
            goals = kwargs['goals']
            if isinstance(goals, list):
                limited_goals = goals[:max_items]
                kwargs['goals_summary'] = ', '.join(map(str, limited_goals))
                if len(goals) > max_items:
                    kwargs['goals_summary'] += ', ...'
            else:
                kwargs['goals_summary'] = str(goals)
        else:
            kwargs['goals_summary'] = 'No specific goals'

        # Shorten knowledge
        if 'world_knowledge' in kwargs:
            knowledge = kwargs['world_knowledge']
            if isinstance(knowledge, list):
                limited_knowledge = knowledge[:max_items]
                kwargs['knowledge_summary'] = ', '.join(map(str, limited_knowledge))
                if len(knowledge) > max_items:
                    kwargs['knowledge_summary'] += ', ...'
            else:
                kwargs['knowledge_summary'] = str(knowledge)
        else:
            kwargs['knowledge_summary'] = 'Basic understanding'

        # Shorten resource availability
        if 'resource_availability' in kwargs:
            resources = kwargs['resource_availability']
            if isinstance(resources, dict):
                limited_resources = dict(list(resources.items())[:max_items])
                kwargs['resource_summary'] = ', '.join(f"{k}: {v}" for k, v in limited_resources.items())
                if len(resources) > max_items:
                    kwargs['resource_summary'] += ', ...'
            else:
                kwargs['resource_summary'] = str(resources)
        else:
            kwargs['resource_summary'] = 'Unknown'

        # Shorten relationships
        if 'relationships' in kwargs:
            relationships = kwargs['relationships']
            if isinstance(relationships, dict):
                limited_relationships = dict(list(relationships.items())[:max_items])
                kwargs['relationships_summary'] = ', '.join(f"{k}: {v}" for k, v in limited_relationships.items())
                if len(relationships) > max_items:
                    kwargs['relationships_summary'] += ', ...'
            else:
                kwargs['relationships_summary'] = str(relationships)
        else:
            kwargs['relationships_summary'] = 'No significant relationships'

        # Other variables can be optimized similarly
        return kwargs

    def _add_prompt_variety(self, prompt_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds randomization to the prompt to encourage variety in responses.

        Args:
            prompt_name (str): The name of the prompt template.
            kwargs (Dict[str, Any]): The variables to format the prompt template.

        Returns:
            Dict[str, Any]: The updated variables with added randomization.
        """
        if prompt_name == "agent_decision":
            # Randomly select a focus for the agent
            focus_options = ["survival", "exploration", "resource gathering"]
            kwargs["agent_focus"] = random.choice(focus_options)
        elif prompt_name == "agent_communication":
            # Randomly select a tone for the message
            tone_options = ["friendly", "neutral", "assertive"]
            kwargs["tone"] = random.choice(tone_options)
        # Add more randomization options for other prompts as needed
        return kwargs

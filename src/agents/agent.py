# agents/agent.py

"""
Agent module for the little-matrix simulation.

This module defines the Agent class, representing a sophisticated autonomous entity
within the simulated world. Agents can perceive their environment, make decisions using
advanced logic and LLM assistance, act upon the world, and communicate with other agents.
They possess internal states, knowledge bases, and can adapt to changing conditions,
embodying the essence of a 'supreme agent' within the matrix.

Classes:
    Agent: Represents an individual agent in the simulation.
"""

import logging
import random
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from llm.client import LLMClient
from environment.objects import WorldObject
import json  # Import json for serialization

if TYPE_CHECKING:
    from environment.world import World
    from agents.agent import Agent


class Agent:
    """
    Represents an agent in the little-matrix simulation.

    Attributes:
        name (str): The unique name of the agent.
        llm_client (LLMClient): Instance of the LLM client for advanced decision-making.
        state (Dict[str, Any]): Current state of the agent, including health, energy, and mood.
        knowledge_base (List[str]): Memory or knowledge accumulated by the agent.
        position (Tuple[int, int]): Coordinates of the agent in the world.
        inventory (Dict[str, int]): Items and resources the agent has collected.
        perception_radius (int): The radius within which the agent can perceive the environment.
        communication_range (int): The range within which the agent can communicate with others.
        logger (logging.Logger): Logger for the agent.
    """

    def __init__(self, name: str, llm_client: LLMClient, position: Optional[Tuple[int, int]] = None):
        """
        Initializes a new Agent instance.

        Args:
            name (str): The unique name of the agent.
            llm_client (LLMClient): The LLM client used for advanced decision-making.
            position (Tuple[int, int], optional): The initial position of the agent in the world.
        """
        self.name = name
        self.llm_client = llm_client
        self.state: Dict[str, Any] = {'health': 100, 'energy': 100, 'mood': 'neutral'}
        self.knowledge_base: List[str] = []
        self.position = position if position is not None else (0, 0)
        self.inventory: Dict[str, int] = {}
        self.perception_radius = 5  # Cells
        self.communication_range = 10  # Cells
        self.logger = logging.getLogger(f"Agent:{self.name}")
        self.logger.info(f"Agent '{self.name}' initialized at position {self.position}.")

    def perceive(self, world: 'World'):
        """
        Perceives the environment within the agent's perception radius and updates internal state.

        Args:
            world (World): The simulation environment.

        Returns:
            None
        """
        # Get nearby agents
        perceived_agents = world.get_agents_within_radius(self.position, self.perception_radius)
        # Exclude self from perceived agents
        perceived_agents = [agent for agent in perceived_agents if agent.name != self.name]
        # Get nearby objects
        perceived_objects = world.get_objects_within_radius(self.position, self.perception_radius)

        # Update the agent's state with serializable data
        self.state['perceived_agents'] = [agent.name for agent in perceived_agents]
        self.state['perceived_objects'] = [
            {'type': obj.__class__.__name__, 'position': obj.position}
            for obj in perceived_objects
        ]

        self.logger.debug(f"{self.name} perceived agents: {self.state['perceived_agents']}")
        self.logger.debug(f"{self.name} perceived objects: {self.state['perceived_objects']}")

    def decide(self):
        """
        Makes a decision based on the perceived environment and internal state.

        Uses the LLM to enhance decision-making with advanced reasoning.

        Returns:
            None
        """
        # Prepare variables for the prompt
        # Serialize complex structures if necessary
        prompt_vars = {
            'agent_name': self.name,
            'state': ", ".join([f"{k}={v}" for k, v in self.state.items() if k not in ['perceived_agents', 'perceived_objects', 'action']]),
            'knowledge_base': ", ".join(self.knowledge_base) if self.knowledge_base else "None",
            'perceived_agents': ", ".join(self.state.get('perceived_agents', [])) if self.state.get('perceived_agents') else "None",
            'perceived_objects': ", ".join(
                [f"{obj['type']} at {obj['position']}" for obj in self.state.get('perceived_objects', [])]
            ) if self.state.get('perceived_objects') else "None",
        }
        self.logger.debug(f"{self.name} is making a decision with variables: {prompt_vars}")

        try:
            # Generate the prompt using the PromptManager
            prompt = self.llm_client.prompt_manager.get_prompt('agent_decision', **prompt_vars)
        except ValueError as e:
            self.logger.error(f"Failed to generate prompt: {e}")
            self.state['action'] = "rest"
            return

        # Generate a response from the LLM using the constructed prompt
        action = self.llm_client.generate_response(prompt)
        action = action.strip().lower()

        # Validate the action
        valid_actions = {'move', 'collect', 'communicate', 'rest'}
        if action not in valid_actions:
            self.logger.warning(f"{self.name} received an invalid action '{action}'. Defaulting to 'rest'.")
            action = 'rest'
        self.state['action'] = action
        self.logger.info(f"{self.name} decided to '{self.state['action']}'.")

    def act(self, world: 'World'):
        """
        Performs the decided action in the environment.

        Args:
            world (World): The simulation environment.

        Returns:
            None
        """
        action = self.state.get('action')
        if not action:
            self.logger.warning(f"{self.name} has no action to perform.")
            return

        if action == 'move':
            self.move(world)
        elif action == 'collect':
            self.collect(world)
        elif action == 'communicate':
            self.initiate_communication(world)
        elif action == 'rest':
            self.rest()
        else:
            self.logger.info(f"{self.name} performs a custom action: {action}")
            # Implement custom action logic if needed

    def move(self, world: 'World'):
        """
        Moves the agent to a new position based on simple pathfinding or random movement.

        Args:
            world (World): The simulation environment.

        Returns:
            None
        """
        empty_adjacent_positions = world.get_empty_adjacent_positions(self.position)
        if empty_adjacent_positions:
            new_position = random.choice(empty_adjacent_positions)
            try:
                world.move_agent(self, new_position)
                self.position = new_position
                self.state['energy'] -= 5  # Moving costs energy
                self.logger.info(f"{self.name} moved to {self.position}.")
            except ValueError as e:
                self.logger.error(f"{self.name} failed to move: {e}")
        else:
            self.logger.info(f"{self.name} cannot move; no empty adjacent positions.")

    def collect(self, world: 'World'):
        """
        Collects resources or items from the current position.

        Args:
            world (World): The simulation environment.

        Returns:
            None
        """
        objects_at_position = world.get_objects_at_position(self.position)
        for obj in objects_at_position:
            if isinstance(obj, WorldObject):
                obj.interact(self)
                if obj.should_be_removed():
                    world.remove_object(obj)
                    self.logger.info(f"{self.name} collected {obj}.")
                break  # Only interact with one object per action
        else:
            self.logger.info(f"{self.name} found nothing to collect at {self.position}.")

    def rest(self):
        """
        Rests to regain energy and improve mood.

        Returns:
            None
        """
        self.state['energy'] += 10
        self.state['mood'] = 'rested'
        if self.state['energy'] > 100:
            self.state['energy'] = 100
        self.logger.info(f"{self.name} rested and now has energy {self.state['energy']}.")

    def initiate_communication(self, world: 'World'):
        """
        Initiates communication with nearby agents within communication range.

        Args:
            world (World): The simulation environment.

        Returns:
            None
        """
        nearby_agents = world.get_agents_within_radius(self.position, self.communication_range)
        for agent in nearby_agents:
            if agent.name != self.name:
                message = f"Hello {agent.name}, this is {self.name}."
                self.communicate(message, agent)
                self.logger.info(f"{self.name} communicated with {agent.name}.")
                break  # Communicate with one agent per action

    def communicate(self, message: str, recipient: 'Agent'):
        """
        Communicates with another agent using the LLM for message generation.

        Args:
            message (str): The message content.
            recipient (Agent): The agent to communicate with.

        Returns:
            None
        """
        # Prepare variables for the prompt
        prompt_vars = {
            'sender_name': self.name,
            'recipient_name': recipient.name,
            'message_content': message,
            'recipient_state': ", ".join([f"{k}={v}" for k, v in recipient.state.items() if k not in ['perceived_agents', 'perceived_objects', 'action']]),
            'recipient_knowledge_base': ", ".join(recipient.knowledge_base) if recipient.knowledge_base else "None",
        }

        try:
            # Generate the prompt using the PromptManager
            prompt = self.llm_client.prompt_manager.get_prompt('agent_communication', **prompt_vars)
        except ValueError as e:
            self.logger.error(f"Failed to generate communication prompt: {e}")
            return

        # Generate a response from the LLM
        response = self.llm_client.generate_response(prompt)
        response = response.strip().lower()

        # Optionally, validate the response format or content here

        recipient.receive_message(response, sender=self)
        self.logger.debug(f"{self.name} sent a message to {recipient.name}: {message}")
        self.logger.debug(f"{self.name} received response: {response}")

    def receive_message(self, message: str, sender: 'Agent'):
        """
        Processes incoming messages from other agents.

        Args:
            message (str): The message content received.
            sender (Agent): The agent who sent the message.

        Returns:
            None
        """
        self.knowledge_base.append({'from': sender.name, 'message': message})
        self.logger.info(f"{self.name} received a message from {sender.name}: {message}")

    def collect_resource(self, resource_type: str, amount: int) -> int:
        """
        Collects a specified amount of a resource, updating the agent's inventory.

        Args:
            resource_type (str): The type of resource to collect.
            amount (int): The amount available to collect.

        Returns:
            int: The amount actually collected.
        """
        collected_amount = min(amount, 10)  # Limit how much can be collected at once
        self.inventory[resource_type] = self.inventory.get(resource_type, 0) + collected_amount
        self.logger.info(f"{self.name} collected {collected_amount} units of {resource_type}.")
        return collected_amount

    def take_damage(self, damage: int):
        """
        Reduces the agent's health by the specified damage amount.

        Args:
            damage (int): The amount of damage to inflict.

        Returns:
            None
        """
        self.state['health'] -= damage
        self.logger.warning(f"{self.name} took {damage} damage. Health is now {self.state['health']}.")
        if self.state['health'] <= 0:
            self.state['health'] = 0
            self.logger.error(f"{self.name} has been defeated.")

    def update_state(self):
        """
        Updates the agent's internal state after each timestep.

        Returns:
            None
        """
        # Decrease energy over time
        self.state['energy'] -= 1
        if self.state['energy'] <= 0:
            self.state['energy'] = 0
            self.state['mood'] = 'exhausted'
            self.logger.warning(f"{self.name} is exhausted.")
        # Update mood based on health and energy
        if self.state['health'] < 50:
            self.state['mood'] = 'injured'
        elif self.state['energy'] > 80:
            self.state['mood'] = 'energetic'
        else:
            self.state['mood'] = 'neutral'
        self.logger.debug(f"{self.name} updated state: {self.state}.")

    def __repr__(self):
        """
        Official string representation of the Agent.

        Returns:
            str: String representation of the agent.
        """
        return (f"Agent(name={self.name}, position={self.position}, "
                f"health={self.state['health']}, energy={self.state['energy']})")

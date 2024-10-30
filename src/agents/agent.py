# src/agents/agent.py

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
import json
import re
import math
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

from ..llm.client import LLMClient
from ..environment.objects import WorldObject
from ..utils.config import (
    Config,
    AgentTypeConfig,
    AgentTypeBehaviorTraits,
    AgentsBehaviorConfig,
    AgentsPerceptionConfig,
)

if TYPE_CHECKING:
    from src.environment.world import World


class Agent:
    """
    Represents a supreme agent in the little-matrix simulation.

    Attributes:
        name (str): The unique name of the agent.
        llm_client (LLMClient): Instance of the LLM client for advanced decision-making.
        config (Config): The configuration object for accessing global settings.
        state (Dict[str, Any]): Current state of the agent.
        knowledge_base (List[Dict[str, Any]]): Memory or knowledge accumulated by the agent.
        position (Tuple[int, int]): Coordinates of the agent in the world.
        inventory (Dict[str, int]): Items and resources the agent has collected.
        perception_radius (int): The radius within which the agent can perceive the environment.
        communication_range (int): The range within which the agent can communicate with others.
        logger (logging.Logger): Logger for the agent.
        skills (Dict[str, float]): Various skills and their proficiency levels.
        goals (List[str]): Current goals or objectives of the agent.
        relationships (Dict[str, float]): Relationships with other agents.
        agent_type (AgentTypeConfig): The type or class of the agent.
        behavior_traits (AgentTypeBehaviorTraits): Behavior traits specific to the agent type.
        status_effects (List[str]): Current status effects impacting the agent.
    """

    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        config: Config,
        position: Optional[Tuple[int, int]] = None,
        state: Optional[Dict[str, Any]] = None,
        agent_type: Optional[AgentTypeConfig] = None,
        behavior_traits: Optional[AgentTypeBehaviorTraits] = None,
    ):
        """
        Initializes a new Agent instance using configuration settings.

        Args:
            name (str): The unique name of the agent.
            llm_client (LLMClient): The LLM client used for advanced decision-making.
            config (Config): The global configuration object.
            position (Tuple[int, int], optional): The initial position of the agent in the world.
            state (Dict[str, Any], optional): The initial state of the agent (used when loading from storage).
            agent_type (AgentTypeConfig, optional): The agent type configuration.
            behavior_traits (AgentTypeBehaviorTraits, optional): The behavior traits of the agent.
        """
        self.name = name
        self.llm_client = llm_client
        self.config = config
        self.logger = logging.getLogger(f"Agent:{self.name}")

        # Determine agent type
        if agent_type is None:
            self.agent_type = random.choice(config.agents.customization.types)
        else:
            self.agent_type = agent_type

        # Assign behavior traits
        if behavior_traits is None:
            self.behavior_traits = self.agent_type.behavior_traits
        else:
            self.behavior_traits = behavior_traits

        # Initialize state
        self.state = state if state is not None else self._initialize_state()
        self._ensure_state_keys()

        # Initialize position
        self.position = position if position is not None else self._get_random_position()

        self.knowledge_base: List[Dict[str, Any]] = []
        self.inventory: Dict[str, int] = self._initialize_inventory()

        self.perception_radius = self._calculate_perception_radius()
        self.communication_range = self.perception_radius + 2  # Example adjustment

        self.skills: Dict[str, float] = self._initialize_skills()
        self.goals: List[str] = random.sample(
            ['survive', 'explore', 'socialize', 'learn', 'dominate'], k=2
        )
        self.relationships: Dict[str, float] = {}
        self.status_effects: List[str] = []

        self.logger.info(
            f"Agent '{self.name}' of type '{self.agent_type.name}' initialized at position {self.position}."
        )

    def _initialize_state(self) -> Dict[str, Any]:
        """
        Initializes the agent's state based on configuration settings.

        Returns:
            Dict[str, Any]: The initialized state dictionary.
        """
        behavior_config: AgentsBehaviorConfig = self.config.agents.behavior
        initial_energy = behavior_config.initial_energy
        initial_health = 100  # Assuming max health is 100

        state = {
            'health': initial_health,
            'energy': initial_energy,
            'mood': random.choice(['excited', 'curious', 'cautious', 'determined']),
            'hunger': random.randint(0, 30),
            'thirst': random.randint(0, 30),
            'experience': 0,
            'level': 1,
            'perceived_agents': [],
            'perceived_objects': [],
            'action': None,
            'action_details': {},
            'last_message': '',
        }
        return state

    def _ensure_state_keys(self):
        """
        Ensures that all required keys are present in the agent's state.
        """
        default_state = self._initialize_state()
        for key, value in default_state.items():
            if key not in self.state:
                self.state[key] = value
                self.logger.debug(f"State key '{key}' missing. Initialized with default value.")

    def _initialize_inventory(self) -> Dict[str, int]:
        """
        Initializes the agent's inventory.

        Returns:
            Dict[str, int]: The initialized inventory.
        """
        inventory = {}
        # Initialize inventory based on agent type preferences
        if self.agent_type.name == 'Gatherer':
            inventory['food'] = random.randint(2, 5)
            inventory['water'] = random.randint(1, 3)
        else:
            inventory['food'] = random.randint(0, 2)
            inventory['water'] = random.randint(0, 2)
        return inventory

    def _initialize_skills(self) -> Dict[str, float]:
        """
        Initializes the agent's skills based on behavior traits.

        Returns:
            Dict[str, float]: The initialized skills.
        """
        skills = {
            'gathering': random.uniform(0.5, 1.5),
            'crafting': random.uniform(0.5, 1.5),
            'combat': random.uniform(0.5, 1.5),
            'social': random.uniform(0.5, 1.5),
        }
        # Apply skill modifiers based on behavior traits
        if self.behavior_traits.gathering_efficiency:
            skills['gathering'] *= self.behavior_traits.gathering_efficiency
        if self.behavior_traits.combat_skill:
            skills['combat'] *= self.behavior_traits.combat_skill
        return skills

    def _get_random_position(self) -> Tuple[int, int]:
        """
        Generates a random position within the world boundaries.

        Returns:
            Tuple[int, int]: A random position tuple.
        """
        width = self.config.environment.grid.width
        height = self.config.environment.grid.height
        return (random.randint(0, width - 1), random.randint(0, height - 1))

    def _calculate_perception_radius(self) -> int:
        """
        Calculates the perception radius based on agent type and configuration.

        Returns:
            int: The perception radius.
        """
        perception_config: AgentsPerceptionConfig = self.config.agents.perception
        base_radius = perception_config.base_radius
        modifiers = perception_config.modifiers
        type_modifier = modifiers.get(self.agent_type.name, 0)
        perception_radius = base_radius + type_modifier
        return max(1, perception_radius)  # Ensure it's at least 1

    def perceive(self, world: 'World'):
        """
        Perceives the environment within the agent's perception radius and updates internal state.

        Args:
            world (World): The simulation environment.
        """
        try:
            perceived_agents = [
                agent
                for agent in world.get_agents_within_radius(self.position, self.perception_radius)
                if agent.name != self.name
            ]
            perceived_objects = world.get_objects_within_radius(self.position, self.perception_radius)

            self.state['perceived_agents'] = [agent.name for agent in perceived_agents]
            self.state['perceived_objects'] = [
                {'type': obj.__class__.__name__, 'position': obj.position} for obj in perceived_objects
            ]

            # Update knowledge base with new perceptions
            self.update_knowledge_base(
                {
                    'timestamp': world.current_time,
                    'perceived_agents': self.state['perceived_agents'],
                    'perceived_objects': self.state['perceived_objects'],
                }
            )

            self.logger.debug(
                f"{self.name} perceived {len(perceived_agents)} agents and {len(perceived_objects)} objects."
            )
        except Exception as e:
            self.logger.error(f"Error during perception: {e}")

    def decide(self):
        """
        Makes a decision based on the agent's current state and knowledge.
        """
        try:
            prompt_vars = {
                'agent_name': self.name,
                'state': json.dumps(self.state),
                'knowledge_base': json.dumps(self.knowledge_base[-10:]),  # Last 10 memories
                'inventory': json.dumps(self.inventory),
                'skills': json.dumps(self.skills),
                'goals': json.dumps(self.goals),
                'relationships': json.dumps(self.relationships),
                'perceived_agents': json.dumps(self.state.get('perceived_agents', [])),
                'perceived_objects': json.dumps(self.state.get('perceived_objects', [])),
            }

            prompt = self.llm_client.prompt_manager.get_prompt('agent_decision', **prompt_vars)
            response = self.llm_client.generate_response(prompt)

            # Parse the LLM response
            action, details = self._parse_llm_response(response)

            self.state['action'] = action
            self.state['action_details'] = details

            self.logger.info(
                f"{self.name} decided to {self.state['action']} with details: {self.state['action_details']}"
            )
        except Exception as e:
            self.logger.error(f"Decision-making failed: {e}. Defaulting to 'rest'.")
            self.state['action'] = 'rest'
            self.state['action_details'] = {}

    def _parse_llm_response(self, response: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Parses the LLM response to extract action and details.

        Args:
            response (Dict[str, Any]): The raw response from the LLM.

        Returns:
            Tuple[str, Dict[str, Any]]: The parsed action and details.
        """
        generated_text = response.get('full_response', '')

        action_match = re.search(r"action:\s*(\w+)", generated_text, re.IGNORECASE)
        action = action_match.group(1).lower() if action_match else 'rest'

        details = {}
        if action == 'move':
            direction_match = re.search(r"direction:\s*(\w+)", generated_text, re.IGNORECASE)
            details['direction'] = (
                direction_match.group(1).lower() if direction_match else random.choice(['north', 'south', 'east', 'west'])
            )
        elif action == 'collect':
            object_match = re.search(r"object_type:\s*(\w+)", generated_text, re.IGNORECASE)
            details['object_type'] = object_match.group(1) if object_match else 'Resource'
        elif action == 'communicate':
            target_match = re.search(r"target_agent:\s*(\w+)", generated_text, re.IGNORECASE)
            message_match = re.search(r"message:\s*'([^']*)'", generated_text, re.IGNORECASE)
            if target_match and message_match:
                details['target_agent'] = target_match.group(1)
                details['message'] = message_match.group(1)
            else:
                self.logger.warning("Incomplete communication details in LLM response.")
        # Add more action parsing as needed

        return action, details

    def act(self, world: 'World'):
        """
        Performs the decided action in the environment.

        Args:
            world (World): The simulation environment.
        """
        action = self.state.get('action', 'rest')
        details = self.state.get('action_details', {})

        action_map = {
            'move': self.move,
            'collect': self.collect,
            'use': self.use_item,
            'craft': self.craft,
            'communicate': self.initiate_communication,
            'rest': self.rest,
            'attack': self.attack,
        }

        try:
            if action in action_map:
                action_method = action_map[action]
                # Handle actions with or without additional details
                if details:
                    action_method(world, **details)
                else:
                    action_method(world)
            else:
                self.logger.warning(f"{self.name} attempted unknown action: {action}")
        except Exception as e:
            self.logger.error(f"Error performing action '{action}': {e}")

        # Update agent's state after acting
        self.update_state()

    def move(self, world: 'World', direction: str):
        """
        Moves the agent in the specified direction.

        Args:
            world (World): The simulation environment.
            direction (str): The direction to move ('north', 'south', 'east', 'west').
        """
        direction_map = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0),
        }
        if direction not in direction_map:
            self.logger.warning(f"Invalid move direction: {direction}")
            return

        dx, dy = direction_map[direction]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        if self.config.environment.grid.wrap_around:
            new_x %= self.config.environment.grid.width
            new_y %= self.config.environment.grid.height

        new_position = (new_x, new_y)

        if world.is_position_valid(new_position):
            # Remove agent from current position
            world.remove_agent(self)
            self.position = new_position
            # Add agent to new position
            world.add_agent(self, position=self.position)
            energy_cost = self._calculate_movement_energy_cost()
            self.state['energy'] = max(0, self.state['energy'] - energy_cost)
            self.state['hunger'] = min(100, self.state['hunger'] + random.randint(0, 1))
            self.state['thirst'] = min(100, self.state['thirst'] + random.randint(0, 1))
            self.logger.info(f"{self.name} moved {direction} to {self.position}")
        else:
            self.logger.warning(f"Invalid move to {new_position}")

    def _calculate_movement_energy_cost(self) -> float:
        """
        Calculates the energy cost of movement based on agent traits and configuration.

        Returns:
            float: The energy cost of movement.
        """
        base_cost = self.config.agents.behavior.energy_consumption_rate
        speed_modifier = self.behavior_traits.speed_modifier or 1.0
        energy_cost = base_cost / speed_modifier
        return energy_cost

    def collect(self, world: 'World', object_type: str):
        """
        Collects an object of the specified type within the agent's perception radius.

        Args:
            world (World): The simulation environment.
            object_type (str): The type of object to collect.
        """
        objects = world.get_objects_within_radius(self.position, self.perception_radius)
        for obj in objects:
            if isinstance(obj, WorldObject) and obj.__class__.__name__ == object_type:
                self.inventory[object_type] = self.inventory.get(object_type, 0) + 1
                world.remove_object(obj)
                energy_cost = self._calculate_action_energy_cost('collect')
                self.state['energy'] = max(0, self.state['energy'] - energy_cost)
                self.state['hunger'] = min(100, self.state['hunger'] + random.randint(0, 2))
                self.state['thirst'] = min(100, self.state['thirst'] + random.randint(0, 2))
                gathering_skill_increase = self._calculate_skill_increase('gathering')
                self.skills['gathering'] += gathering_skill_increase
                self.logger.info(f"{self.name} collected {object_type} at {self.position}")
                return

        self.logger.warning(f"No {object_type} found within perception radius")

    def _calculate_action_energy_cost(self, action_type: str) -> float:
        """
        Calculates the energy cost for a specific action.

        Args:
            action_type (str): The type of action.

        Returns:
            float: The energy cost.
        """
        base_cost = self.config.agents.behavior.energy_consumption_rate
        if action_type == 'collect':
            efficiency = self.behavior_traits.gathering_efficiency or 1.0
            return base_cost / efficiency
        elif action_type == 'attack':
            combat_skill = self.behavior_traits.combat_skill or 1.0
            return base_cost / combat_skill
        # Add more action-specific calculations as needed
        return base_cost

    def _calculate_skill_increase(self, skill_name: str) -> float:
        """
        Calculates the increase in a skill based on agent traits.

        Args:
            skill_name (str): The name of the skill.

        Returns:
            float: The amount to increase the skill by.
        """
        base_increase = random.uniform(0.01, 0.05)
        if skill_name == 'gathering' and self.behavior_traits.gathering_efficiency:
            return base_increase * self.behavior_traits.gathering_efficiency
        if skill_name == 'combat' and self.behavior_traits.combat_skill:
            return base_increase * self.behavior_traits.combat_skill
        # Add more skill calculations as needed
        return base_increase

    def use_item(self, world: 'World', item_type: str):
        """
        Uses an item of the specified type from the agent's inventory.

        Args:
            world (World): The simulation environment.
            item_type (str): The type of item to use.
        """
        if item_type in self.inventory and self.inventory[item_type] > 0:
            self.inventory[item_type] -= 1
            if item_type == 'food':
                self.state['hunger'] = max(0, self.state['hunger'] - random.randint(15, 25))
                self.state['energy'] = min(100, self.state['energy'] + random.randint(5, 10))
            elif item_type == 'water':
                self.state['thirst'] = max(0, self.state['thirst'] - random.randint(15, 25))
                self.state['energy'] = min(100, self.state['energy'] + random.randint(3, 7))
            self.logger.info(f"{self.name} used {item_type} at {self.position}")
        else:
            self.logger.warning(f"No {item_type} in inventory")

    def craft(self, world: 'World', item_type: str, recipe: Dict[str, int]):
        """
        Crafts an item of the specified type using the provided recipe.

        Args:
            world (World): The simulation environment.
            item_type (str): The type of item to craft.
            recipe (Dict[str, int]): The recipe for crafting the item.
        """
        if all(self.inventory.get(item, 0) >= quantity for item, quantity in recipe.items()):
            for item, quantity in recipe.items():
                self.inventory[item] -= quantity
            self.inventory[item_type] = self.inventory.get(item_type, 0) + 1
            energy_cost = self._calculate_action_energy_cost('craft')
            self.state['energy'] = max(0, self.state['energy'] - energy_cost)
            crafting_skill_increase = self._calculate_skill_increase('crafting')
            self.skills['crafting'] += crafting_skill_increase
            self.logger.info(f"{self.name} crafted {item_type} at {self.position}")
        else:
            self.logger.warning(f"Insufficient resources to craft {item_type}")

    def initiate_communication(self, world: 'World', target_agent: str, message: str):
        """
        Initiates communication with another agent.

        Args:
            world (World): The simulation environment.
            target_agent (str): The name of the target agent.
            message (str): The message to send.
        """
        target_agent_obj = world.get_agent(target_agent)
        if target_agent_obj:
            if self._is_within_communication_range(target_agent_obj):
                target_agent_obj.receive_communication(self.name, message)
                energy_cost = self.config.agents.behavior.energy_consumption_rate * 0.5  # Communication is less costly
                self.state['energy'] = max(0, self.state['energy'] - energy_cost)
                social_skill_increase = self._calculate_skill_increase('social')
                self.skills['social'] += social_skill_increase
                self.logger.info(f"{self.name} sent message '{message}' to {target_agent}")
            else:
                self.logger.warning(f"{target_agent} is out of communication range.")
        else:
            self.logger.warning(f"Target agent {target_agent} not found")

    def receive_communication(self, sender_agent: str, message: str):
        """
        Receives a communication from another agent.

        Args:
            sender_agent (str): The name of the sender agent.
            message (str): The received message.
        """
        self.logger.info(f"{self.name} received message '{message}' from {sender_agent}")
        # Update relationships and knowledge base
        self.relationships[sender_agent] = min(10, self.relationships.get(sender_agent, 0) + random.uniform(0.1, 0.5))
        self.update_knowledge_base({'communication': {'sender': sender_agent, 'message': message}})
        # Store the message for potential rendering
        self.state['last_message'] = message

    def rest(self, world: 'World'):
        """
        Rests and regenerates energy.

        Args:
            world (World): The simulation environment.
        """
        energy_gain = random.randint(8, 15)
        self.state['energy'] = min(100, self.state['energy'] + energy_gain)
        self.state['hunger'] = min(100, self.state['hunger'] + random.randint(1, 3))
        self.state['thirst'] = min(100, self.state['thirst'] + random.randint(1, 3))
        self.logger.info(f"{self.name} rested and regained {energy_gain} energy")

    def attack(self, world: 'World', target_agent: str):
        """
        Attacks another agent.

        Args:
            world (World): The simulation environment.
            target_agent (str): The name of the target agent.
        """
        target_agent_obj = world.get_agent(target_agent)
        if target_agent_obj:
            if self._is_within_attack_range(target_agent_obj):
                damage = random.randint(5, 15) * (self.behavior_traits.combat_skill or 1.0)
                damage = int(damage)
                target_agent_obj.state['health'] -= damage
                energy_cost = self._calculate_action_energy_cost('attack')
                self.state['energy'] = max(0, self.state['energy'] - energy_cost)
                combat_skill_increase = self._calculate_skill_increase('combat')
                self.skills['combat'] += combat_skill_increase
                self.logger.info(f"{self.name} attacked {target_agent} for {damage} damage")
                if target_agent_obj.state['health'] <= 0:
                    self.logger.info(f"{target_agent} has been defeated by {self.name}")
                    world.remove_agent(target_agent_obj)
            else:
                self.logger.warning(f"{target_agent} is out of attack range.")
        else:
            self.logger.warning(f"Target agent {target_agent} not found")

    def _is_within_communication_range(self, other_agent: 'Agent') -> bool:
        """
        Checks if another agent is within communication range.

        Args:
            other_agent (Agent): The other agent to check.

        Returns:
            bool: True if within range, False otherwise.
        """
        distance = self._calculate_distance(self.position, other_agent.position)
        return distance <= self.communication_range

    def _is_within_attack_range(self, other_agent: 'Agent') -> bool:
        """
        Checks if another agent is within attack range.

        Args:
            other_agent (Agent): The other agent to check.

        Returns:
            bool: True if within range, False otherwise.
        """
        distance = self._calculate_distance(self.position, other_agent.position)
        return distance <= 1  # Attack range is adjacent cells

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculates Euclidean distance between two positions.

        Args:
            pos1 (Tuple[int, int]): First position.
            pos2 (Tuple[int, int]): Second position.

        Returns:
            float: Distance between the positions.
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if self.config.environment.grid.wrap_around:
            dx = min(dx, self.config.environment.grid.width - dx)
            dy = min(dy, self.config.environment.grid.height - dy)

        return math.sqrt(dx ** 2 + dy ** 2)

    def update_knowledge_base(self, new_memory: Dict[str, Any]):
        """
        Updates the agent's knowledge base with new information.

        Args:
            new_memory (Dict[str, Any]): The new memory to add to the knowledge base.
        """
        self.knowledge_base.append(new_memory)
        max_memory_size = 100
        if len(self.knowledge_base) > max_memory_size:
            self.knowledge_base = self.knowledge_base[-max_memory_size:]  # Keep the latest memories
        self.logger.debug(f"{self.name} updated knowledge base.")

    def update_state(self):
        """
        Updates the agent's state based on their current situation.
        """
        # Update hunger and thirst
        self.state['hunger'] = min(100, self.state['hunger'] + random.randint(1, 2))
        self.state['thirst'] = min(100, self.state['thirst'] + random.randint(1, 2))
        self.state['energy'] = max(0, self.state['energy'] - random.randint(1, 2))

        # Apply health effects
        if self.state['hunger'] > 80 or self.state['thirst'] > 80:
            health_loss = random.randint(1, 3)
            self.state['health'] = max(0, self.state['health'] - health_loss)
            if 'Weakness' not in self.status_effects:
                self.status_effects.append('Weakness')
        elif self.state['hunger'] < 30 and self.state['thirst'] < 30 and self.state['energy'] > 50:
            health_gain = random.randint(1, 2)
            self.state['health'] = min(100, self.state['health'] + health_gain)
            if 'Weakness' in self.status_effects:
                self.status_effects.remove('Weakness')

        # Update mood based on state
        self.state['mood'] = self._calculate_mood()

        # Experience and leveling
        self.state['experience'] += random.randint(1, 3)
        if self.state['experience'] >= 100 * self.state['level']:
            self.level_up()

    def _calculate_mood(self) -> str:
        """
        Calculates the agent's mood based on their current state.

        Returns:
            str: The agent's mood.
        """
        health_factor = self.state['health'] / 100
        energy_factor = self.state['energy'] / 100
        hunger_factor = (100 - self.state['hunger']) / 100
        thirst_factor = (100 - self.state['thirst']) / 100

        mood_score = (health_factor + energy_factor + hunger_factor + thirst_factor) / 4

        if mood_score > 0.8:
            return 'ecstatic'
        elif mood_score > 0.6:
            return 'happy'
        elif mood_score > 0.4:
            return 'neutral'
        elif mood_score > 0.2:
            return 'sad'
        else:
            return 'depressed'

    def level_up(self):
        """
        Levels up the agent, increasing their level and improving their skills.
        """
        self.state['level'] += 1
        self.state['experience'] = 0
        self.state['health'] = min(100, self.state['health'] + random.randint(5, 10))
        self.state['energy'] = min(100, self.state['energy'] + random.randint(5, 10))
        self.skills['gathering'] += random.uniform(0.05, 0.1)
        self.skills['crafting'] += random.uniform(0.05, 0.1)
        self.skills['combat'] += random.uniform(0.05, 0.1)
        self.skills['social'] += random.uniform(0.05, 0.1)
        self.logger.info(f"{self.name} leveled up to level {self.state['level']}")

    def __str__(self):
        return f"Agent {self.name} at {self.position} with state {self.state}"

    def __repr__(self):
        return self.__str__()

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
import json
import re
import math

if TYPE_CHECKING:
    from environment.world import World
    from agents.agent import Agent

class Agent:
    """
    Represents a supreme agent in the little-matrix simulation.

    Attributes:
        name (str): The unique name of the agent.
        llm_client (LLMClient): Instance of the LLM client for advanced decision-making.
        state (Dict[str, Any]): Current state of the agent, including health, energy, mood, and more.
        knowledge_base (List[Dict[str, Any]]): Memory or knowledge accumulated by the agent.
        position (Tuple[int, int]): Coordinates of the agent in the world.
        inventory (Dict[str, int]): Items and resources the agent has collected.
        perception_radius (int): The radius within which the agent can perceive the environment.
        communication_range (int): The range within which the agent can communicate with others.
        logger (logging.Logger): Logger for the agent.
        skills (Dict[str, float]): Various skills and their proficiency levels.
        goals (List[str]): Current goals or objectives of the agent.
        relationships (Dict[str, float]): Relationships with other agents.
    """

    def __init__(self, name: str, llm_client: LLMClient, position: Optional[Tuple[int, int]] = None):
        """
        Initializes a new Agent instance with randomized attributes.

        Args:
            name (str): The unique name of the agent.
            llm_client (LLMClient): The LLM client used for advanced decision-making.
            position (Tuple[int, int], optional): The initial position of the agent in the world.
        """
        self.name = name
        self.llm_client = llm_client
        self.state: Dict[str, Any] = {
            'health': random.randint(70, 100),
            'energy': random.randint(50, 100),
            'mood': random.choice(['excited', 'curious', 'cautious', 'determined']),
            'hunger': random.randint(0, 30),
            'thirst': random.randint(0, 30),
            'experience': 0,
            'level': 1
        }
        self.knowledge_base: List[Dict[str, Any]] = []
        self.position = position if position is not None else (random.randint(0, 49), random.randint(0, 49))
        self.inventory: Dict[str, int] = {'food': random.randint(0, 3), 'water': random.randint(0, 3)}
        self.perception_radius = random.randint(3, 7)
        self.communication_range = random.randint(8, 12)
        self.logger = logging.getLogger(f"Agent:{self.name}")
        self.skills: Dict[str, float] = {
            'gathering': random.uniform(0.5, 1.5),
            'crafting': random.uniform(0.5, 1.5),
            'combat': random.uniform(0.5, 1.5),
            'social': random.uniform(0.5, 1.5)
        }
        self.goals: List[str] = random.sample(['survive', 'explore', 'socialize', 'learn', 'dominate'], k=2)
        self.relationships: Dict[str, float] = {}
        self.logger.info(f"Agent '{self.name}' initialized at position {self.position}.")

    def perceive(self, world: 'World'):
        """
        Perceives the environment within the agent's perception radius and updates internal state.

        Args:
            world (World): The simulation environment.
        """
        try:
            perceived_agents = [
                agent for agent in world.get_agents_within_radius(self.position, self.perception_radius)
                if agent.name != self.name
            ]
            perceived_objects = world.get_objects_within_radius(self.position, self.perception_radius)

            self.state['perceived_agents'] = [agent.name for agent in perceived_agents]
            self.state['perceived_objects'] = [
                {'type': obj.__class__.__name__, 'position': obj.position}
                for obj in perceived_objects
            ]

            # Update knowledge base with new perceptions
            self.update_knowledge_base({
                'timestamp': world.current_time,
                'perceived_agents': self.state['perceived_agents'],
                'perceived_objects': self.state['perceived_objects']
            })

            self.logger.debug(f"{self.name} perceived {len(perceived_agents)} agents and {len(perceived_objects)} objects.")
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
                'perceived_objects': json.dumps(self.state.get('perceived_objects', []))
            }

            prompt = self.llm_client.prompt_manager.get_prompt('agent_decision', **prompt_vars)
            response = self.llm_client.generate_response(prompt)
            
            # Parse the LLM response
            action, details = self._parse_llm_response(response)
            
            self.state['action'] = action
            self.state['action_details'] = details
            
            self.logger.info(f"{self.name} decided to {self.state['action']} with details: {self.state['action_details']}")
        except Exception as e:
            self.logger.error(f"Decision-making failed: {e}. Defaulting to 'rest'.")
            self.state['action'] = 'rest'
            self.state['action_details'] = {}

    def _parse_llm_response(self, response: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Parses the LLM response to extract action and details.

        Args:
            response (List[Dict[str, str]]): The raw response from the LLM.

        Returns:
            Tuple[str, Dict[str, Any]]: The parsed action and details.
        """
        if not response or not isinstance(response, list) or not response[0].get('generated_text'):
            return 'rest', {}

        generated_text = response[0]['generated_text']

        action_match = re.search(r"'(move|collect|communicate|rest|use|craft|attack)'", generated_text)
        action = action_match.group(1) if action_match else 'rest'

        details = {}
        if action == 'move':
            direction_match = re.search(r"direction:\s*'(\w+)'", generated_text)
            if direction_match:
                details['direction'] = direction_match.group(1)
        elif action == 'collect':
            object_match = re.search(r"object_type:\s*'(\w+)'", generated_text)
            if object_match:
                details['object_type'] = object_match.group(1)
        elif action == 'communicate':
            target_match = re.search(r"target_agent:\s*'(\w+)'", generated_text)
            message_match = re.search(r"message:\s*'([^']*)'", generated_text)
            if target_match and message_match:
                details['target_agent'] = target_match.group(1)
                details['message'] = message_match.group(1)
        elif action == 'use':
            item_match = re.search(r"item_type:\s*'(\w+)'", generated_text)
            if item_match:
                details['item_type'] = item_match.group(1)
        elif action == 'craft':
            item_match = re.search(r"item_type:\s*'(\w+)'", generated_text)
            recipe_match = re.search(r"recipe:\s*(\{[^}]+\})", generated_text)
            if item_match and recipe_match:
                details['item_type'] = item_match.group(1)
                details['recipe'] = json.loads(recipe_match.group(1).replace("'", '"'))
        elif action == 'attack':
            target_match = re.search(r"target_agent:\s*'(\w+)'", generated_text)
            if target_match:
                details['target_agent'] = target_match.group(1)

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
            'attack': self.attack
        }

        try:
            if action in action_map:
                action_map[action](world, **details)
            else:
                self.logger.warning(f"{self.name} attempted unknown action: {action}")
        except Exception as e:
            self.logger.error(f"Error performing action {action}: {e}")

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
            'west': (-1, 0)
        }
        if direction not in direction_map:
            self.logger.warning(f"Invalid move direction: {direction}")
            return

        dx, dy = direction_map[direction]
        new_position = (self.position[0] + dx, self.position[1] + dy)

        if world.is_position_valid(new_position):
            self.position = new_position
            self.state['energy'] -= random.randint(1, 3)
            self.state['hunger'] += random.randint(0, 1)
            self.state['thirst'] += random.randint(0, 1)
            self.logger.info(f"{self.name} moved {direction} to {self.position}")
        else:
            self.logger.warning(f"Invalid move to {new_position}")

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
                self.state['energy'] -= random.randint(2, 5)
                self.state['hunger'] += random.randint(0, 2)
                self.state['thirst'] += random.randint(0, 2)
                self.skills['gathering'] += random.uniform(0.01, 0.05)
                self.logger.info(f"{self.name} collected {object_type} at {self.position}")
                return

        self.logger.warning(f"No {object_type} found within perception radius")

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
                self.state['energy'] += random.randint(5, 10)
            elif item_type == 'water':
                self.state['thirst'] = max(0, self.state['thirst'] - random.randint(15, 25))
                self.state['energy'] += random.randint(3, 7)
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
        if all(item in self.inventory and self.inventory[item] >= quantity for item, quantity in recipe.items()):
            for item, quantity in recipe.items():
                self.inventory[item] -= quantity
            self.inventory[item_type] = self.inventory.get(item_type, 0) + 1
            self.state['energy'] -= random.randint(5, 10)
            self.skills['crafting'] += random.uniform(0.02, 0.08)
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
            target_agent_obj.receive_communication(self.name, message)
            self.state['energy'] -= random.randint(1, 3)
            self.skills['social'] += random.uniform(0.01, 0.05)
            self.logger.info(f"{self.name} sent message '{message}' to {target_agent}")
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
            damage = random.randint(5, 15)
            target_agent_obj.state['health'] -= damage
            self.state['energy'] -= random.randint(5, 10)
            self.skills['combat'] += random.uniform(0.02, 0.08)
            self.logger.info(f"{self.name} attacked {target_agent} for {damage} damage")
        else:
            self.logger.warning(f"Target agent {target_agent} not found")

    def update_knowledge_base(self, new_memory: Dict[str, Any]):
        """
        Updates the agent's knowledge base with new information.

        Args:
            new_memory (Dict [str, Any]): The new memory to add to the knowledge base.
        """
        self.knowledge_base.append(new_memory)
        self.logger.debug(f"{self.name} updated knowledge base with {new_memory}")

    def update_state(self):
        """
        Updates the agent's state based on their current situation.
        """
        self.state['hunger'] = min(100, self.state['hunger'] + random.randint(0, 2))
        self.state['thirst'] = min(100, self.state['thirst'] + random.randint(0, 2))
        self.state['energy'] = max(0, self.state['energy'] - random.randint(0, 1))
        
        if self.state['hunger'] > 80 or self.state['thirst'] > 80:
            self.state['health'] = max(0, self.state['health'] - random.randint(1, 3))
        
        if self.state['hunger'] < 30 and self.state['thirst'] < 30 and self.state['energy'] > 50:
            self.state['health'] = min(100, self.state['health'] + random.randint(0, 1))
        
        self.state['mood'] = self._calculate_mood()
        
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
        self.state['health'] += random.randint(5, 10)
        self.state['energy'] += random.randint(5, 10)
        self.skills['gathering'] += random.uniform(0.05, 0.1)
        self.skills['crafting'] += random.uniform(0.05, 0.1)
        self.skills['combat'] += random.uniform(0.05, 0.1)
        self.skills['social'] += random.uniform(0.05, 0.1)
        self.logger.info(f"{self.name} leveled up to level {self.state['level']}")

    def __str__(self):
        return f"Agent {self.name} at {self.position} with state {self.state}"
# world.py

"""
World module for the little-matrix simulation.

This module defines the World class, representing the simulation environment where agents and objects interact.
It provides methods for managing the grid, adding and removing agents and objects, and updating the world's state.

Classes:
    World: Represents the simulation environment.
"""

import logging
import random
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING
from environment.objects import WorldObject

if TYPE_CHECKING:
    from agents.agent import Agent


class World:
    """
    Represents the simulation environment.

    Attributes:
        width (int): The width of the world grid.
        height (int): The height of the world grid.
        agents (Dict[str, Agent]): A dictionary of agents in the world, keyed by their names.
        objects (Dict[Tuple[int, int], List[WorldObject]]): A dictionary mapping positions to lists of objects.
        logger (logging.Logger): Logger for the world.
    """

    def __init__(self, width: int, height: int):
        """
        Initializes the World instance.

        Args:
            width (int): The width of the world grid.
            height (int): The height of the world grid.
        """
        self.width = width
        self.height = height
        self.agents: Dict[str, 'Agent'] = {}  # Keyed by agent name
        self.objects: Dict[Tuple[int, int], List[WorldObject]] = {}  # Keyed by position
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"World initialized with size ({self.width}, {self.height})")

    def add_agent(self, agent: 'Agent', position: Optional[Tuple[int, int]] = None):
        """
        Adds an agent to the world at a specified position.

        Args:
            agent (Agent): The agent to add.
            position (Tuple[int, int], optional): The (x, y) position to place the agent.
                If None, a random empty position is assigned.

        Raises:
            ValueError: If the position is occupied or invalid.
        """
        if position is None:
            position = self.get_random_empty_position()
        if not self.is_position_valid(position):
            raise ValueError(f"Invalid position {position} for agent '{agent.name}'")
        if self.is_position_occupied_by_agent(position):
            raise ValueError(f"Position {position} is already occupied by another agent.")
        agent.position = position
        self.agents[agent.name] = agent
        self.logger.info(f"Agent '{agent.name}' added at position {position}")

    def remove_agent(self, agent: 'Agent'):
        """
        Removes an agent from the world.

        Args:
            agent (Agent): The agent to remove.
        """
        if agent.name in self.agents:
            del self.agents[agent.name]
            self.logger.info(f"Agent '{agent.name}' removed from the world.")

    def add_object(self, obj: WorldObject):
        """
        Adds an object to the world at its specified position.

        Args:
            obj (WorldObject): The object to add.
        """
        if not self.is_position_valid(obj.position):
            raise ValueError(f"Invalid position {obj.position} for object of type '{type(obj).__name__}'")
        self.objects.setdefault(obj.position, []).append(obj)
        self.logger.info(f"Object of type '{type(obj).__name__}' added at position {obj.position}")

    def remove_object(self, obj: WorldObject):
        """
        Removes an object from the world.

        Args:
            obj (WorldObject): The object to remove.
        """
        position = obj.position
        if position in self.objects and obj in self.objects[position]:
            self.objects[position].remove(obj)
            if not self.objects[position]:
                del self.objects[position]
            self.logger.info(f"Object of type '{type(obj).__name__}' removed from position {position}")

    def move_agent(self, agent: 'Agent', new_position: Tuple[int, int]):
        """
        Moves an agent to a new position in the world.

        Args:
            agent (Agent): The agent to move.
            new_position (Tuple[int, int]): The new (x, y) position.

        Raises:
            ValueError: If the new position is invalid or occupied.
        """
        if not self.is_position_valid(new_position):
            raise ValueError(f"Invalid position {new_position} for agent '{agent.name}'")
        if self.is_position_occupied_by_agent(new_position):
            raise ValueError(f"Position {new_position} is already occupied by another agent.")
        old_position = agent.position
        agent.position = new_position
        self.logger.info(f"Agent '{agent.name}' moved from {old_position} to {new_position}")

    def get_agents_within_radius(self, position: Tuple[int, int], radius: int) -> List['Agent']:
        """
        Retrieves a list of agents within a given radius of a position.

        Args:
            position (Tuple[int, int]): The central (x, y) position.
            radius (int): The radius within which to search.

        Returns:
            List[Agent]: A list of agents within the radius.
        """
        nearby_agents = []
        x0, y0 = position
        for agent in self.agents.values():
            x1, y1 = agent.position
            distance = abs(x1 - x0) + abs(y1 - y0)  # Manhattan distance
            if distance <= radius and agent.position != position:
                nearby_agents.append(agent)
        return nearby_agents

    def get_objects_within_radius(self, position: Tuple[int, int], radius: int) -> List[WorldObject]:
        """
        Retrieves objects within a certain radius from a position.

        Args:
            position (Tuple[int, int]): The central position.
            radius (int): The radius within which to search.

        Returns:
            List[WorldObject]: A list of objects within the radius.
        """
        objects_in_radius = []
        x_min = max(0, position[0] - radius)
        x_max = min(self.width - 1, position[0] + radius)
        y_min = max(0, position[1] - radius)
        y_max = min(self.height - 1, position[1] + radius)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                distance = abs(position[0] - x) + abs(position[1] - y)
                if distance <= radius:
                    pos = (x, y)
                    if pos in self.objects:
                        objects_in_radius.extend(self.objects[pos])
        return objects_in_radius

    def get_objects_at_position(self, position: Tuple[int, int]) -> List[WorldObject]:
        """
        Retrieves the list of objects at a given position.

        Args:
            position (Tuple[int, int]): The (x, y) position.

        Returns:
            List[WorldObject]: The list of objects at the position.
        """
        return self.objects.get(position, [])

    def is_position_valid(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is within the bounds of the world.

        Args:
            position (Tuple[int, int]): The (x, y) position to check.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_position_occupied_by_agent(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is occupied by an agent.

        Args:
            position (Tuple[int, int]): The (x, y) position to check.

        Returns:
            bool: True if the position is occupied by an agent, False otherwise.
        """
        for agent in self.agents.values():
            if agent.position == position:
                return True
        return False

    def get_random_empty_position(self) -> Tuple[int, int]:
        """
        Finds a random empty position in the world.

        Returns:
            Tuple[int, int]: An empty (x, y) position.

        Raises:
            ValueError: If no empty positions are available.
        """
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        occupied_positions = {agent.position for agent in self.agents.values()}
        occupied_positions.update(self.objects.keys())
        empty_positions = [pos for pos in all_positions if pos not in occupied_positions]
        if not empty_positions:
            raise ValueError("No empty positions available in the world.")
        position = random.choice(empty_positions)
        self.logger.debug(f"Random empty position found at {position}")
        return position

    def update(self):
        """
        Updates the state of the world.

        This method handles environmental dynamics, such as object updates, hazards, resource regeneration, etc.
        """
        # Update objects if necessary
        for position, objects in list(self.objects.items()):
            for obj in objects[:]:
                obj.update(self)
                # If the object is depleted or should be removed
                if obj.should_be_removed():
                    self.remove_object(obj)
                    self.logger.info(f"Object '{obj}' at {position} has been removed.")

    def get_adjacent_positions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retrieves a list of valid adjacent positions (up, down, left, right) from a given position.

        Args:
            position (Tuple[int, int]): The (x, y) position.

        Returns:
            List[Tuple[int, int]]: A list of valid adjacent positions.
        """
        x, y = position
        potential_positions = [
            (x, y - 1),  # Up
            (x, y + 1),  # Down
            (x - 1, y),  # Left
            (x + 1, y),  # Right
        ]
        valid_positions = [pos for pos in potential_positions if self.is_position_valid(pos)]
        return valid_positions

    def get_empty_adjacent_positions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retrieves a list of adjacent positions that are valid and not occupied by agents.

        Args:
            position (Tuple[int, int]): The (x, y) position.

        Returns:
            List[Tuple[int, int]]: A list of empty adjacent positions.
        """
        adjacent_positions = self.get_adjacent_positions(position)
        empty_positions = [pos for pos in adjacent_positions if not self.is_position_occupied_by_agent(pos)]
        return empty_positions

    def get_agent_at_position(self, position: Tuple[int, int]) -> Optional['Agent']:
        """
        Retrieves the agent at a specific position, if any.

        Args:
            position (Tuple[int, int]): The (x, y) position.

        Returns:
            Agent or None: The agent at the position, or None if no agent is present.
        """
        for agent in self.agents.values():
            if agent.position == position:
                return agent
        return None

    def move_agent(self, agent: 'Agent', new_position: Tuple[int, int]):
        """
        Moves an agent to a new position in the world.

        Args:
            agent (Agent): The agent to move.
            new_position (Tuple[int, int]): The new (x, y) position.

        Raises:
            ValueError: If the new position is invalid or occupied.
        """
        if not self.is_position_valid(new_position):
            raise ValueError(f"Invalid position {new_position} for agent '{agent.name}'")
        if self.is_position_occupied_by_agent(new_position):
            raise ValueError(f"Position {new_position} is already occupied by another agent.")
        old_position = agent.position
        agent.position = new_position
        self.logger.info(f"Agent '{agent.name}' moved from {old_position} to {new_position}")

    def is_position_occupied(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is occupied by an agent or an impassable object.

        Args:
            position (Tuple[int, int]): The (x, y) position to check.

        Returns:
            bool: True if the position is occupied, False otherwise.
        """
        if self.is_position_occupied_by_agent(position):
            return True
        if position in self.objects:
            for obj in self.objects[position]:
                if obj.is_impassable():
                    return True
        return False

    def get_entities_within_radius(self, position: Tuple[int, int], radius: int) -> Dict[Tuple[int, int], List]:
        """
        Retrieves agents and objects within a certain radius from a position.

        Args:
            position (Tuple[int, int]): The central position.
            radius (int): The radius within which to search.

        Returns:
            Dict[Tuple[int, int], List]: A dictionary mapping positions to lists of entities (agents and objects).
        """
        entities = {}
        x_min = max(0, position[0] - radius)
        x_max = min(self.width - 1, position[0] + radius)
        y_min = max(0, position[1] - radius)
        y_max = min(self.height - 1, position[1] + radius)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                distance = abs(position[0] - x) + abs(position[1] - y)
                if distance <= radius:
                    pos = (x, y)
                    entities_at_pos = []
                    # Check for agents at this position
                    agent = self.get_agent_at_position(pos)
                    if agent:
                        entities_at_pos.append(agent)
                    # Check for objects at this position
                    if pos in self.objects:
                        entities_at_pos.extend(self.objects[pos])
                    if entities_at_pos:
                        entities[pos] = entities_at_pos
        return entities

    def render(self):
        """
        Renders the world's current state to the console.

        For debugging purposes. This method can be expanded or replaced with more advanced rendering techniques.
        """
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for agent in self.agents.values():
            x, y = agent.position
            grid[y][x] = 'A'  # Represent agents with 'A'
        for position, objects in self.objects.items():
            x, y = position
            if objects:
                grid[y][x] = objects[-1].symbol  # Use the symbol of the topmost object
        for row in grid:
            print(' '.join(row))
        print('\n')

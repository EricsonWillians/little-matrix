# src/environment/world.py

"""
World module for the little-matrix simulation.

This module defines the World class, representing the simulation environment where agents and objects interact.
It provides methods for managing the grid, adding and removing agents and objects, and updating the world's state.

Classes:
    World: Represents the simulation environment.
"""

import logging
import random
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING, Any
from ..environment.objects import WorldObject, TerrainFeature
from ..utils.config import Config, TerrainTypeConfig
import numpy as np

if TYPE_CHECKING:
    from agents.agent import Agent

logger = logging.getLogger(__name__)

class World:
    """
    Represents the simulation environment.

    Attributes:
        width (int): The width of the world grid.
        height (int): The height of the world grid.
        agents (Dict[str, Agent]): A dictionary of agents in the world, keyed by their names.
        objects (Dict[Tuple[int, int], List[WorldObject]]): A dictionary mapping positions to lists of objects.
        terrain (np.ndarray): A 2D array representing the terrain types in the world.
        logger (logging.Logger): Logger for the world.
    """

    def __init__(self, config: Config):
        """
        Initializes the World instance using configurations.

        Args:
            config (Config): The configuration object loaded from config.yaml.
        """
        self.config = config
        self.width = config.environment.grid.width
        self.height = config.environment.grid.height
        self.wrap_around = config.environment.grid.wrap_around

        self.agents: Dict[str, 'Agent'] = {}
        self.objects: Dict[Tuple[int, int], List[WorldObject]] = {}
        self.current_time = 0
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"World initialized with size ({self.width}, {self.height})")

        # Initialize terrain
        self.terrain = np.empty((self.height, self.width), dtype=object)
        self._initialize_terrain()

    def _initialize_terrain(self):
        """
        Initializes the terrain grid based on the configuration.
        """
        terrain_types = self.config.environment.grid.terrain.types
        distribution = self.config.environment.grid.terrain.distribution

        # Create a list of terrain types according to their distribution
        terrain_list = []
        for terrain_type in terrain_types:
            terrain_name = terrain_type.name
            count = distribution.get(terrain_name, 0)
            terrain_list.extend([terrain_type] * count)

        # Ensure the terrain_list fills the entire grid
        total_cells = self.width * self.height
        if len(terrain_list) < total_cells:
            default_terrain = next((t for t in terrain_types if t.name == 'default'), terrain_types[0])
            terrain_list.extend([default_terrain] * (total_cells - len(terrain_list)))
        elif len(terrain_list) > total_cells:
            terrain_list = terrain_list[:total_cells]

        random.shuffle(terrain_list)

        # Assign terrain types to grid positions
        index = 0
        for y in range(self.height):
            for x in range(self.width):
                terrain_type = terrain_list[index]
                self.terrain[y, x] = TerrainFeature(position=(x, y), feature_type=terrain_type.name, config=self.config)
                index += 1

    def add_agent(self, agent: 'Agent', position: Optional[Tuple[int, int]] = None) -> None:
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
        if self.is_position_occupied(position):
            raise ValueError(f"Position {position} is already occupied.")
        agent.position = position
        self.agents[agent.name] = agent
        self.logger.info(f"Agent '{agent.name}' added at position {position}")

    def remove_agent(self, agent: 'Agent') -> None:
        """
        Removes an agent from the world.

        Args:
            agent (Agent): The agent to remove.
        """
        if agent.name in self.agents:
            del self.agents[agent.name]
            self.logger.info(f"Agent '{agent.name}' removed from the world.")

    def add_object(self, obj: WorldObject) -> None:
        """
        Adds an object to the world at its specified position.

        Args:
            obj (WorldObject): The object to add.

        Raises:
            ValueError: If the position is invalid.
        """
        if not self.is_position_valid(obj.position):
            raise ValueError(f"Invalid position {obj.position} for object of type '{type(obj).__name__}'")
        self.objects.setdefault(obj.position, []).append(obj)
        self.logger.info(f"Object of type '{type(obj).__name__}' added at position {obj.position}")

    def remove_object(self, obj: WorldObject) -> None:
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

    def move_agent(self, agent: 'Agent', new_position: Tuple[int, int]) -> bool:
        """
        Moves an agent to a new position in the world.

        Args:
            agent (Agent): The agent to move.
            new_position (Tuple[int, int]): The new (x, y) position.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        if self.wrap_around:
            new_position = self._wrap_position(new_position)

        if not self.is_position_valid(new_position) or self.is_position_occupied(new_position):
            self.logger.debug(f"Agent '{agent.name}' cannot move to {new_position}: Invalid or occupied.")
            return False

        terrain_feature = self.terrain[new_position[1], new_position[0]]
        if terrain_feature.is_impassable():
            self.logger.debug(f"Agent '{agent.name}' cannot move to {new_position}: Impassable terrain.")
            return False

        old_position = agent.position
        agent.position = new_position
        self.logger.info(f"Agent '{agent.name}' moved from {old_position} to {new_position}")
        return True

    def get_agents_within_radius(self, position: Tuple[int, int], radius: int) -> List['Agent']:
        """
        Retrieves a list of agents within a given radius of a position.

        Args:
            position (Tuple[int, int]): The central (x, y) position.
            radius (int): The radius within which to search.

        Returns:
            List[Agent]: A list of agents within the radius.
        """
        agents_in_radius = []
        for agent in self.agents.values():
            distance = self.manhattan_distance(position, agent.position)
            if distance <= radius:
                agents_in_radius.append(agent)
        return agents_in_radius

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
        for pos, objects in self.objects.items():
            distance = self.manhattan_distance(position, pos)
            if distance <= radius:
                objects_in_radius.extend(objects)
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

    def is_position_occupied(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is occupied by an agent or an impassable object.

        Args:
            position (Tuple[int, int]): The (x, y) position to check.

        Returns:
            bool: True if the position is occupied, False otherwise.
        """
        if any(agent.position == position for agent in self.agents.values()):
            return True
        if any(obj.is_impassable() for obj in self.objects.get(position, [])):
            return True
        
        # Access the 'impassable' attribute directly
        terrain_feature = self.terrain[position[1], position[0]]
        if terrain_feature.impassable:
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
        empty_positions = [(x, y) for x in range(self.width) for y in range(self.height)
                           if not self.is_position_occupied((x, y))]
        if not empty_positions:
            raise ValueError("No empty positions available in the world.")
        return random.choice(empty_positions)

    def update(self) -> None:
        """
        Updates the state of the world.

        This method handles environmental dynamics, such as object updates, hazards, resource regeneration, etc.
        """
        self.current_time += 1

        # Update objects
        for position, objects in list(self.objects.items()):
            for obj in objects[:]:
                obj.update(self, self.config)
                if obj.should_be_removed():
                    self.remove_object(obj)
                    self.logger.info(f"Object '{obj}' at {position} has been removed.")

        # Update terrain if necessary (e.g., weather effects)
        # Implement terrain updates based on config settings if required

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
        adjacent_positions = []
        for pos in potential_positions:
            if self.wrap_around:
                pos = self._wrap_position(pos)
            if self.is_position_valid(pos):
                adjacent_positions.append(pos)
        return adjacent_positions

    def get_empty_adjacent_positions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retrieves a list of adjacent positions that are valid and not occupied by agents or impassable terrain.

        Args:
            position (Tuple[int, int]): The (x, y) position.

        Returns:
            List[Tuple[int, int]]: A list of empty adjacent positions.
        """
        return [pos for pos in self.get_adjacent_positions(position) if not self.is_position_occupied(pos)]

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

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two positions.

        Args:
            pos1 (Tuple[int, int]): The first position.
            pos2 (Tuple[int, int]): The second position.

        Returns:
            int: The Manhattan distance between the positions.
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if self.wrap_around:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)

        return dx + dy

    def _wrap_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Wraps the position around the world boundaries if wrap-around is enabled.

        Args:
            position (Tuple[int, int]): The position to wrap.

        Returns:
            Tuple[int, int]: The wrapped position.
        """
        x, y = position
        x %= self.width
        y %= self.height
        return x, y

    def render(self) -> None:
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
        # Add terrain symbols
        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] == '.':
                    terrain_feature = self.terrain[y, x]
                    grid[y][x] = terrain_feature.symbol
        for row in grid:
            print(' '.join(row))
        print('\n')

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
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING, Self
from ..environment.objects import WorldObject, TerrainFeature
from ..utils.config import Config
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
            KeyError: If an agent with the same name already exists.
        """
        if agent.name in self.agents:
            error_msg = f"Failed to add agent '{agent.name}': An agent with this name already exists."
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        if position is None:
            position = self.get_random_empty_position()
            self.logger.debug(
                f"No position provided for agent '{agent.name}'. Assigned random position {position}."
            )

        if not self.is_position_valid(position):
            error_msg = f"Failed to add agent '{agent.name}': Position {position} is invalid."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.is_position_occupied(position):
            error_msg = f"Failed to add agent '{agent.name}': Position {position} is already occupied."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        agent.position = position
        self.agents[agent.name] = agent
        self.logger.info(
            f"Agent '{agent.name}' successfully added at position {position}."
        )

    def remove_agent(self, agent: 'Agent') -> None:
        """
        Removes an agent from the world.

        Args:
            agent (Agent): The agent to remove.

        Raises:
            KeyError: If the agent is not found in the world.
        """
        if agent.name not in self.agents:
            error_msg = f"Failed to remove agent '{agent.name}': Agent not found in the world."
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        removed_position = self.agents[agent.name].position
        del self.agents[agent.name]
        self.logger.info(
            f"Agent '{agent.name}' successfully removed from position {removed_position}."
        )

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

    def remove_object(self, obj: WorldObject) -> bool:
        """
        Removes an object from the world.

        Args:
            obj (WorldObject): The object to remove.

        Returns:
            bool: True if the object was removed, False otherwise.
        """
        position = obj.position
        if position in self.objects:
            object_list = self.objects[position]
            if obj in object_list:
                object_list.remove(obj)
                if not object_list:
                    del self.objects[position]
                self.logger.info(
                    f"Object of type '{getattr(obj, 'type', type(obj).__name__)}' "
                    f"removed from position {position}"
                )
                return True
        return False

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
        if terrain_feature.is_impassable:
            self.logger.debug(f"Agent '{agent.name}' cannot move to {new_position}: Impassable terrain.")
            return False

        old_position = agent.position
        agent.position = new_position
        self.logger.info(f"Agent '{agent.name}' moved from {old_position} to {new_position}")
        return True

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
        if any(obj.is_impassable for obj in self.objects.get(position, [])):
            return True

        terrain_feature = self.terrain[position[1], position[0]]
        if terrain_feature.is_impassable:
            return True

        return False

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
        Retrieves objects within a radius.

        Args:
            position (Tuple[int, int]): The center position.
            radius (int): The search radius.

        Returns:
            List[WorldObject]: Objects within the radius.
        """
        objects_in_radius = []
        for pos, objects in self.objects.items():
            if self.manhattan_distance(position, pos) <= radius:
                objects_in_radius.extend(objects)
        return objects_in_radius

    def get_objects_at_position(self, position: Tuple[int, int]) -> List[WorldObject]:
        """
        Retrieves all objects at a specific position, including terrain features.

        Args:
            position (Tuple[int, int]): Position to check.

        Returns:
            List[WorldObject]: List of objects at the position.
        """
        # Get regular objects
        objects = self.objects.get(position, []).copy()

        # Get terrain at position if within bounds
        if 0 <= position[1] < self.height and 0 <= position[0] < self.width:
            terrain = self.terrain[position[1], position[0]]
            if terrain:
                objects.append(terrain)

        return objects

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

    def is_resource_present(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a resource exists at the given position.

        Args:
            position (Tuple[int, int]): Position to check.

        Returns:
            bool: True if resource present.
        """
        objects = self.get_objects_at_position(position)
        return any(
            getattr(obj, 'type', obj.__class__.__name__.lower()) in ['food', 'water']
            for obj in objects
        )

    def get_terrain_cost(self, position: Tuple[int, int]) -> float:
        """
        Get the movement cost of the terrain at the given position.

        Args:
            position: The position as a tuple (x, y).

        Returns:
            A float representing the terrain cost; default is 1.0.
        """
        # If you have different terrain types, adjust the cost accordingly.
        # For now, return 1.0 for all positions.
        return 1.0

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

        This method handles environmental dynamics, such as object updates, hazards,
        resource regeneration, etc.
        """
        self.current_time += 1

        # Update only WorldObjects, not terrain
        for position, objects in list(self.objects.items()):
            for obj in objects[:]:
                if hasattr(obj, 'update'):  # Only update if object has update method
                    obj.update(self, self.config)
                    if hasattr(obj, 'should_be_removed') and obj.should_be_removed():
                        self.remove_object(obj)
                        self.logger.info(f"Object '{obj}' at {position} has been removed.")

        # Handle resource regeneration if configured
        if hasattr(self.config.environment, 'resource'):
            resource_config = self.config.environment.resource
            if hasattr(resource_config, 'spawn_rate') and random.random() < resource_config.spawn_rate:
                self._spawn_new_resource()

    def _spawn_new_resource(self) -> None:
        """Spawn new resources in the world based on configuration."""
        if not hasattr(self.config.environment, 'resource'):
            return

        resource_config = self.config.environment.resource

        # Check if under resource limit
        current_resources = sum(
            1 for objects in self.objects.values()
            for obj in objects
            if hasattr(obj, 'type') and obj.type in ['food', 'water']
        )

        if current_resources >= resource_config.max_resources:
            return

        # Get random resource type
        if hasattr(resource_config, 'types'):
            resource_type = random.choice(resource_config.types)

            # Find valid spawn position
            valid_positions = []
            for y in range(self.height):
                for x in range(self.width):
                    pos = (x, y)
                    if (not self.is_position_occupied(pos) and
                        self.terrain[y, x].feature_type in resource_type.spawn_on):
                        valid_positions.append(pos)

            if valid_positions:
                spawn_pos = random.choice(valid_positions)
                quantity = random.randint(
                    resource_type.quantity_range[0],
                    resource_type.quantity_range[1]
                )

                new_resource = WorldObject(
                    position=spawn_pos,
                    object_type=resource_type.name,
                    config=self.config,
                    quantity=quantity
                )

                self.add_object(new_resource)
                self.logger.info(
                    f"Spawned {resource_type.name} resource at {spawn_pos}"
                )

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

    def save_world_state(self, world: Self):
        """Saves the current state of the world."""
        try:
            world_data = self._serialize_world(world)
            # Save `world_data` to a file or database
            # Here, we'll save it as a JSON file
            with open('world_state.json', 'w') as f:
                json.dump(world_data, f)
            self.logger.info("World state saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save world state: {e}")

    def _serialize_world(self, world: Self) -> dict:
        """Serializes the world object to a dictionary."""
        return {
            'current_time': world.current_time,
            'objects': self._serialize_objects(world.objects),
            # Add other world attributes if necessary
        }

    def _serialize_objects(self, objects) -> dict:
        """Serializes world objects."""
        serialized_objects = {}
        for position, obj_list in objects.items():
            serialized_objects[str(position)] = [obj.to_dict() for obj in obj_list]
        return serialized_objects

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


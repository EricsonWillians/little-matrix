# objects.py

"""
Objects module for the little-matrix simulation.

This module defines the `WorldObject` base class and its subclasses representing different types of objects
within the simulation world, such as obstacles, resources, hazards, collectibles, tools, and terrain features.
Each object interacts differently with agents and the environment, contributing to the richness of the simulation.

Classes:
    WorldObject: Abstract base class for all objects in the simulation world.
    Obstacle: Represents an impassable obstacle within the simulation world.
    Resource: Represents a collectible resource within the simulation world.
    Hazard: Represents a hazardous area that can harm agents.
    Collectible: Represents items that agents can collect.
    Tool: Represents tools that agents can use to perform actions.
    TerrainFeature: Represents static features of the terrain that may affect agent movement or perception.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging

class WorldObject(ABC):
    """
    Abstract base class for all objects within the simulation world.

    Attributes:
        position (Tuple[int, int]): The (x, y) position of the object in the world grid.
        symbol (str): A single character representing the object visually.
        logger (logging.Logger): Logger for the object.
    """

    def __init__(self, position: Tuple[int, int], symbol: str):
        """
        Initializes the WorldObject.

        Args:
            position (Tuple[int, int]): The (x, y) position of the object.
            symbol (str): A single character representing the object visually.
        """
        self.position = position
        self.symbol = symbol
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} initialized at position {self.position}.")

    @abstractmethod
    def interact(self, agent: 'Agent'):
        """
        Defines how an agent interacts with the object.

        Args:
            agent (Agent): The agent interacting with the object.
        """
        pass

    def update(self, world: 'World'):
        """
        Updates the state of the object. Called each timestep.

        Args:
            world (World): The simulation world.
        """
        # Default implementation does nothing.
        pass

    def should_be_removed(self) -> bool:
        """
        Determines whether the object should be removed from the world.

        Returns:
            bool: True if the object should be removed, False otherwise.
        """
        return False

    def is_impassable(self) -> bool:
        """
        Determines whether the object is impassable (blocks agent movement).

        Returns:
            bool: True if the object is impassable, False otherwise.
        """
        return False  # Default to passable


class Obstacle(WorldObject):
    """
    Represents an impassable obstacle within the simulation world.

    Obstacles prevent agents from moving into their position.
    """

    def __init__(self, position: Tuple[int, int]):
        """
        Initializes the Obstacle.

        Args:
            position (Tuple[int, int]): The (x, y) position of the obstacle.
        """
        super().__init__(position, symbol='X')

    def interact(self, agent: 'Agent'):
        """
        Interaction with an obstacle prevents the agent from moving into the obstacle's position.

        Args:
            agent (Agent): The agent attempting to interact with the obstacle.
        """
        self.logger.debug(f"Agent '{agent.name}' tried to interact with an Obstacle at {self.position}.")

    def is_impassable(self) -> bool:
        """
        Obstacles are always impassable.

        Returns:
            bool: True, as obstacles block movement.
        """
        return True


class Resource(WorldObject):
    """
    Represents a collectible resource within the simulation world.

    Attributes:
        quantity (int): The amount of resource available.
        resource_type (str): The type of resource (e.g., 'energy', 'material').
    """

    def __init__(self, position: Tuple[int, int], quantity: int, resource_type: str):
        """
        Initializes the Resource.

        Args:
            position (Tuple[int, int]): The (x, y) position of the resource.
            quantity (int): The initial quantity of the resource.
            resource_type (str): The type of resource.
        """
        symbol_map = {
            'energy': 'E',
            'material': 'M',
            # Add more resource types and their symbols as needed.
        }
        symbol = symbol_map.get(resource_type, '?')
        super().__init__(position, symbol=symbol)
        self.quantity = quantity
        self.resource_type = resource_type

    def interact(self, agent: 'Agent'):
        """
        Allows an agent to collect the resource.

        Args:
            agent (Agent): The agent interacting with the resource.
        """
        if self.quantity > 0:
            collected_amount = agent.collect_resource(self.resource_type, self.quantity)
            self.quantity -= collected_amount
            self.logger.debug(f"Agent '{agent.name}' collected {collected_amount} of '{self.resource_type}' from Resource at {self.position}.")

    def should_be_removed(self) -> bool:
        """
        Determines whether the resource is depleted and should be removed.

        Returns:
            bool: True if the resource quantity is zero, False otherwise.
        """
        return self.quantity <= 0


class Hazard(WorldObject):
    """
    Represents a hazardous area within the simulation world.

    Attributes:
        damage (int): The amount of damage inflicted on an agent.
    """

    def __init__(self, position: Tuple[int, int], damage: int):
        """
        Initializes the Hazard.

        Args:
            position (Tuple[int, int]): The (x, y) position of the hazard.
            damage (int): The damage inflicted on agents.
        """
        super().__init__(position, symbol='H')
        self.damage = damage

    def interact(self, agent: 'Agent'):
        """
        Inflicts damage on the agent.

        Args:
            agent (Agent): The agent interacting with the hazard.
        """
        agent.take_damage(self.damage)
        self.logger.debug(f"Agent '{agent.name}' took {self.damage} damage from Hazard at {self.position}.")

    def is_impassable(self) -> bool:
        """
        Hazards are typically passable but dangerous.

        Returns:
            bool: False, as agents can move through hazards (at their own risk).
        """
        return False


class Collectible(WorldObject):
    """
    Represents an item that agents can collect.

    Attributes:
        item_type (str): The type of item.
        value (int): The value or effect of the item.
    """

    def __init__(self, position: Tuple[int, int], item_type: str, value: int):
        """
        Initializes the Collectible.

        Args:
            position (Tuple[int, int]): The (x, y) position of the collectible.
            item_type (str): The type of the item.
            value (int): The value or effect of the item.
        """
        symbol_map = {
            'coin': 'C',
            'gem': 'G',
            # Add more item types and their symbols as needed.
        }
        symbol = symbol_map.get(item_type, '?')
        super().__init__(position, symbol=symbol)
        self.item_type = item_type
        self.value = value

    def interact(self, agent: 'Agent'):
        """
         Allows the agent to collect the item.

        Args:
            agent (Agent): The agent interacting with the collectible.
        """
        agent.collect_item(self.item_type, self.value)
        self.logger.debug(f"Agent '{agent.name}' collected '{self.item_type}' worth {self.value} at {self.position}.")

    def should_be_removed(self) -> bool:
        """
        Collectibles are removed after being collected.

        Returns:
            bool: True, since collectibles are single-use.
        """
        return True


class Tool(WorldObject):
    """
    Represents a tool that agents can pick up and use.

    Attributes:
        tool_type (str): The type of tool.
        durability (int): The remaining durability of the tool.
    """

    def __init__(self, position: Tuple[int, int], tool_type: str, durability: int):
        """
        Initializes the Tool.

        Args:
            position (Tuple[int, int]): The (x, y) position of the tool.
            tool_type (str): The type of the tool.
            durability (int): The initial durability of the tool.
        """
        symbol_map = {
            'pickaxe': 'P',
            'axe': 'A',
            # Add more tool types and their symbols as needed.
        }
        symbol = symbol_map.get(tool_type, '?')
        super().__init__(position, symbol=symbol)
        self.tool_type = tool_type
        self.durability = durability

    def interact(self, agent: 'Agent'):
        """
        Allows the agent to pick up the tool.

        Args:
            agent (Agent): The agent interacting with the tool.
        """
        agent.pick_up_tool(self.tool_type, self.durability)
        self.logger.debug(f"Agent '{agent.name}' picked up tool '{self.tool_type}' with durability {self.durability} at {self.position}.")

    def should_be_removed(self) -> bool:
        """
        Tools are removed from the world once picked up by an agent.

        Returns:
            bool: True, since the tool is now in the agent's possession.
        """
        return True


class TerrainFeature(WorldObject):
    """
    Represents a static feature of the terrain that may affect agent movement or perception.

    Attributes:
        feature_type (str): The type of terrain feature.
    """

    def __init__(self, position: Tuple[int, int], feature_type: str):
        """
        Initializes the TerrainFeature.

        Args:
            position (Tuple[int, int]): The (x, y) position of the feature.
            feature_type (str): The type of terrain feature.
        """
        symbol_map = {
            'water': 'W',
            'forest': 'F',
            # Add more feature types and their symbols as needed.
        }
        symbol = symbol_map.get(feature_type, '?')
        super().__init__(position, symbol=symbol)
        self.feature_type = feature_type

    def interact(self, agent: 'Agent'):
        """
        Modifies agent's movement or perception based on the terrain.

        Args:
            agent (Agent): The agent interacting with the terrain feature.
        """
        if self.feature_type == 'water':
            agent.modify_speed(0.5)  # Agent moves slower in water.
            self.logger.debug(f"Agent '{agent.name}' speed reduced due to Water at {self.position}.")
        elif self.feature_type == 'forest':
            agent.modify_visibility(-1)  # Agent's visibility reduced in forest.
            self.logger.debug(f"Agent '{agent.name}' visibility reduced due to Forest at {self.position}.")

    def is_impassable(self) -> bool:
        """
        Terrain features can be impassable depending on their type.

        Returns:
            bool: True if the feature is impassable, False otherwise.
        """
        return self.feature_type in ['mountain', 'water']  # Example impassable terrain types
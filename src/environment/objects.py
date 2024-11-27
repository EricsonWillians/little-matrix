# src/environment/objects.py

"""
Objects module for the little-matrix simulation.

This module defines the `WorldObject` base class and its subclasses representing different types of objects
within the simulation world, such as obstacles, resources, hazards, collectibles, tools, and terrain features.
Each object interacts differently with agents and the environment, contributing to the richness of the simulation.

Classes:
    WorldObject: Abstract base class for all objects in the simulation world.
    Obstacle: Represents an is_impassable obstacle within the simulation world.
    Resource: Represents a collectible resource within the simulation world.
    Hazard: Represents a hazardous area that can harm agents.
    Collectible: Represents items that agents can collect.
    Tool: Represents tools that agents can use to perform actions.
    TerrainFeature: Represents static features of the terrain that may affect agent movement or perception.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import logging
from ..utils.config import Config

if 'Agent' in globals():
    from agents.agent import Agent
else:
    Agent = Any  # Fallback for type hints if Agent is not imported

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
    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Defines how an agent interacts with the object.

        Args:
            agent (Agent): The agent interacting with the object.
            world (World): The simulation world.
            config (Config): Configuration settings.
        """
        pass

    def update(self, world: 'World', config: Config):
        """
        Updates the state of the object. Called each timestep.

        Args:
            world (World): The simulation world.
            config (Config): Configuration settings.
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
        Determines whether the object is is_impassable (blocks agent movement).

        Returns:
            bool: True if the object is is_impassable, False otherwise.
        """
        return False  # Default to passable
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation for serialization.

        Returns:
            Dict[str, Any]: Dictionary containing the object's attributes
        """
        return {
            'type': self.__class__.__name__,
            'position': self.position,
            'symbol': self.symbol
        }


class Obstacle(WorldObject):
    """
    Represents an is_impassable obstacle within the simulation world.

    Obstacles prevent agents from moving into their position.
    """

    def __init__(self, position: Tuple[int, int], symbol: str = 'X'):
        """
        Initializes the Obstacle.

        Args:
            position (Tuple[int, int]): The (x, y) position of the obstacle.
            symbol (str): Symbol representing the obstacle.
        """
        super().__init__(position, symbol=symbol)

    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Interaction with an obstacle prevents the agent from moving into the obstacle's position.

        Args:
            agent (Agent): The agent attempting to interact with the obstacle.
            world (World): The simulation world.
            config (Config): Configuration settings.
        """
        self.logger.debug(f"Agent '{agent.name}' tried to interact with an Obstacle at {self.position}.")

    def is_impassable(self) -> bool:
        """
        Obstacles are always is_impassable.

        Returns:
            bool: True, as obstacles block movement.
        """
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the obstacle to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing the obstacle's attributes
        """
        base_dict = super().to_dict()
        return base_dict


class Resource(WorldObject):
    """
    Represents a collectible resource within the simulation world.

    Attributes:
        quantity (int): The amount of resource available.
        resource_type (str): The type of resource (e.g., 'Food', 'Water', 'Metal').
    """

    def __init__(self, position: Tuple[int, int], quantity: int, resource_type: str, config: Config):
        """
        Initializes the Resource.

        Args:
            position (Tuple[int, int]): The (x, y) position of the resource.
            quantity (int): The initial quantity of the resource.
            resource_type (str): The type of resource.
            config (Config): Configuration settings.
        """
        resource_config = next((res for res in config.environment.resource.types if res.name == resource_type), None)
        symbol = resource_config.symbol if resource_config else '?'
        super().__init__(position, symbol=symbol)
        self.quantity = quantity
        self.resource_type = resource_type

    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Allows an agent to collect the resource.

        Args:
            agent (Agent): The agent interacting with the resource.
            world (World): The simulation world.
            config (Config): Configuration settings.
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the resource to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing the resource's attributes
        """
        base_dict = super().to_dict()
        base_dict.update({
            'quantity': self.quantity,
            'resource_type': self.resource_type
        })
        return base_dict


class Hazard(WorldObject):
    """
    Represents a hazardous area within the simulation world.

    Attributes:
        damage (int): The amount of damage inflicted on an agent.
    """

    def __init__(self, position: Tuple[int, int], damage: int, symbol: str = 'H'):
        """
        Initializes the Hazard.

        Args:
            position (Tuple[int, int]): The (x, y) position of the hazard.
            damage (int): The damage inflicted on agents.
            symbol (str): Symbol representing the hazard.
        """
        super().__init__(position, symbol=symbol)
        self.damage = damage

    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Inflicts damage on the agent.

        Args:
            agent (Agent): The agent interacting with the hazard.
            world (World): The simulation world.
            config (Config): Configuration settings.
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the hazard to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing the hazard's attributes
        """
        base_dict = super().to_dict()
        base_dict.update({
            'damage': self.damage
        })
        return base_dict


class Collectible(WorldObject):
    """
    Represents an item that agents can collect.

    Attributes:
        item_type (str): The type of item.
        value (int): The value or effect of the item.
    """

    def __init__(self, position: Tuple[int, int], item_type: str, value: int, symbol: str = 'I'):
        """
        Initializes the Collectible.

        Args:
            position (Tuple[int, int]): The (x, y) position of the collectible.
            item_type (str): The type of the item.
            value (int): The value or effect of the item.
            symbol (str): Symbol representing the collectible.
        """
        super().__init__(position, symbol=symbol)
        self.item_type = item_type
        self.value = value

    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Allows the agent to collect the item.

        Args:
            agent (Agent): The agent interacting with the collectible.
            world (World): The simulation world.
            config (Config): Configuration settings.
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the tool to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing the tool's attributes
        """
        base_dict = super().to_dict()
        base_dict.update({
            'tool_type': self.tool_type,
            'durability': self.durability
        })
        return base_dict


class Tool(WorldObject):
    """
    Represents a tool that agents can pick up and use.

    Attributes:
        tool_type (str): The type of tool.
        durability (int): The remaining durability of the tool.
    """

    def __init__(self, position: Tuple[int, int], tool_type: str, durability: int, symbol: str = 'T'):
        """
        Initializes the Tool.

        Args:
            position (Tuple[int, int]): The (x, y) position of the tool.
            tool_type (str): The type of the tool.
            durability (int): The initial durability of the tool.
            symbol (str): Symbol representing the tool.
        """
        super().__init__(position, symbol=symbol)
        self.tool_type = tool_type
        self.durability = durability

    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Allows the agent to pick up the tool.

        Args:
            agent (Agent): The agent interacting with the tool.
            world (World): The simulation world.
            config (Config): Configuration settings.
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
        movement_cost (int): The movement cost associated with this terrain.
        is_impassable (bool): Whether the terrain is is_impassable.
    """

    def __init__(self, position: Tuple[int, int], feature_type: str, config: Config):
        """
        Initializes the TerrainFeature.

        Args:
            position (Tuple[int, int]): The (x, y) position of the feature.
            feature_type (str): The type of terrain feature.
            config (Config): Configuration settings.
        """
        terrain_config = next((terrain for terrain in config.environment.grid.terrain.types if terrain.name == feature_type), None)
        symbol = terrain_config.symbol if terrain_config else '?'
        super().__init__(position, symbol=symbol)
        self.feature_type = feature_type
        self.movement_cost = terrain_config.movement_cost if terrain_config else 1
        self.is_impassable = terrain_config.is_impassable if terrain_config else False

    def interact(self, agent: Agent, world: 'World', config: Config):
        """
        Modifies agent's movement or perception based on the terrain.

        Args:
            agent (Agent): The agent interacting with the terrain feature.
            world (World): The simulation world.
            config (Config): Configuration settings.
        """
        if self.is_impassable:
            agent.prevent_movement()
            self.logger.debug(f"Agent '{agent.name}' cannot pass through {self.feature_type} at {self.position}.")
        else:
            agent.adjust_movement_cost(self.movement_cost)
            self.logger.debug(f"Agent '{agent.name}' movement cost adjusted due to {self.feature_type} at {self.position}.")

    def is_impassable(self) -> bool:
        """
        Determines if the terrain feature is is_impassable.

        Returns:
            bool: True if the feature is is_impassable, False otherwise.
        """
        return self.is_impassable
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the terrain feature to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing the terrain feature's attributes
        """
        base_dict = super().to_dict()
        base_dict.update({
            'feature_type': self.feature_type,
            'movement_cost': self.movement_cost,
            'is_impassable': self.is_impassable
        })
        return base_dict

import logging
import random
from src.environment.objects import Resource, Hazard
from src.utils.config import Config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.world import World

logger = logging.getLogger(__name__)

def populate_world(world: "World", config: Config):
    """
    Populates the simulation world with terrain, resources, and hazards according to configuration settings.

    Args:
        world (World): The simulation world instance.
        config (Config): Configuration settings for the simulation.
    """
    logger.info("Commencing world population with terrain, resources, and hazards.")
    initialize_terrain(world, config)
    initialize_resources(world, config)
    initialize_hazards(world, config)
    logger.info("World population completed successfully.")

def initialize_terrain(world: "World", config: Config):
    """
    Initializes the terrain of the world based on distribution settings in the configuration.

    This function creates a randomized distribution of different terrain types across the grid, ensuring
    that the types are assigned proportionally to their specified distribution.

    Args:
        world (World): The simulation world instance.
        config (Config): Configuration settings containing terrain parameters.
    """
    terrain_types = config.environment.grid.terrain.types
    distribution = config.environment.grid.terrain.distribution

    terrain_list = []
    for terrain_type in terrain_types:
        count = distribution.get(terrain_type.name, 0)
        terrain_list.extend([terrain_type] * count)

    total_cells = world.width * world.height
    if len(terrain_list) < total_cells:
        default_terrain = next((t for t in terrain_types if t.name == 'default'), terrain_types[0])
        terrain_list.extend([default_terrain] * (total_cells - len(terrain_list)))
    elif len(terrain_list) > total_cells:
        terrain_list = terrain_list[:total_cells]

    random.shuffle(terrain_list)

    index = 0
    for y in range(world.height):
        for x in range(world.width):
            terrain_type = terrain_list[index]
            world.terrain[y, x] = terrain_type
            logger.debug(f"Assigned terrain '{terrain_type.name}' at position ({x}, {y}).")
            index += 1
    logger.info("Terrain initialization complete.")

def initialize_resources(world, config):
    """
    Initializes resources in the world according to the configuration.

    Args:
        world (World): The simulation world.
        config (Config): The simulation configuration.
    """
    resource_config = config.environment.resource
    max_resources = resource_config.max_resources

    for _ in range(max_resources):
        position = world.get_random_empty_position()
        resource_type = random.choice(resource_config.types)
        resource_quantity = random.randint(*resource_type.quantity_range)
        
        # Pass `config` to the Resource initialization
        resource = Resource(position=position, resource_type=resource_type.name, quantity=resource_quantity, config=config)
        
        world.add_object(resource)
        logging.info(f"Resource of type '{resource_type.name}' added at position {position} with quantity {resource_quantity}")

def initialize_hazards(world: "World", config: Config):
    """
    Places hazards within the world based on hazard configuration settings.

    Hazards add complexity to the world, requiring agents to navigate around or manage the risks associated with 
    these environmental challenges.

    Args:
        world (World): The simulation world instance.
        config (Config): Configuration containing hazard parameters.
    """
    hazard_config = getattr(config.environment, 'hazard', None)
    if not hazard_config:
        logger.info("No hazards configured, skipping hazard initialization.")
        return

    hazard_density = hazard_config.density or 0.05
    max_hazards = int(hazard_density * world.width * world.height)

    for _ in range(max_hazards):
        hazard_type = random.choice(hazard_config.types)
        position = world.get_random_empty_position()
        severity = random.randint(*hazard_type.severity_range)

        if world.terrain[position[1], position[0]].impassable:
            logger.debug(f"Position {position} is impassable, skipping hazard placement.")
            continue

        hazard = Hazard(position=position, hazard_type=hazard_type.name, severity=severity)
        world.add_object(hazard)
        logger.info(f"Placed hazard '{hazard_type.name}' with severity {severity} at position {position}.")

    logger.info(f"{max_hazards} hazards initialized in the world.")

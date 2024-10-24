"""
Main module for the little-matrix simulation.

This module serves as the entry point for the simulation. It initializes the simulation environment,
including the world and agents, and runs the main simulation loop where agents interact within the world.
The simulation is designed to emulate a complex and dynamic environment where multiple agents coexist,
interact, and evolve within a 'little matrix'.

Functions:
    main(): The main function that runs the simulation.
"""

import argparse
import logging
import sys
import os
from agents.agent import Agent
from environment.world import World
from llm.client import LLMClient
from data.storage import StorageManager
from utils.logger import setup_logging
from utils.config import load_config
from renderer import Renderer  # Import the Renderer class
from environment.objects import Obstacle, Resource, Hazard
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def initialize_agents(num_agents: int, world: World, storage_manager: StorageManager, llm_client: LLMClient) -> List[Agent]:
    """
    Initializes agents for the simulation, either by loading existing agents from storage or creating new ones.

    Args:
        num_agents (int): The number of agents to initialize.
        world (World): The simulation world.
        storage_manager (StorageManager): Manages persistence of agent states.
        llm_client (LLMClient): The LLM client for generating agent responses.

    Returns:
        List[Agent]: A list of initialized agents.
    """
    agents = []
    for i in range(num_agents):
        agent_name = f"Agent_{i}"
        agent_data = storage_manager.load_agent_state(agent_name)
        if agent_data:
            agent = Agent(name=agent_name, llm_client=llm_client)
            agent.state = agent_data['state']
            agent.knowledge_base = agent_data['knowledge_base']
            agent.position = agent_data['position']
            logging.info(f"Loaded existing agent '{agent_name}' from storage at position {agent.position}.")
        else:
            # Assign a random empty position to the new agent
            position = world.get_random_empty_position()
            agent = Agent(name=agent_name, llm_client=llm_client, position=position)
            logging.info(f"Created new agent '{agent_name}' at position {position}.")
        agents.append(agent)
        world.add_agent(agent, position=agent.position)
    return agents


def populate_world(world: World):
    """
    Populates the simulation world with objects such as obstacles, resources, and hazards.

    Args:
        world (World): The simulation world.

    Returns:
        None
    """
    obstacles = [
        Obstacle(position=(10, 10)),
        Obstacle(position=(15, 15)),
        Obstacle(position=(20, 20)),
        Obstacle(position=(25, 25)),
        Obstacle(position=(30, 30)),
    ]
    for obstacle in obstacles:
        world.add_object(obstacle)
        logging.debug(f"Obstacle added at position {obstacle.position}.")

    resources = [
        Resource(position=(5, 5), quantity=50, resource_type='energy'),
        Resource(position=(25, 5), quantity=30, resource_type='material'),
        Resource(position=(5, 25), quantity=20, resource_type='energy'),
        Resource(position=(25, 25), quantity=40, resource_type='material'),
    ]
    for resource in resources:
        world.add_object(resource)
        logging.debug(f"Resource '{resource.resource_type}' added at position {resource.position} with quantity {resource.quantity}.")

    hazards = [
        Hazard(position=(35, 35), damage=10),
        Hazard(position=(40, 40), damage=15),
        Hazard(position=(45, 45), damage=20),
    ]
    for hazard in hazards:
        world.add_object(hazard)
        logging.debug(f"Hazard added at position {hazard.position} with damage {hazard.damage}.")


def run_simulation(timesteps: int, agents: List[Agent], world: World, storage_manager: StorageManager, renderer: Renderer = None):
    """
    Runs the main simulation loop for a specified number of timesteps.

    Args:
        timesteps (int): The number of timesteps to run the simulation.
        agents (List[Agent]): The list of agents in the simulation.
        world (World): The simulation world.
        storage_manager (StorageManager): Manages persistence of agent states.
        renderer (Renderer, optional): The renderer for visualizing the simulation.

    Returns:
        None
    """
    try:
        for timestep in range(timesteps):
            logging.info(f"--- Timestep {timestep + 1}/{timesteps} ---")

            if renderer:
                renderer.handle_events()

            # Agents perceive, decide, and act
            for agent in agents:
                agent.perceive(world)
                agent.decide()
                agent.act(world)
                agent.update_state()
                storage_manager.save_agent_state(agent)
                logging.debug(f"Agent '{agent.name}' state saved.")

            world.update()
            logging.debug("World state updated.")

            if renderer:
                renderer.render()

    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user.")
    finally:
        if renderer:
            renderer.close()
        storage_manager.close()
        logging.info("Simulation ended.")


def main():
    """
    Runs the little-matrix simulation.

    This function initializes the logging system, parses command-line arguments, loads configuration settings,
    initializes the storage manager, LLM client, world, and agents, and then enters the main simulation loop.
    """
    setup_logging(log_level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the little-matrix simulation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--timesteps", type=int, default=None, help="Number of timesteps to run the simulation.")
    parser.add_argument("--render", action="store_true", help="Render the world visually using Pygame.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override timesteps if provided as an argument
    timesteps = args.timesteps if args.timesteps is not None else config.get("timesteps", 100)

    # Retrieve the API key and model from environment variables
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL_ID")

    if not api_key or not model:
        logging.error("API_KEY or MODEL_ID not set in the environment variables.")
        sys.exit(1)

    # Initialize components
    storage_manager = StorageManager(db_file=config.get("database", "little_matrix.db"))
    llm_client = LLMClient(api_key=api_key, model=model)
    world_width = config.get("world_width", 50)
    world_height = config.get("world_height", 50)
    world = World(width=world_width, height=world_height)

    logging.info(f"Simulation world initialized with size ({world_width}, {world_height}).")

    # Initialize agents
    num_agents = config.get("num_agents", 10)
    agents = initialize_agents(num_agents, world, storage_manager, llm_client)
    logging.info(f"Initialized {len(agents)} agents in the simulation.")

    # Populate the world with objects
    populate_world(world)

    # Initialize renderer if rendering is enabled
    if args.render:
        try:
            renderer = Renderer(world)
            logging.info("Renderer initialized.")
        except ImportError as e:
            logging.error(f"Failed to initialize renderer: {e}")
            sys.exit(1)
    else:
        renderer = None

    # Run the simulation
    run_simulation(timesteps, agents, world, storage_manager, renderer)


if __name__ == "__main__":
    main()

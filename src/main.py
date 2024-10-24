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
import random
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
    Populates the simulation world procedurally with objects such as obstacles, resources, and hazards.

    Args:
        world (World): The simulation world.

    Returns:
        None
    """
    # Randomize the number of objects for each type
    num_obstacles = random.randint(5, 15)  # Between 5 and 15 obstacles
    num_resources = random.randint(5, 10)  # Between 5 and 10 resources
    num_hazards = random.randint(3, 8)     # Between 3 and 8 hazards

    # Procedurally generate obstacles at random positions
    for _ in range(num_obstacles):
        position = world.get_random_empty_position()
        obstacle = Obstacle(position=position)
        world.add_object(obstacle)
        logging.debug(f"Obstacle added at position {position}.")

    # Procedurally generate resources with random quantities and types
    resource_types = ['energy', 'material', 'food', 'water']
    for _ in range(num_resources):
        position = world.get_random_empty_position()
        quantity = random.randint(10, 100)  # Random quantity between 10 and 100
        resource_type = random.choice(resource_types)  # Randomly choose a resource type
        resource = Resource(position=position, quantity=quantity, resource_type=resource_type)
        world.add_object(resource)
        logging.debug(f"Resource '{resource_type}' added at position {position} with quantity {quantity}.")

    # Procedurally generate hazards with random damage values
    for _ in range(num_hazards):
        position = world.get_random_empty_position()
        damage = random.randint(5, 25)  # Random damage between 5 and 25
        hazard = Hazard(position=position, damage=damage)
        world.add_object(hazard)
        logging.debug(f"Hazard added at position {position} with damage {damage}.")

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
                agent.decide()  # This method updates the agent's state with the decided action
                agent.act(world)  # Perform the action decided
                
                # Update agent's state
                agent.state['energy'] -= 1  # Decrease energy each timestep
                if agent.state['energy'] <= 0:
                    agent.state['health'] -= 1  # Decrease health if energy is depleted
                
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

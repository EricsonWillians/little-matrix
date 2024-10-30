# src/main.py

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
from typing import List, Optional
from src.agents.agent import Agent
from src.environment.world import World
from src.llm.client import LLMClient
from src.data.storage import StorageManager
from src.utils.logger import setup_logging
from src.utils.config import load_config, Config
from src.renderer import Renderer  # Import the Renderer class
from src.environment.objects import Obstacle, Resource, Hazard
from src.environment.world_helpers import populate_world  # Newly integrated function
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.utils.config import AgentTypeBehaviorTraits

def initialize_agents(config: Config, world: World, storage_manager: StorageManager, llm_client: LLMClient) -> List[Agent]:
    """
    Initializes agents in the simulation.

    Args:
        config (Config): Configuration settings.
        world (World): The simulation world.
        storage_manager (StorageManager): Manages persistence of agent states.
        llm_client (LLMClient): The client for interacting with the LLM.

    Returns:
        List[Agent]: A list of initialized agents.
    """
    agents = []
    agent_configs = config.agents
    num_agents = agent_configs.count

    for i in range(num_agents):
        agent_name = f"Agent_{i}"
        agent_data = storage_manager.load_agent_state(agent_name)
        if agent_data:
            behavior_traits = agent_data.get('behavior_traits', {})
            if isinstance(behavior_traits, dict):
                behavior_traits = AgentTypeBehaviorTraits(**behavior_traits)  # Convert dictionary to AgentTypeBehaviorTraits

            agent = Agent(
                name=agent_name,
                llm_client=llm_client,
                config=config,
                position=agent_data['position'],
                state=agent_data['state'],
                agent_type=agent_configs.customization.types[0],
                behavior_traits=behavior_traits
            )
            agent.knowledge_base = agent_data['knowledge_base']
            agent.position = agent_data['position']
            logging.info(f"Loaded existing agent '{agent_name}' from storage at position {agent.position}.")
        else:
            # Assign a random empty position to the new agent
            position = world.get_random_empty_position()
            # Randomly assign an agent type from the configuration
            agent_type_config = random.choice(agent_configs.customization.types)
            behavior_traits = agent_type_config.behavior_traits
            if isinstance(behavior_traits, dict):
                behavior_traits = AgentTypeBehaviorTraits(**behavior_traits)  # Ensure it's the correct type
            
            agent = Agent(
                name=agent_name,
                llm_client=llm_client,
                config=config,
                position=position,
                agent_type=agent_type_config,
                behavior_traits=behavior_traits
            )
            logging.info(f"Created new agent '{agent_name}' of type '{agent_type_config.name}' at position {position}.")
        agents.append(agent)
        world.add_agent(agent, position=agent.position)
    return agents


def run_simulation(config: Config, agents: List[Agent], world: World, storage_manager: StorageManager, renderer: Optional[Renderer] = None):
    """
    Runs the main simulation loop for a specified number of timesteps.

    Args:
        config (Config): Configuration settings.
        agents (List[Agent]): The list of agents in the simulation.
        world (World): The simulation world.
        storage_manager (StorageManager): Manages persistence of agent states.
        renderer (Renderer, optional): The renderer for visualizing the simulation.

    Returns:
        None
    """
    timesteps = config.simulation.timesteps
    try:
        for timestep in range(timesteps):
            logging.info(f"--- Timestep {timestep + 1}/{timesteps} ---")

            if renderer:
                renderer.handle_events()

            for agent in agents:
                agent.perceive(world)
                agent.decide()
                agent.act(world)

                # Update agent's state and check if they are still alive
                if agent.state['energy'] <= 0:
                    agent.state['health'] -= 1  # Decrease health if energy is depleted

                if agent.state['health'] <= 0:
                    logging.info(f"Agent '{agent.name}' has died.")
                    world.remove_agent(agent)
                    agents.remove(agent)
                    continue

                storage_manager.save_agent_state(agent)

            world.update()
            if renderer:
                renderer.render()

            # Save simulation state at intervals
            if config.simulation.save_state_interval and (timestep + 1) % config.simulation.save_state_interval == 0:
                logging.info(f"Simulation state saved at timestep {timestep + 1}.")

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
    parser = argparse.ArgumentParser(description="Run the little-matrix simulation.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--timesteps", type=int, default=None, help="Number of timesteps to run the simulation.")
    parser.add_argument("--render", action="store_true", help="Render the world visually using Pygame.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = parser.parse_args()

    setup_logging(log_level=args.log_level.upper())
    config = load_config(args.config)

    if args.timesteps is not None:
        config.simulation.timesteps = args.timesteps

    api_key, model = config.llm.api_key, config.llm.model
    if not api_key or not model:
        logging.error("API key or model not provided in the configuration.")
        sys.exit(1)

    storage_manager = StorageManager(config=config)
    llm_client = LLMClient(config=config)
    world = World(config=config)

    logging.info(f"Simulation world initialized with size ({world.width}, {world.height}).")

    agents = initialize_agents(config, world, storage_manager, llm_client)
    logging.info(f"Initialized {len(agents)} agents in the simulation.")

    # Seamlessly integrate the world population
    populate_world(world, config)

    renderer = None
    if args.render and config.simulation.render:
        try:
            renderer = Renderer(world, config)
            logging.info("Renderer initialized.")
        except ImportError as e:
            logging.error(f"Failed to initialize renderer: {e}")
            sys.exit(1)

    run_simulation(config, agents, world, storage_manager, renderer)


if __name__ == "__main__":
    main()

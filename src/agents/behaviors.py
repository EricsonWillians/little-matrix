# src/agents/behaviors.py

"""
Behaviors module for the little-matrix simulation.

This module defines behavior patterns and strategies that agents can adopt in the simulation.
Behaviors encapsulate complex actions or sequences of actions, allowing agents to interact
with the environment and other agents in sophisticated ways.

Classes:
    Behavior: Base class for defining behaviors.
    ExploreBehavior: Behavior for exploring the environment.
    RestBehavior: Behavior for resting to regain energy.
    AvoidBehavior: Behavior for avoiding threats or enemies.
    CommunicateBehavior: Behavior for communicating with other agents.
    CustomBehavior: Template for creating custom behaviors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List

from environment.world import World
from agents.agent import Agent
from utils.config import Config

import random

logger = logging.getLogger(__name__)

class Behavior(ABC):
    """
    Abstract base class for all agent behaviors.

    Methods:
        execute(agent, environment, config): Executes the behavior for the given agent.
    """

    @abstractmethod
    def execute(self, agent: Agent, environment: World, config: Config):
        """
        Executes the behavior for the given agent in the environment.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.
            config (Config): Configuration settings.

        Returns:
            None
        """
        pass

class ExploreBehavior(Behavior):
    """
    Behavior for exploring the environment.
    """

    def execute(self, agent: Agent, environment: World, config: Config):
        """
        Moves the agent to explore new areas in the environment.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.
            config (Config): Configuration settings.

        Returns:
            None
        """
        try:
            # Decide direction based on agent's perception and configuration
            directions = ['north', 'south', 'east', 'west']
            direction = random.choice(directions)

            # Modify movement based on agent type and configuration
            speed_modifier = agent.behavior_traits.get('speed_modifier', 1.0)
            energy_cost = config.agents.behavior.energy_consumption_rate * speed_modifier

            agent.move(environment, direction=direction)

            # Perceive the new environment
            agent.perceive(environment)

            # Update knowledge base
            new_info = {
                'position': agent.position,
                'perceived_agents': agent.state.get('perceived_agents', []),
                'perceived_objects': agent.state.get('perceived_objects', [])
            }
            agent.update_knowledge_base(new_info)

            # Adjust energy levels
            agent.state['energy'] -= energy_cost

            # Log the action
            logger.info(f"{agent.name} is exploring towards {direction}.")
        except Exception as e:
            logger.error(f"Error during exploration behavior for {agent.name}: {e}")

class RestBehavior(Behavior):
    """
    Behavior for resting to regain energy.
    """

    def execute(self, agent: Agent, environment: World, config: Config):
        """
        Allows the agent to rest and regain energy.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.
            config (Config): Configuration settings.

        Returns:
            None
        """
        try:
            agent.rest(environment)
            agent.state['mood'] = 'rested'
            # Log the action
            logger.info(f"{agent.name} is resting to regain energy.")
        except Exception as e:
            logger.error(f"Error during rest behavior for {agent.name}: {e}")

class AvoidBehavior(Behavior):
    """
    Behavior for avoiding threats or enemies.
    """

    def execute(self, agent: Agent, environment: World, config: Config):
        """
        Moves the agent away from nearby threats or enemies.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.
            config (Config): Configuration settings.

        Returns:
            None
        """
        try:
            threats = environment.get_threats_near(agent.position)
            if threats:
                closest_threat = min(threats, key=lambda t: environment.distance(agent.position, t.position))
                agent.move_away_from(closest_threat.position, environment)
                agent.state['mood'] = 'alert'

                # Adjust energy levels based on movement
                speed_modifier = agent.behavior_traits.get('speed_modifier', 1.0)
                energy_cost = config.agents.behavior.energy_consumption_rate * speed_modifier
                agent.state['energy'] -= energy_cost

                # Log the action
                logger.info(f"{agent.name} is avoiding a threat at position {closest_threat.position}.")
            else:
                # If no threats are nearby, default to exploring
                ExploreBehavior().execute(agent, environment, config)
        except Exception as e:
            logger.error(f"Error during avoid behavior for {agent.name}: {e}")

class CommunicateBehavior(Behavior):
    """
    Behavior for communicating with nearby agents.
    """

    def execute(self, agent: Agent, environment: World, config: Config):
        """
        Initiates communication with nearby agents.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.
            config (Config): Configuration settings.

        Returns:
            None
        """
        try:
            if not config.agents.behavior.communication_enabled:
                logger.debug(f"Communication is disabled in the configuration.")
                return

            nearby_agents = environment.get_agents_within_radius(
                agent.position, agent.communication_range
            )
            for other_agent in nearby_agents:
                if other_agent != agent:
                    message = f"Hello {other_agent.name}, I am {agent.name}."
                    agent.initiate_communication(environment, target_agent=other_agent.name, message=message)
                    # Adjust energy levels
                    agent.state['energy'] -= config.agents.behavior.energy_consumption_rate * 0.5
                    # Log the action
                    logger.info(f"{agent.name} is communicating with {other_agent.name}.")
        except Exception as e:
            logger.error(f"Error during communicate behavior for {agent.name}: {e}")

class CustomBehavior(Behavior):
    """
    Template for creating custom behaviors.

    This class can be used as a starting point for defining new behaviors.
    """

    def execute(self, agent: Agent, environment: World, config: Config):
        """
        Defines the custom behavior execution logic.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.
            config (Config): Configuration settings.

        Returns:
            None
        """
        # Implement custom behavior logic here
        pass

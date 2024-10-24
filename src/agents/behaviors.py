# behaviors.py

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

from abc import ABC, abstractmethod
from environment.world import World
from agents.agent import Agent
import random

class Behavior(ABC):
    """
    Abstract base class for all agent behaviors.

    Methods:
        execute(agent, environment): Executes the behavior for the given agent.
    """

    @abstractmethod
    def execute(self, agent: Agent, environment: World):
        """
        Executes the behavior for the given agent in the environment.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.

        Returns:
            None
        """
        pass

class ExploreBehavior(Behavior):
    """
    Behavior for exploring the environment.
    """

    def execute(self, agent: Agent, environment: World):
        """
        Moves the agent to explore new areas in the environment.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.

        Returns:
            None
        """
        # Move in a random direction
        agent.move(environment, direction='random')
        # Perceive the new environment
        agent.perceive(environment)
        # Update knowledge base
        new_info = environment.get_info_at(agent.position)
        agent.knowledge_base.append(new_info)
        # Log the action
        print(f"{agent.name} is exploring new areas.")
        # Adjust energy levels
        agent.state['energy'] -= 5

class RestBehavior(Behavior):
    """
    Behavior for resting to regain energy.
    """

    def execute(self, agent: Agent, environment: World):
        """
        Allows the agent to rest and regain energy.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.

        Returns:
            None
        """
        agent.rest()
        agent.state['mood'] = 'rested'
        # Log the action
        print(f"{agent.name} is resting to regain energy.")

class AvoidBehavior(Behavior):
    """
    Behavior for avoiding threats or enemies.
    """

    def execute(self, agent: Agent, environment: World):
        """
        Moves the agent away from nearby threats or enemies.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.

        Returns:
            None
        """
        threats = environment.get_threats_near(agent.position)
        if threats:
            closest_threat = min(threats, key=lambda t: environment.distance(agent.position, t.position))
            agent.move_away_from(closest_threat.position, environment)
            agent.state['mood'] = 'alert'
            agent.state['energy'] -= 5
            # Log the action
            print(f"{agent.name} is avoiding a threat at position {closest_threat.position}.")
        else:
            # If no threats are nearby, default to exploring
            ExploreBehavior().execute(agent, environment)

class CommunicateBehavior(Behavior):
    """
    Behavior for communicating with nearby agents.
    """

    def execute(self, agent: Agent, environment: World):
        """
        Initiates communication with nearby agents.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.

        Returns:
            None
        """
        nearby_agents = environment.get_agents_near(agent.position)
        for other_agent in nearby_agents:
            if other_agent != agent:
                message = "Greetings! How are you today?"
                agent.communicate(message, other_agent)
                # Log the action
                print(f"{agent.name} is communicating with {other_agent.name}.")
                # Adjust energy levels
                agent.state['energy'] -= 2

class CustomBehavior(Behavior):
    """
    Template for creating custom behaviors.

    This class can be used as a starting point for defining new behaviors.
    """

    def execute(self, agent: Agent, environment: World):
        """
        Defines the custom behavior execution logic.

        Args:
            agent (Agent): The agent performing the behavior.
            environment (World): The simulation environment.

        Returns:
            None
        """
        # Implement custom behavior logic here
        pass

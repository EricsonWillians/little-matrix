"""
Behaviors module for the little-matrix simulation.
"""

from dataclasses import dataclass
import logging
import random
import math
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from ..environment.world import World
from ..agents.agent import Agent, MovementStrategy
from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class AgentTypeBehaviorTraits:
    """Behavior traits with proper serialization support."""
    gathering_efficiency: float = 1.0
    combat_skill: float = 1.0
    speed_modifier: float = 1.0
    recovery_rate: float = 1.0
    intelligence: float = 1.0
    risk_tolerance: float = 0.5
    protective_instinct: bool = False
    storage_capacity: float = 1.0
    perception_modifier: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert traits to dictionary for serialization."""
        return {
            'gathering_efficiency': float(self.gathering_efficiency),
            'combat_skill': float(self.combat_skill),
            'speed_modifier': float(self.speed_modifier),
            'recovery_rate': float(self.recovery_rate),
            'intelligence': float(self.intelligence),
            'risk_tolerance': float(self.risk_tolerance),
            'protective_instinct': bool(self.protective_instinct),
            'storage_capacity': float(self.storage_capacity),
            'perception_modifier': float(self.perception_modifier)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTypeBehaviorTraits':
        """Create instance from dictionary."""
        return cls(**{
            k: v for k, v in data.items() 
            if k in cls.__dataclass_fields__
        })

class BehaviorType(Enum):
    """Defines the types of behaviors available to agents."""
    SURVIVE = "survive"
    EXPLORE = "explore"
    GATHER = "gather"
    REST = "rest"
    AVOID = "avoid"
    COMMUNICATE = "communicate"
    SOCIALIZE = "socialize"
    COMBAT = "combat"

class Behavior(ABC):
    """Abstract base class for all agent behaviors."""

    @abstractmethod
    def execute(self, agent: Agent, environment: World, config: Config) -> bool:
        """
        Executes the behavior for the given agent in the environment.
        Returns success status of behavior execution.
        """
        pass

    def _calculate_priority(self, agent: Agent) -> float:
        """Calculate the priority of this behavior based on agent state."""
        return 0.0

class SurviveBehavior(Behavior):
    """Primary behavior for ensuring agent survival."""

    def execute(self, agent: Agent, environment: World, config: Config) -> bool:
        try:
            # Check critical needs
            critical_action = self._check_critical_needs(agent)
            if critical_action:
                action, details = critical_action
                
                if action == 'use':
                    agent.use_item(environment, item_type=details['item_type'])
                elif action == 'seek_resource':
                    self._seek_resource(agent, environment, details['resource_type'])
                elif action == 'rest':
                    agent.rest(environment)
                
                return True

            # Maintain healthy resource levels
            if self._should_gather_resources(agent):
                needed_resource = self._determine_needed_resource(agent)
                self._seek_resource(agent, environment, needed_resource)
                return True

            return False
            
        except Exception as e:
            logger.error(f"Error in survive behavior for {agent.name}: {str(e)}")
            return False

    def _check_critical_needs(
        self, 
        agent: Agent
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check for critical needs requiring immediate action."""
        if agent.state['hunger'] >= agent.CRITICAL_HUNGER:
            if agent.inventory.get('food', 0) > 0:
                return ('use', {'item_type': 'food'})
            return ('seek_resource', {'resource_type': 'food'})
            
        if agent.state['thirst'] >= agent.CRITICAL_THIRST:
            if agent.inventory.get('water', 0) > 0:
                return ('use', {'item_type': 'water'})
            return ('seek_resource', {'resource_type': 'water'})
            
        if agent.state['energy'] <= agent.CRITICAL_ENERGY:
            return ('rest', {})
            
        if agent.state['health'] <= agent.CRITICAL_HEALTH:
            if agent.inventory.get('medical_supplies', 0) > 0:
                return ('use', {'item_type': 'medical_supplies'})
            return ('seek_resource', {'resource_type': 'medical_supplies'})
            
        return None

    def _should_gather_resources(self, agent: Agent) -> bool:
        """Determine if agent should gather resources."""
        return (
            agent.state['hunger'] > agent.URGENT_HUNGER or
            agent.state['thirst'] > agent.URGENT_THIRST or
            agent.state['energy'] < agent.URGENT_ENERGY or
            agent.inventory.get('food', 0) < 2 or
            agent.inventory.get('water', 0) < 2
        )

    def _determine_needed_resource(self, agent: Agent) -> str:
        """Determine which resource is most needed."""
        priorities = {
            'food': (agent.state['hunger'] - agent.URGENT_HUNGER) / (100 - agent.URGENT_HUNGER),
            'water': (agent.state['thirst'] - agent.URGENT_THIRST) / (100 - agent.URGENT_THIRST),
            'medical_supplies': (agent.CRITICAL_HEALTH - agent.state['health']) / agent.CRITICAL_HEALTH
        }
        
        # Adjust based on inventory
        if agent.inventory.get('food', 0) > 2:
            priorities['food'] *= 0.5
        if agent.inventory.get('water', 0) > 2:
            priorities['water'] *= 0.5
            
        return max(priorities.items(), key=lambda x: x[1])[0]

    def _seek_resource(
        self, 
        agent: Agent, 
        environment: World, 
        resource_type: str
    ) -> None:
        """Seek out a specific type of resource."""
        target = agent._find_nearest_resource_position(environment, resource_type)
        if target:
            agent.target_position = target
            agent.current_strategy = MovementStrategy.SEEK_RESOURCE
            agent._move_towards_target(environment)
        else:
            agent._random_movement(environment)

class GatherBehavior(Behavior):
    """Behavior for gathering resources efficiently."""

    def execute(self, agent: Agent, environment: World, config: Config) -> bool:
        try:
            nearby_resources = agent._analyze_nearby_resources()
            if not nearby_resources:
                return False

            # Find most valuable nearby resource
            resource = max(
                nearby_resources,
                key=lambda r: self._calculate_resource_value(agent, r)
            )
            
            if resource['distance'] <= 1:  # Adjacent to resource
                agent.collect(environment, resource['type'])
                return True
            else:  # Move towards resource
                direction = agent._get_direction_to_position(resource['position'])
                if direction:
                    agent._execute_move(environment, direction)
                return True
                
        except Exception as e:
            logger.error(f"Error in gather behavior for {agent.name}: {str(e)}")
            return False

    def _calculate_resource_value(
        self, 
        agent: Agent, 
        resource: Dict[str, Any]
    ) -> float:
        """Calculate the value of a resource based on agent needs."""
        base_value = {
            'food': 1.0 if agent.state['hunger'] > 50 else 0.5,
            'water': 1.0 if agent.state['thirst'] > 50 else 0.5,
            'medical_supplies': 1.0 if agent.state['health'] < 70 else 0.3
        }.get(resource['type'], 0.1)

        # Adjust for distance
        distance_factor = 1.0 / (1.0 + resource['distance'])
        
        # Adjust for inventory
        inventory_count = agent.inventory.get(resource['type'], 0)
        inventory_factor = 1.0 if inventory_count < 3 else 0.5
        
        return base_value * distance_factor * inventory_factor

class RestBehavior(Behavior):
    """Enhanced behavior for resting and recovery."""

    def execute(self, agent: Agent, environment: World, config: Config) -> bool:
        try:
            # Find safe resting spot if needed
            if self._is_position_unsafe(agent, environment):
                safe_pos = self._find_safe_resting_spot(agent, environment)
                if safe_pos:
                    direction = agent._get_direction_to_position(safe_pos)
                    if direction:
                        agent._execute_move(environment, direction)
                        return True

            # Rest if position is safe
            if not self._is_position_unsafe(agent, environment):
                agent.rest(environment)
                return True

            return False
            
        except Exception as e:
            logger.error(f"Error in rest behavior for {agent.name}: {str(e)}")
            return False

    def _is_position_unsafe(
        self, 
        agent: Agent, 
        environment: World
    ) -> bool:
        """Determine if current position is unsafe for resting."""
        # Check for nearby threats
        threats = agent._analyze_threats()
        if threats:
            return True
            
        # Check if position is exposed
        nearby_cover = environment.get_objects_within_radius(
            agent.position, 
            1
        )
        if not any(obj.provides_cover for obj in nearby_cover):
            return True
            
        return False

    def _find_safe_resting_spot(
        self, 
        agent: Agent, 
        environment: World
    ) -> Optional[Tuple[int, int]]:
        """Find a safe position for resting."""
        positions = environment.get_empty_adjacent_positions(agent.position)
        
        if not positions:
            return None
            
        # Score positions for safety
        scored_positions = [
            (pos, self._calculate_position_safety(agent, environment, pos))
            for pos in positions
        ]
        
        # Return safest position
        return max(scored_positions, key=lambda x: x[1])[0]

    def _calculate_position_safety(
        self, 
        agent: Agent, 
        environment: World, 
        position: Tuple[int, int]
    ) -> float:
        """Calculate how safe a position is for resting."""
        safety_score = 1.0
        
        # Reduce score based on distance to threats
        for threat in agent._analyze_threats():
            if threat.get('last_known_position'):
                distance = agent._calculate_distance(
                    position,
                    threat['last_known_position']
                )
                safety_score -= 1.0 / (1.0 + distance)
        
        # Increase score for positions with cover
        nearby_objects = environment.get_objects_within_radius(position, 1)
        if any(obj.provides_cover for obj in nearby_objects):
            safety_score += 0.5
            
        return max(0.0, safety_score)

class ExploreBehavior(Behavior):
    """Enhanced behavior for systematic exploration."""

    def execute(self, agent: Agent, environment: World, config: Config) -> bool:
        try:
            agent.current_strategy = MovementStrategy.EXPLORE
            agent._explore_movement(environment)
            return True
        except Exception as e:
            logger.error(f"Error in explore behavior for {agent.name}: {str(e)}")
            return False

class CommunicateBehavior(Behavior):
    """Enhanced behavior for agent communication."""

    def execute(self, agent: Agent, environment: World, config: Config) -> bool:
        try:
            if not config.agents.behavior.communication_enabled:
                return False

            nearby_agents = [
                a for a in agent.state['perceived_agents']
                if environment.get_agent(a) is not None
            ]

            if not nearby_agents:
                return False

            # Find best agent to communicate with
            target_agent = self._select_communication_target(
                agent, 
                environment, 
                nearby_agents
            )
            
            if target_agent:
                message = self._generate_message(agent, target_agent)
                agent.initiate_communication(
                    environment,
                    target_agent,
                    message
                )
                return True

            return False
            
        except Exception as e:
            logger.error(f"Error in communicate behavior for {agent.name}: {str(e)}")
            return False

    def _select_communication_target(
        self, 
        agent: Agent, 
        environment: World, 
        nearby_agents: List[str]
    ) -> Optional[str]:
        """Select the best agent to communicate with."""
        scored_agents = []
        
        for other_agent_name in nearby_agents:
            other_agent = environment.get_agent(other_agent_name)
            if not other_agent:
                continue
                
            # Calculate communication value
            relationship = agent.relationships.get(other_agent_name, 0)
            distance = agent._calculate_distance(
                agent.position,
                other_agent.position
            )
            
            # Prefer agents we haven't communicated with recently
            last_communication = agent.memory.interaction_history.get(
                other_agent_name, 
                []
            )
            recency_factor = 1.0 if not last_communication else 0.5
            
            score = (relationship + 1.0) * recency_factor / (1.0 + distance)
            scored_agents.append((other_agent_name, score))
        
        if scored_agents:
            return max(scored_agents, key=lambda x: x[1])[0]
        return None

    def _generate_message(
        self, 
        agent: Agent, 
        target_agent: str
    ) -> str:
        """Generate appropriate message based on context."""
        # Start with a greeting
        relationship = agent.relationships.get(target_agent, 0)
        
        if relationship > 5:
            greeting = f"Hello friend {target_agent}!"
        elif relationship > 0:
            greeting = f"Hello {target_agent}."
        else:
            greeting = f"Greetings, {target_agent}."

        # Add relevant information
        info_parts = []
        
        # Share resource information
        if agent.memory.resource_locations:
            resource_type = random.choice(list(agent.memory.resource_locations.keys()))
            resources = agent.memory.resource_locations[resource_type]
            if resources:
                resource = random.choice(resources)
                info_parts.append(
                    f"I found {resource_type} at position {resource.position}"
                )

        # Share danger information
        if agent.memory.danger_zones:
            danger_zone = random.choice(agent.memory.danger_zones)
            info_parts.append(f"Be careful near position {danger_zone}")

        # Combine message parts
        message_parts = [greeting]
        if info_parts:
            message_parts.extend(info_parts)

        return " ".join(message_parts)

# Add other behavior classes as needed...
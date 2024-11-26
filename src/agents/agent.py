"""
Agent module for the Little Matrix Simulation.

This module defines the Agent class, representing an autonomous entity within the simulated world.
Agents can perceive their environment, make decisions using LLM assistance, act upon the world,
and communicate with other agents. They possess internal states, knowledge bases, and can adapt
to changing conditions.

Classes:
    MovementStrategy: Enum defining possible movement strategies
    Agent: Represents an individual agent in the simulation
"""

import logging
import random
import json
import re
import math
import time 
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from heapq import heappush, heappop

from ..llm.client import LLMClient
from ..environment.objects import WorldObject
from ..utils.config import (
    Config,
    AgentTypeConfig,
    AgentTypeBehaviorTraits,
    AgentsBehaviorConfig,
    AgentsPerceptionConfig,
)

if TYPE_CHECKING:
    from src.environment.world import World

class MovementStrategy(Enum):
    """Defines possible movement strategies for agents."""
    EXPLORE = "explore"
    SEEK_RESOURCE = "seek_resource"
    FOLLOW_AGENT = "follow_agent"
    FLEE = "flee"
    PATROL = "patrol"
    RANDOM = "random"

@dataclass
class ResourceInfo:
    """Information about a resource in the environment."""
    type: str
    position: Tuple[int, int]
    quantity: int
    last_seen: int  # Timestamp when resource was last observed

@dataclass
class AgentMemory:
    """Represents agent's memory of important locations and events."""
    resource_locations: Dict[str, List[ResourceInfo]]
    danger_zones: List[Tuple[int, int]]
    visited_locations: List[Tuple[int, int]]
    interaction_history: Dict[str, List[Dict[str, Any]]]

class Agent:
    """Represents a sophisticated autonomous agent in the simulation."""

    # Class-level constants
    CRITICAL_HEALTH = 30
    CRITICAL_ENERGY = 20
    CRITICAL_HUNGER = 70
    CRITICAL_THIRST = 70
    URGENT_HUNGER = 50  # Start looking for food earlier
    URGENT_THIRST = 50  # Start looking for water earlier
    URGENT_ENERGY = 30  # Rest before completely exhausted
    
    MAX_MEMORY_SIZE = 100
    STRATEGY_COOLDOWN = 10

    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        config: Config,
        position: Optional[Tuple[int, int]] = None,
        state: Optional[Dict[str, Any]] = None,
        agent_type: Optional[AgentTypeConfig] = None,
        behavior_traits: Optional[AgentTypeBehaviorTraits] = None,
    ):
        # Basic attributes
        self.name = name
        self.llm_client = llm_client
        self.config = config
        self.logger = logging.getLogger(f"Agent:{self.name}")

        # Agent type and behavior
        self.agent_type = agent_type or random.choice(config.agents.customization.types)
        self.behavior_traits = behavior_traits or self._initialize_behavior_traits()
        
        # Initialize state first
        self.state = self._initialize_state() if state is None else state
        
        # Movement and navigation
        self.movement_memory = []
        self.movement_memory_size = 10
        self.current_strategy = MovementStrategy.EXPLORE
        self.target_position = None
        self.path_to_target = []
        self.stuck_counter = 0
        self.exploration_zones = self._initialize_exploration_zones()
        
        # Position
        self.position = position if position is not None else self._get_random_position()
        
        # Knowledge and inventory
        self.knowledge_base = []
        self.inventory = self._initialize_inventory()
        self.memory = AgentMemory(
            resource_locations={},
            danger_zones=[],
            visited_locations=[],
            interaction_history={}
        )
        
        # Capabilities
        self.perception_radius = self._calculate_perception_radius()
        self.communication_range = self.perception_radius + 2
        self.skills = self._initialize_skills()
        
        # Social and goals
        self.goals = random.sample(['survive', 'explore', 'socialize', 'learn', 'dominate'], k=2)
        self.relationships = {}
        self.status_effects = []
        
        self._ensure_state_keys()
        
        self.logger.info(
            f"Agent '{self.name}' of type '{self.agent_type.name}' initialized at {self.position}"
        )

    def _initialize_inventory(self) -> Dict[str, int]:
        """Initialize the agent's inventory with starting items."""
        inventory = {
            'food': 0,
            'water': 0,
            'medical_supplies': 0
        }
        
        # Give starting items based on agent type
        if self.agent_type.name == 'Gatherer':
            inventory.update({
                'food': random.randint(2, 5),
                'water': random.randint(1, 3)
            })
        else:
            inventory.update({
                'food': random.randint(0, 2),
                'water': random.randint(0, 2)
            })
        
        return inventory

    def _get_random_position(self) -> Tuple[int, int]:
        """Generate a random starting position within world bounds."""
        width = self.config.environment.grid.width
        height = self.config.environment.grid.height
        return (random.randint(0, width - 1), random.randint(0, height - 1))

    def _initialize_exploration_zones(self) -> List[Tuple[int, int, int, int]]:
        """Initialize zones for systematic world exploration."""
        world_width = self.config.environment.grid.width
        world_height = self.config.environment.grid.height
        zone_size = min(20, min(world_width, world_height) // 4)
        
        zones = []
        for x in range(0, world_width, zone_size):
            for y in range(0, world_height, zone_size):
                zones.append((
                    x, y,
                    min(x + zone_size, world_width),
                    min(y + zone_size, world_height)
                ))
        
        random.shuffle(zones)
        return zones

    def _initialize_skills(self) -> Dict[str, float]:
        """Initialize agent's skills with base values and type-specific bonuses."""
        # Base skills with random starting values
        skills = {
            'gathering': random.uniform(0.5, 1.5),
            'crafting': random.uniform(0.5, 1.5),
            'combat': random.uniform(0.5, 1.5),
            'social': random.uniform(0.5, 1.5)
        }
        
        try:
            # Safely apply behavior trait modifiers
            gathering_efficiency = getattr(self.behavior_traits, 'gathering_efficiency', None)
            if gathering_efficiency is not None and gathering_efficiency > 0:
                skills['gathering'] *= gathering_efficiency
                
            combat_skill = getattr(self.behavior_traits, 'combat_skill', None)
            if combat_skill is not None and combat_skill > 0:
                skills['combat'] *= combat_skill
            
            # Apply type-specific bonuses
            if self.agent_type.name == 'Explorer':
                skills['gathering'] *= 1.2
                skills['social'] *= 1.1
            elif self.agent_type.name == 'Gatherer':
                skills['gathering'] *= 1.5
                skills['crafting'] *= 1.2
            elif self.agent_type.name == 'Defender':
                skills['combat'] *= 1.5
                skills['social'] *= 0.9
            
            # Ensure no skill goes below minimum threshold
            min_skill_value = 0.1
            for skill in skills:
                skills[skill] = max(min_skill_value, skills[skill])
                
            self.logger.debug(f"Initialized skills for {self.name}: {skills}")
            
        except Exception as e:
            self.logger.error(f"Error initializing skills: {str(e)}. Using base values.")
        
        return skills

    def _initialize_behavior_traits(self) -> AgentTypeBehaviorTraits:
        """Initialize default behavior traits if none provided."""
        default_traits = {
            'gathering_efficiency': 1.0,
            'combat_skill': 1.0,
            'speed_modifier': 1.0,
            'recovery_rate': 1.0,
            'intelligence': 1.0,
            'risk_tolerance': 0.5,
            'protective_instinct': False,
            'storage_capacity': 1.0,
            'perception_modifier': 0.0
        }
        
        if self.agent_type.name == 'Gatherer':
            default_traits.update({
                'gathering_efficiency': 1.5,
                'storage_capacity': 2.0,
                'risk_tolerance': 0.3
            })
        elif self.agent_type.name == 'Explorer':
            default_traits.update({
                'speed_modifier': 1.2,
                'perception_modifier': 1.0,
                'risk_tolerance': 0.8,
                'intelligence': 1.5
            })
        elif self.agent_type.name == 'Defender':
            default_traits.update({
                'combat_skill': 1.5,
                'protective_instinct': True,
                'risk_tolerance': 0.6,
                'recovery_rate': 1.2
            })
        
        return AgentTypeBehaviorTraits(**default_traits)

    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize the agent's state with default values."""
        behavior_config: AgentsBehaviorConfig = self.config.agents.behavior
        
        return {
            'health': 100,
            'energy': behavior_config.initial_energy,
            'mood': random.choice(['excited', 'curious', 'cautious', 'determined']),
            'hunger': random.randint(0, 30),
            'thirst': random.randint(0, 30),
            'experience': 0,
            'level': 1,
            'perceived_agents': [],
            'perceived_objects': [],
            'action': None,
            'action_details': {},
            'last_message': '',
            'strategy_cooldown': 0,
        }

    def _ensure_state_keys(self) -> None:
        """Ensure all required keys are present in the agent's state."""
        default_state = {
            'health': 100,
            'energy': self.config.agents.behavior.initial_energy,
            'mood': 'neutral',
            'hunger': 0,
            'thirst': 0,
            'experience': 0,
            'level': 1,
            'perceived_agents': [],
            'perceived_objects': [],
            'action': None,
            'action_details': {},
            'last_message': '',
            'strategy_cooldown': 0,
        }

        for key, default_value in default_state.items():
            if key not in self.state:
                self.state[key] = default_value
                self.logger.debug(f"Added missing state key '{key}' with default value")

    def _calculate_perception_radius(self) -> int:
        """Calculate the agent's perception radius based on type and traits."""
        perception_config: AgentsPerceptionConfig = self.config.agents.perception
        base_radius = perception_config.base_radius
        modifiers = getattr(perception_config, 'modifiers', {})
        type_modifier = modifiers.get(self.agent_type.name, 0)
        
        # Apply trait modifiers if they exist
        trait_modifier = 0
        if hasattr(self.behavior_traits, 'perception_modifier'):
            trait_modifier = self.behavior_traits.perception_modifier
        
        final_radius = base_radius + type_modifier + trait_modifier
        return max(1, final_radius)  # Ensure radius is at least 1

    def _initialize_exploration_zones(self) -> List[Tuple[int, int, int, int]]:
        """Initialize zones for systematic world exploration."""
        world_width = self.config.environment.grid.width
        world_height = self.config.environment.grid.height
        zone_size = min(20, min(world_width, world_height) // 4)
        
        zones = []
        for x in range(0, world_width, zone_size):
            for y in range(0, world_height, zone_size):
                zones.append((
                    x, y,
                    min(x + zone_size, world_width),
                    min(y + zone_size, world_height)
                ))
        
        random.shuffle(zones)
        return zones

    def perceive(self, world: 'World') -> None:
        """Update agent's perception of the environment."""
        try:
            # Gather perceptions
            perceived_agents = [
                agent for agent in world.get_agents_within_radius(
                    self.position,
                    self.perception_radius
                )
                if agent.name != self.name
            ]
            
            perceived_objects = world.get_objects_within_radius(
                self.position,
                self.perception_radius
            )
            
            # Update state with perceptions
            self.state['perceived_agents'] = [agent.name for agent in perceived_agents]
            self.state['perceived_objects'] = [
                {
                    'type': obj.__class__.__name__,
                    'position': obj.position,
                    'quantity': getattr(obj, 'quantity', 1)
                }
                for obj in perceived_objects
            ]
            
            # Update memory with new information
            timestamp = world.current_time
            self._update_resource_memory(timestamp)
            self._update_danger_zones(perceived_agents)
            
            self.logger.debug(
                f"{self.name} perceived {len(perceived_agents)} agents and "
                f"{len(perceived_objects)} objects."
            )
            
        except Exception as e:
            self.logger.error(f"Error during perception: {str(e)}")

    def _get_survival_priorities(self) -> Dict[str, float]:
        """Calculate current survival priorities."""
        return {
            'find_food': max(0, (self.state['hunger'] - 20) / 80),
            'find_water': max(0, (self.state['thirst'] - 20) / 80),
            'rest': max(0, (100 - self.state['energy']) / 100),
            'heal': max(0, (100 - self.state['health']) / 100),
        }

    def _analyze_nearby_resources(self) -> List[Dict[str, Any]]:
        """Analyze nearby resources with improved priority."""
        resources = []
        for obj in self.state.get('perceived_objects', []):
            if isinstance(obj, dict) and 'type' in obj and 'position' in obj:
                distance = self._calculate_distance(self.position, obj['position'])
                priority = self._calculate_resource_priority(obj['type'])
                
                if distance <= 1:  # Immediately collectable
                    priority *= 2
                    
                resources.append({
                    'type': obj['type'],
                    'position': obj['position'],
                    'distance': distance,
                    'priority': priority,
                    'is_critical': self._is_resource_critical(obj['type'])
                })
        
        return sorted(resources, key=lambda x: (-x['priority'], x['distance']))

    def _is_resource_critical(self, resource_type: str) -> bool:
        """Determine if a resource type is critically needed."""
        if resource_type == 'food':
            return self.state['hunger'] >= self.URGENT_HUNGER
        elif resource_type == 'water':
            return self.state['thirst'] >= self.URGENT_THIRST
        return False

    def _calculate_resource_priority(self, resource_type: str) -> float:
        """Calculate priority of collecting a resource."""
        priority = 0.0
        
        if resource_type == 'food':
            hunger_ratio = self.state['hunger'] / 100.0
            inventory_food = self.inventory.get('food', 0)
            priority = hunger_ratio * (1 + (1 / (inventory_food + 1)))
            
        elif resource_type == 'water':
            thirst_ratio = self.state['thirst'] / 100.0
            inventory_water = self.inventory.get('water', 0)
            priority = thirst_ratio * (1 + (1 / (inventory_water + 1)))
        
        if self.state['energy'] < self.URGENT_ENERGY:
            priority *= 0.5  # Reduce priority if energy is low
            
        return priority

    def _prepare_decision_context(self) -> Dict[str, Any]:
        """Prepare context for LLM decision making with proper knowledge."""
        nearby_resources = self._analyze_nearby_resources()
        survival_status = {
            'health': self.state['health'],
            'energy': self.state['energy'],
            'hunger': self.state['hunger'],
            'thirst': self.state['thirst'],
        }
        
        # Add urgency indicators
        survival_status['needs_food'] = self.state['hunger'] >= self.URGENT_HUNGER
        survival_status['needs_water'] = self.state['thirst'] >= self.URGENT_THIRST
        survival_status['needs_rest'] = self.state['energy'] <= self.URGENT_ENERGY
        
        return {
            'agent_name': self.name,
            'agent_type': self.agent_type.name,
            'knowledge_base': self.knowledge_base[-10:],  # Last 10 memories
            'state': survival_status,
            'inventory': self.inventory,
            'perceived_agents': self.state.get('perceived_agents', []),
            'perceived_objects': nearby_resources,
            'position': self.position,
            'current_strategy': self.current_strategy.value,
        }

    def decide(self) -> None:
        """Make a decision based on current state and knowledge."""
        try:
            # Check for critical conditions first
            critical_action = self._check_critical_conditions()
            if critical_action:
                self.state['action'], self.state['action_details'] = critical_action
                return
            
            # Prepare context for LLM
            prompt_vars = self._prepare_decision_context()
            
            # Get LLM decision
            prompt = self.llm_client.prompt_manager.get_prompt('agent_decision', **prompt_vars)
            response = self.llm_client.generate_response(prompt)
            
            # Parse and validate decision
            action, details = self._parse_llm_response(response)
            action, details = self._validate_decision(action, details)
            
            # Update state with decision
            self.state['action'] = action
            self.state['action_details'] = details
            
            self.logger.info(
                f"{self.name} decided to {action} with details: {details}"
            )
            
        except Exception as e:
            self.logger.error(f"Decision-making failed: {str(e)}. Defaulting to survival action.")
            self.state['action'], self.state['action_details'] = self._get_emergency_action()

    def _handle_seek_resource(self, world: 'World', resource_type: str = None) -> None:
        """Handle the seek_resource action."""
        if resource_type is None:
            resource_type = self._determine_most_urgent_resource()
            
        target = self._find_nearest_resource_position(world, resource_type)
        if target:
            self.target_position = target
            self.current_strategy = MovementStrategy.SEEK_RESOURCE
            self._move_towards_target(world)
        else:
            self._random_movement(world)  # Fallback to random movement if no resource found

    def act(self, world: 'World') -> None:
        """Execute actions with priority on resource collection."""
        # Always try to collect resources at current position first
        if self.collect(world):
            return

        # Then proceed with regular action
        action = self.state.get('action', 'rest')
        details = self.state.get('action_details', {})

        action_map = {
            'move': self.move,
            'collect': self.collect,
            'use': self.use_item,
            'rest': self.rest,
            'seek_resource': self._handle_seek_resource
        }

        try:
            if action in action_map:
                action_method = action_map[action]
                if details:
                    action_method(world, **details)
                else:
                    action_method(world)
                
            self.update_state()
            
        except Exception as e:
            self.logger.error(f"Error executing action '{action}': {str(e)}")
            self._handle_action_failure(world)

    # Movement methods
    def move(self, world: 'World', direction: Optional[str] = None) -> None:
        """Execute movement based on current strategy or specified direction."""
        if direction:
            self._execute_move(world, direction)
            return

        # Update strategy if cooldown expired
        if self.state['strategy_cooldown'] <= 0:
            self._update_movement_strategy(world)
        else:
            self.state['strategy_cooldown'] -= 3

        # Execute movement based on strategy
        strategy_movements = {
            MovementStrategy.EXPLORE: self._explore_movement,
            MovementStrategy.SEEK_RESOURCE: self._seek_resource_movement,
            MovementStrategy.FLEE: self._flee_movement,
            MovementStrategy.PATROL: self._patrol_movement,
            MovementStrategy.FOLLOW_AGENT: self._follow_agent_movement,
            MovementStrategy.RANDOM: self._random_movement
        }

        movement_func = strategy_movements.get(self.current_strategy, self._random_movement)
        movement_func(world)

        # Update memory and check status
        self._update_movement_memory()
        self._check_stuck_status()

    def _update_movement_strategy(self, world: 'World') -> None:
        """Update movement strategy with better resource seeking."""
        # First check if we need to rest
        if self.state['energy'] <= self.URGENT_ENERGY:
            self.current_strategy = MovementStrategy.RANDOM
            return

        # Then check for critical resources
        nearby_resources = self._analyze_nearby_resources()
        for resource in nearby_resources:
            if resource['is_critical'] and resource['distance'] <= self.perception_radius:
                self.current_strategy = MovementStrategy.SEEK_RESOURCE
                self.target_position = resource['position']
                return

        # Default exploration behavior
        if random.random() < 0.2:  # 20% chance to change strategy
            self.current_strategy = random.choice([
                MovementStrategy.EXPLORE,
                MovementStrategy.PATROL
            ])

    def _execute_move(self, world: 'World', direction: str) -> None:
        """
        Execute a single movement step in the specified direction, handling collisions.
        
        Args:
            world: The simulation world
            direction: Direction to move ('north', 'south', 'east', 'west')
        """
        direction_map = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0)
        }
        
        if direction not in direction_map:
            return
            
        dx, dy = direction_map[direction]
        current_x, current_y = self.position
        new_x = current_x + dx
        new_y = current_y + dy
        
        if self.config.environment.grid.wrap_around:
            new_x %= self.config.environment.grid.width
            new_y %= self.config.environment.grid.height
            
        new_position = (new_x, new_y)
        
        # Check if the new position is valid and not occupied
        try:
            if world.is_position_valid(new_position) and not world.is_position_occupied(new_position):
                # Remove from current position
                world.remove_agent(self)
                
                # Update position
                self.position = new_position
                
                # Add to new position
                world.add_agent(self, position=self.position)
                
                # Apply energy cost
                energy_cost = self._calculate_movement_energy_cost()
                self.state['energy'] = max(0, self.state['energy'] - energy_cost)
                
                # Record visit
                if new_position not in self.memory.visited_locations:
                    self.memory.visited_locations.append(new_position)
                    
                self.logger.debug(f"{self.name} moved to {new_position}")
            else:
                # If position is occupied or invalid, try to find an alternative
                alternative_position = self._find_alternative_position(world, new_position)
                if alternative_position:
                    # Remove from current position
                    world.remove_agent(self)
                    
                    # Update position
                    self.position = alternative_position
                    
                    # Add to new position
                    world.add_agent(self, position=self.position)
                    
                    # Apply energy cost (slightly higher for detour)
                    energy_cost = self._calculate_movement_energy_cost() * 1.2
                    self.state['energy'] = max(0, self.state['energy'] - energy_cost)
                    
                    self.logger.debug(f"{self.name} took alternative route to {alternative_position}")
                else:
                    self.logger.debug(f"{self.name} stayed at {self.position} due to blocked path")
                    
        except Exception as e:
            self.logger.error(f"Error during movement: {str(e)}")
            # Ensure agent stays in a valid position
            if not hasattr(self, 'position') or not world.is_position_valid(self.position):
                self.position = self._get_random_position()
                world.add_agent(self, position=self.position)

    def _find_alternative_position(self, world: 'World', blocked_position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Find an alternative position when the desired position is blocked.
        
        Args:
            world: The simulation world
            blocked_position: The position that was blocked
            
        Returns:
            Optional[Tuple[int, int]]: Alternative position or None if none found
        """
        # Check adjacent positions in random order
        adjacent_positions = [
            (blocked_position[0] + 1, blocked_position[1]),
            (blocked_position[0] - 1, blocked_position[1]),
            (blocked_position[0], blocked_position[1] + 1),
            (blocked_position[0], blocked_position[1] - 1)
        ]
        random.shuffle(adjacent_positions)
        
        for pos in adjacent_positions:
            # Handle world wrapping
            if self.config.environment.grid.wrap_around:
                pos = (
                    pos[0] % self.config.environment.grid.width,
                    pos[1] % self.config.environment.grid.height
                )
            
            if world.is_position_valid(pos) and not world.is_position_occupied(pos):
                return pos
                
        return None

    def _get_random_position(self) -> Tuple[int, int]:
        """
        Generate a random valid position in the world.
        
        Returns:
            Tuple[int, int]: Valid random position
        """
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            x = random.randint(0, self.config.environment.grid.width - 1)
            y = random.randint(0, self.config.environment.grid.height - 1)
            position = (x, y)
            
            try:
                if world.is_position_valid(position) and not world.is_position_occupied(position):
                    return position
            except:
                pass
                
            attempts += 1
        
        # If we couldn't find a free position, try to get any valid position
        return (
            random.randint(0, self.config.environment.grid.width - 1),
            random.randint(0, self.config.environment.grid.height - 1)
        )

    def _explore_movement(self, world: 'World') -> None:
        """Execute exploration movement strategy."""
        if not self.target_position or self._reached_target():
            if self.exploration_zones:
                zone = self.exploration_zones[0]
                # Pick random point in zone
                target_x = random.randint(zone[0], zone[2])
                target_y = random.randint(zone[1], zone[3])
                self.target_position = (target_x, target_y)
                # Cycle zone to end
                self.exploration_zones = self.exploration_zones[1:] + [self.exploration_zones[0]]
        
        if self.target_position:
            self._move_towards_target(world)

    def _seek_resource_movement(self, world: 'World') -> None:
        """Move towards needed resources."""
        if not self.target_position or self._reached_target():
            # Find new resource target
            resource_type = self._determine_needed_resource()
            target = self._find_nearest_resource_position(world, resource_type)
            
            if target:
                self.target_position = target
            else:
                self.current_strategy = MovementStrategy.EXPLORE
                return
        
        # Execute pathfinding movement
        if not self.path_to_target:
            self.path_to_target = self._find_path_to_target(world, self.target_position)
        
        if self.path_to_target:
            next_pos = self.path_to_target[0]
            direction = self._get_direction_to_position(next_pos)
            if direction:
                self._execute_move(world, direction)
                self.path_to_target = self.path_to_target[1:]

    def _flee_movement(self, world: 'World') -> None:
        """Move away from threats."""
        threats = self._analyze_threats()
        if not threats:
            self.current_strategy = MovementStrategy.EXPLORE
            return

        # Calculate average threat position
        threat_positions = [t.get('last_known_position') for t in threats if t.get('last_known_position')]
        if threat_positions:
            avg_x = sum(p[0] for p in threat_positions) / len(threat_positions)
            avg_y = sum(p[1] for p in threat_positions) / len(threat_positions)
            
            # Calculate opposite direction
            dx = self.position[0] - avg_x
            dy = self.position[1] - avg_y
            
            # Choose direction that maximizes distance from threats
            if abs(dx) > abs(dy):
                direction = 'east' if dx > 0 else 'west'
            else:
                direction = 'south' if dy > 0 else 'north'
            
            self._execute_move(world, direction)

    def _patrol_movement(self, world: 'World') -> None:
        """Execute patrol movement pattern."""
        if not hasattr(self, 'patrol_points'):
            # Create patrol points around current position
            x, y = self.position
            radius = 5
            self.patrol_points = [
                (x + radius, y),
                (x, y + radius),
                (x - radius, y),
                (x, y - radius)
            ]
            self.current_patrol_index = 0

        target = self.patrol_points[self.current_patrol_index]
        if self._calculate_distance(self.position, target) < 2:
            # Move to next patrol point
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
            target = self.patrol_points[self.current_patrol_index]

        direction = self._get_direction_to_position(target)
        if direction:
            self._execute_move(world, direction)

    def _follow_agent_movement(self, world: 'World') -> None:
        """Follow another agent."""
        perceived_agents = self.state.get('perceived_agents', [])
        if not perceived_agents:
            self.current_strategy = MovementStrategy.EXPLORE
            return

        # Find agent with best relationship
        target_agent = None
        best_relationship = -float('inf')
        
        for agent_name in perceived_agents:
            relationship = self.relationships.get(agent_name, 0)
            if relationship > best_relationship:
                best_relationship = relationship
                target_agent = world.get_agent(agent_name)

        if target_agent and best_relationship > 0:
            self.target_position = target_agent.position
            self._move_towards_target(world)
        else:
            self.current_strategy = MovementStrategy.EXPLORE

    def _random_movement(self, world: 'World') -> None:
        """Execute random movement."""
        direction = random.choice(['north', 'south', 'east', 'west'])
        self._execute_move(world, direction)

    # Pathfinding and navigation methods
    def _find_path_to_target(self, world: 'World', target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding implementation."""
        def heuristic(pos: Tuple[int, int]) -> float:
            return self._calculate_distance(pos, target)

        frontier = [(0, self.position)]
        came_from = {self.position: None}
        cost_so_far = {self.position: 0}

        while frontier:
            current = heappop(frontier)[1]
            
            if current == target:
                break
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if not world.is_position_valid(next_pos):
                    continue
                    
                # Consider terrain cost
                terrain_cost = world.get_terrain_cost(next_pos)
                new_cost = cost_so_far[current] + terrain_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        path = []
        current = target
        while current and current != self.position:
            path.append(current)
            current = came_from.get(current)
        
        return list(reversed(path))

    def _move_towards_target(self, world: 'World') -> None:
        """Move towards current target position."""
        if not self.target_position:
            return

        if not self.path_to_target:
            self.path_to_target = self._find_path_to_target(world, self.target_position)

        if self.path_to_target:
            next_pos = self.path_to_target[0]
            direction = self._get_direction_to_position(next_pos)
            if direction:
                self._execute_move(world, direction)
                self.path_to_target = self.path_to_target[1:]

    # Resource management methods
    def _determine_needed_resource(self) -> str:
        """Determine most critically needed resource."""
        if self.state['hunger'] >= self.CRITICAL_HUNGER:
            return 'food'
        elif self.state['thirst'] >= self.CRITICAL_THIRST:
            return 'water'
        elif self.state['health'] < self.CRITICAL_HEALTH:
            return 'medical_supplies'
        return 'food'  # Default to food

    def _find_nearest_resource_position(self, world: 'World', resource_type: str) -> Optional[Tuple[int, int]]:
        """Find position of nearest needed resource."""
        # Check memory first
        memory_resources = self.memory.resource_locations.get(resource_type, [])
        valid_memory_resources = []
        
        for resource in memory_resources:
            if world.is_resource_still_present(resource.position, resource_type):
                distance = self._calculate_distance(self.position, resource.position)
                valid_memory_resources.append((distance, resource.position))
        
        # Check currently perceived resources
        perceived_resources = []
        for obj in self.state['perceived_objects']:
            if obj.get('type') == resource_type:
                pos = obj.get('position')
                if pos:
                    distance = self._calculate_distance(self.position, pos)
                    perceived_resources.append((distance, pos))
        
        # Combine and find nearest
        all_resources = valid_memory_resources + perceived_resources
        if all_resources:
            return min(all_resources, key=lambda x: x[0])[1]
        
        return None

    def _update_resource_memory(self, timestamp: int) -> None:
        """Update memory of resource locations."""
        for obj in self.state['perceived_objects']:
            if 'type' in obj and 'position' in obj:
                resource_type = obj['type']
                if resource_type not in self.memory.resource_locations:
                    self.memory.resource_locations[resource_type] = []
                
                # Add or update resource location
                resource_info = ResourceInfo(
                    type=resource_type,
                    position=obj['position'],
                    quantity=obj.get('quantity', 1),
                    last_seen=timestamp
                )
                
                # Update existing or add new
                existing = False
                for i, res in enumerate(self.memory.resource_locations[resource_type]):
                    if res.position == obj['position']:
                        self.memory.resource_locations[resource_type][i] = resource_info
                        existing = True
                        break
                
                if not existing:
                    self.memory.resource_locations[resource_type].append(resource_info)

    # Threat analysis methods
    def _analyze_threats(self) -> List[Dict[str, Any]]:
        """Analyze and evaluate potential threats."""
        threats = []
        for agent_name in self.state['perceived_agents']:
            relationship = self.relationships.get(agent_name, 0)
            if relationship < 0:  # Negative relationship indicates threat
                agent = world.get_agent(agent_name)
                if agent:
                    threat_level = abs(relationship) * (1 + (agent.state['level'] - self.state['level']) * 0.2)
                    threats.append({
                        'name': agent_name,
                        'threat_level': threat_level,
                        'last_known_position': agent.position,
                        'relative_strength': agent.state['health'] / self.state['health']
                    })
        
        return sorted(threats, key=lambda x: x['threat_level'], reverse=True)

    def _update_danger_zones(self, perceived_agents: List['Agent']) -> None:
        """Update memory of dangerous areas."""
        current_time = time.time()
        
        # Add new danger zones
        for agent in perceived_agents:
            if self.relationships.get(agent.name, 0) < 0:
                if agent.position not in self.memory.danger_zones:
                    self.memory.danger_zones.append(agent.position)

        # Remove old danger zones (older than 5 minutes)
        self.memory.danger_zones = self.memory.danger_zones[-10:]  # Keep only recent zones

    # Status checks and updates
    def _check_critical_conditions(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check for critical conditions requiring immediate action."""
        if self.state['health'] <= self.CRITICAL_HEALTH:
            return self._get_healing_action()
            
        if self.state['energy'] <= self.CRITICAL_ENERGY:
            return ('rest', {})
            
        if self.state['hunger'] >= self.CRITICAL_HUNGER:
            if self.inventory.get('food', 0) > 0:
                return ('use', {'item_type': 'food'})
            return ('seek_resource', {'resource_type': 'food'})
            
        if self.state['thirst'] >= self.CRITICAL_THIRST:
            if self.inventory.get('water', 0) > 0:
                return ('use', {'item_type': 'water'})
            return ('seek_resource', {'resource_type': 'water'})
            
        return None

    def _get_healing_action(self) -> Tuple[str, Dict[str, Any]]:
        """Determine best action for healing."""
        if self.inventory.get('medical_supplies', 0) > 0:
            return ('use', {'item_type': 'medical_supplies'})
        if self.state['energy'] < 50:
            return ('rest', {})
        return ('seek_resource', {'resource_type': 'medical_supplies'})

    def _should_seek_resources(self) -> bool:
        """Determine if agent should seek resources."""
        return (
            self.state['hunger'] > self.URGENT_HUNGER or
            self.state['thirst'] > self.URGENT_THIRST or
            self.state['energy'] < self.URGENT_ENERGY or
            self.state['health'] < self.CRITICAL_HEALTH * 1.5 or
            self.inventory.get('food', 0) < 2 or
            self.inventory.get('water', 0) < 2
        )

    def _get_emergency_action(self) -> Tuple[str, Dict[str, Any]]:
        """Get safe fallback action when normal decision-making fails."""
        if self.state['energy'] < self.CRITICAL_ENERGY:
            return ('rest', {})
        return ('move', {'direction': self._get_safe_direction()})

    def use_item(self, world: 'World', item_type: str) -> None:
        """Use an item from the agent's inventory."""
        if item_type in self.inventory and self.inventory[item_type] > 0:
            self.inventory[item_type] -= 1
            
            # Apply item effects
            if item_type == 'food':
                hunger_reduction = random.randint(15, 25)
                energy_boost = random.randint(5, 10)
                self.state['hunger'] = max(0, self.state['hunger'] - hunger_reduction)
                self.state['energy'] = min(100, self.state['energy'] + energy_boost)
                self.logger.info(f"{self.name} ate food, reducing hunger by {hunger_reduction}")
                
            elif item_type == 'water':
                thirst_reduction = random.randint(15, 25)
                energy_boost = random.randint(3, 7)
                self.state['thirst'] = max(0, self.state['thirst'] - thirst_reduction)
                self.state['energy'] = min(100, self.state['energy'] + energy_boost)
                self.logger.info(f"{self.name} drank water, reducing thirst by {thirst_reduction}")
                
            elif item_type == 'medical_supplies':
                healing = random.randint(20, 30)
                self.state['health'] = min(100, self.state['health'] + healing)
                self.logger.info(f"{self.name} used medical supplies, healing {healing} health")
            
            # Consider consumption time
            energy_cost = self.config.agents.behavior.energy_consumption_rate * 0.2  # Less energy than actions
            self.state['energy'] = max(0, self.state['energy'] - energy_cost)
        else:
            self.logger.warning(f"No {item_type} in inventory")

    def craft(self, world: 'World', item_type: str, recipe: Dict[str, int]) -> None:
        """
        Craft items using resources from inventory.

        Args:
            world: The simulation world
            item_type: Type of item to craft
            recipe: Dictionary of required resources and their quantities
        """
        try:
            # Check if has all required resources
            if all(self.inventory.get(item, 0) >= quantity for item, quantity in recipe.items()):
                # Calculate crafting success chance based on skills
                base_chance = 0.6
                skill_bonus = self.skills.get('crafting', 0) * 0.4
                success_chance = min(0.95, base_chance + skill_bonus)
                
                if random.random() < success_chance:
                    # Consume resources
                    for item, quantity in recipe.items():
                        self.inventory[item] -= quantity
                    
                    # Add crafted item
                    self.inventory[item_type] = self.inventory.get(item_type, 0) + 1
                    
                    # Apply energy cost
                    energy_cost = self._calculate_action_energy_cost('craft')
                    self.state['energy'] = max(0, self.state['energy'] - energy_cost)
                    
                    # Improve crafting skill
                    crafting_skill_increase = self._calculate_skill_increase('crafting')
                    self.skills['crafting'] += crafting_skill_increase
                    
                    self.logger.info(f"{self.name} successfully crafted {item_type}")
                else:
                    # Failed craft attempt - consume half resources
                    for item, quantity in recipe.items():
                        self.inventory[item] -= quantity // 2
                    self.logger.info(f"{self.name} failed to craft {item_type}")
            else:
                self.logger.warning(f"Insufficient resources to craft {item_type}")
                
        except Exception as e:
            self.logger.error(f"Error during crafting: {str(e)}")

    def _get_safe_direction(self) -> str:
        """Determine safest direction to move."""
        threats = self._analyze_threats()
        if not threats:
            return random.choice(['north', 'south', 'east', 'west'])
            
        # Move away from threats
        threat_positions = [t['last_known_position'] for t in threats]
        if threat_positions:
            # Calculate average threat position and move opposite
            avg_x = sum(p[0] for p in threat_positions) / len(threat_positions)
            avg_y = sum(p[1] for p in threat_positions) / len(threat_positions)
            
            dx = self.position[0] - avg_x
            dy = self.position[1] - avg_y
            
            if abs(dx) > abs(dy):
                return 'east' if dx > 0 else 'west'
            return 'south' if dy > 0 else 'north'
            
        return random.choice(['north', 'south', 'east', 'west'])

    def _handle_action_failure(self, world: 'World') -> None:
        """Handle failed actions gracefully."""
        # Reset any stuck states
        self.stuck_counter = 0
        self.path_to_target = []
        
        # Choose safe fallback action
        action, details = self._get_emergency_action()
        self.state['action'] = action
        self.state['action_details'] = details
        
        # Execute fallback action
        if action == 'move':
            self.move(world, details.get('direction'))
        elif action == 'rest':
            self.rest(world)

    def update_state(self) -> None:
        """Update agent's state based on current situation."""
        # Update basic needs
        self.state['hunger'] = min(100, self.state['hunger'] + random.randint(1, 2))
        self.state['thirst'] = min(100, self.state['thirst'] + random.randint(1, 2))
        self.state['energy'] = max(0, self.state['energy'] - random.randint(1, 2))

        # Apply status effects
        self._apply_status_effects()

        # Update mood and experience
        self.state['mood'] = self._calculate_mood()
        self._update_experience()

        # Decay relationships slightly over time
        self._decay_relationships()

    def _apply_status_effects(self) -> None:
        """Apply current status effects to agent state."""
        if self.state['hunger'] > 80 or self.state['thirst'] > 80:
            health_loss = random.randint(1, 3)
            self.state['health'] = max(0, self.state['health'] - health_loss)
            if 'Weakness' not in self.status_effects:
                self.status_effects.append('Weakness')
        elif self.state['hunger'] < 30 and self.state['thirst'] < 30 and self.state['energy'] > 50:
            health_gain = random.randint(1, 2)
            self.state['health'] = min(100, self.state['health'] + health_gain)
            if 'Weakness' in self.status_effects:
                self.status_effects.remove('Weakness')

    def _update_experience(self) -> None:
        """Update experience and handle leveling."""
        self.state['experience'] += random.randint(1, 3)
        if self.state['experience'] >= 100 * self.state['level']:
            self.level_up()

    def _decay_relationships(self) -> None:
        """Decay relationship values over time."""
        decay_rate = 0.01
        for agent_name in list(self.relationships.keys()):
            self.relationships[agent_name] *= (1 - decay_rate)
            if abs(self.relationships[agent_name]) < 0.1:
                del self.relationships[agent_name]

    def collect(self, world: 'World', object_type: str = None) -> bool:
        """
        Collect resources at the current position.
        
        Args:
            world: World instance
            object_type: Optional specific type to collect
        
        Returns:
            bool: True if collection successful
        """
        # Check current position first
        objects = world.get_objects_at_position(self.position)
        
        for obj in objects:
            obj_type = getattr(obj, 'type', obj.__class__.__name__.lower())
            
            # Match food resources
            if obj_type in ['food', '*'] and (object_type is None or object_type in ['food', '*']):
                if world.remove_object(obj):
                    self.inventory['food'] = self.inventory.get('food', 0) + 1
                    if self.state['hunger'] >= self.URGENT_HUNGER:
                        self.use_item(world, 'food')
                    self.logger.info(f"{self.name} collected food at {self.position}")
                    return True
                    
            # Match water resources
            elif obj_type in ['water', '~'] and (object_type is None or object_type in ['water', '~']):
                if world.remove_object(obj):
                    self.inventory['water'] = self.inventory.get('water', 0) + 1
                    if self.state['thirst'] >= self.URGENT_THIRST:
                        self.use_item(world, 'water')
                    self.logger.info(f"{self.name} collected water at {self.position}")
                    return True
        
        return False

    def rest(self, world: 'World') -> None:
        """Rest to recover energy and health."""
        # Base recovery values
        energy_gain = random.randint(8, 15)
        health_gain = random.randint(1, 3)
        
        # Apply trait modifiers
        if hasattr(self.behavior_traits, 'recovery_rate'):
            energy_gain *= self.behavior_traits.recovery_rate
            health_gain *= self.behavior_traits.recovery_rate
        
        # Apply recovery
        self.state['energy'] = min(100, self.state['energy'] + energy_gain)
        self.state['health'] = min(100, self.state['health'] + health_gain)
        
        # Increase hunger and thirst while resting
        self.state['hunger'] = min(100, self.state['hunger'] + random.randint(1, 3))
        self.state['thirst'] = min(100, self.state['thirst'] + random.randint(1, 3))
        
        self.logger.info(
            f"{self.name} rested and recovered {energy_gain} energy and {health_gain} health"
        )

    def attack(self, world: 'World', target_agent: str) -> None:
        """Attack another agent."""
        target_agent_obj = world.get_agent(target_agent)
        if not target_agent_obj:
            self.logger.warning(f"Target agent {target_agent} not found")
            return
            
        if not self._is_within_attack_range(target_agent_obj):
            self.logger.warning(f"{target_agent} is out of attack range")
            return
            
        # Calculate damage based on skills and traits
        base_damage = random.randint(5, 15)
        damage_multiplier = self.behavior_traits.combat_skill or 1.0
        damage = int(base_damage * damage_multiplier)
        
        # Apply damage
        target_agent_obj.state['health'] -= damage
        
        # Apply energy cost
        energy_cost = self._calculate_action_energy_cost('attack')
        self.state['energy'] = max(0, self.state['energy'] - energy_cost)
        
        # Improve combat skill
        combat_skill_increase = self._calculate_skill_increase('combat')
        self.skills['combat'] += combat_skill_increase
        
        # Update relationships
        self.relationships[target_agent] = max(-10, self.relationships.get(target_agent, 0) - 2)
        
        self.logger.info(f"{self.name} attacked {target_agent} for {damage} damage")
        
        # Check if target is defeated
        if target_agent_obj.state['health'] <= 0:
            self.logger.info(f"{target_agent} has been defeated by {self.name}")
            # Gain experience for defeat
            self.state['experience'] += 25 * target_agent_obj.state['level']
            world.remove_agent(target_agent_obj)

    def level_up(self) -> None:
        """Level up the agent and improve attributes."""
        self.state['level'] += 1
        self.state['experience'] = 0
        
        # Improve base stats
        health_boost = random.randint(5, 10)
        energy_boost = random.randint(5, 10)
        self.state['health'] = min(100, self.state['health'] + health_boost)
        self.state['energy'] = min(100, self.state['energy'] + energy_boost)
        
        # Improve skills
        skill_improvement = random.uniform(0.05, 0.1)
        for skill in self.skills:
            self.skills[skill] += skill_improvement
        
        # Special trait improvements based on agent type
        if self.agent_type.name == 'Explorer':
            self.perception_radius += 1
        elif self.agent_type.name == 'Gatherer':
            self.behavior_traits.gathering_efficiency *= 1.1
        elif self.agent_type.name == 'Defender':
            self.behavior_traits.combat_skill *= 1.1
        
        self.logger.info(
            f"{self.name} leveled up to {self.state['level']}! "
            f"Health +{health_boost}, Energy +{energy_boost}, "
            f"Skills +{skill_improvement:.2f}"
        )

    def _calculate_movement_energy_cost(self) -> float:
        """Calculate energy cost for movement with safe speed modifier access."""
        base_cost = self.config.agents.behavior.energy_consumption_rate
        
        # Safely get speed modifier with default
        speed_modifier = getattr(self.behavior_traits, 'speed_modifier', 1.0)
        if speed_modifier is None or speed_modifier <= 0:
            speed_modifier = 1.0
        
        # Additional cost for carrying items
        inventory_weight = sum(self.inventory.values())
        weight_factor = 1 + (inventory_weight * 0.1)
        
        return base_cost * weight_factor / speed_modifier

    def _calculate_mood(self) -> str:
        """Calculate agent's current mood based on state."""
        # Weight factors for different aspects
        weights = {
            'health': 0.3,
            'energy': 0.2,
            'hunger': 0.25,
            'thirst': 0.25
        }
        
        # Calculate normalized scores
        scores = {
            'health': self.state['health'] / 100,
            'energy': self.state['energy'] / 100,
            'hunger': (100 - self.state['hunger']) / 100,
            'thirst': (100 - self.state['thirst']) / 100
        }
        
        # Calculate weighted average
        mood_score = sum(scores[k] * weights[k] for k in weights)
        
        # Determine mood based on score
        if mood_score > 0.8:
            return 'ecstatic'
        elif mood_score > 0.6:
            return 'happy'
        elif mood_score > 0.4:
            return 'neutral'
        elif mood_score > 0.2:
            return 'sad'
        else:
            return 'depressed'

    def _get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of agent's current status."""
        return {
            'name': self.name,
            'type': self.agent_type.name,
            'level': self.state['level'],
            'health': self.state['health'],
            'energy': self.state['energy'],
            'hunger': self.state['hunger'],
            'thirst': self.state['thirst'],
            'mood': self.state['mood'],
            'position': self.position,
            'inventory': self.inventory,
            'skills': self.skills,
            'status_effects': self.status_effects
        }
    
    def initiate_communication(self, world: 'World', target_agent: str, message: str) -> None:
        """
        Initiate communication with another agent.
        
        Args:
            world: The simulation world
            target_agent: Name of the agent to communicate with
            message: Message to send
        """
        target_agent_obj = world.get_agent(target_agent)
        if not target_agent_obj:
            self.logger.warning(f"Target agent {target_agent} not found")
            return
            
        if not self._is_within_communication_range(target_agent_obj):
            self.logger.warning(f"{target_agent} is out of communication range")
            return
            
        try:
            # Send the message
            target_agent_obj.receive_communication(self.name, message)
            
            # Apply energy cost
            energy_cost = self.config.agents.behavior.energy_consumption_rate * 0.5  # Communication is less costly
            self.state['energy'] = max(0, self.state['energy'] - energy_cost)
            
            # Improve social skill
            social_skill_increase = self._calculate_skill_increase('social')
            self.skills['social'] += social_skill_increase
            
            # Update relationships positively
            self._update_relationship(target_agent, 0.1)
            
            self.logger.info(f"{self.name} sent message '{message}' to {target_agent}")
            
            # Record interaction in memory
            self._record_communication(target_agent, message, 'sent')
            
        except Exception as e:
            self.logger.error(f"Error during communication: {str(e)}")

    def receive_communication(self, sender_agent: str, message: str) -> None:
        """
        Handle incoming communication from another agent.
        
        Args:
            sender_agent: Name of the sending agent
            message: Received message
        """
        try:
            self.logger.info(f"{self.name} received message '{message}' from {sender_agent}")
            
            # Update relationship with sender
            relationship_change = 0.1 if 'friendly' in message.lower() else 0.05
            self._update_relationship(sender_agent, relationship_change)
            
            # Store the message for potential rendering
            self.state['last_message'] = message
            
            # Record interaction in memory
            self._record_communication(sender_agent, message, 'received')
            
        except Exception as e:
            self.logger.error(f"Error processing received communication: {str(e)}")

    def _is_within_communication_range(self, other_agent: 'Agent') -> bool:
        """
        Check if another agent is within communication range.
        
        Args:
            other_agent: The agent to check range to
            
        Returns:
            bool: True if within range, False otherwise
        """
        distance = self._calculate_distance(self.position, other_agent.position)
        return distance <= self.communication_range

    def _update_relationship(self, other_agent: str, change: float) -> None:
        """
        Update relationship value with another agent.
        
        Args:
            other_agent: Name of the other agent
            change: Amount to change relationship by (positive or negative)
        """
        current = self.relationships.get(other_agent, 0)
        self.relationships[other_agent] = max(-10, min(10, current + change))

    def _record_communication(self, other_agent: str, message: str, direction: str) -> None:
        """
        Record communication in agent's memory.
        
        Args:
            other_agent: Name of the other agent
            message: Content of the message
            direction: Either 'sent' or 'received'
        """
        if other_agent not in self.memory.interaction_history:
            self.memory.interaction_history[other_agent] = []
            
        self.memory.interaction_history[other_agent].append({
            'timestamp': time.time(),
            'message': message,
            'direction': direction,
            'position': self.position
        })
        
        # Limit history size
        if len(self.memory.interaction_history[other_agent]) > self.MAX_MEMORY_SIZE:
            self.memory.interaction_history[other_agent] = (
                self.memory.interaction_history[other_agent][-self.MAX_MEMORY_SIZE:]
            )

    def _calculate_skill_increase(self, skill_name: str) -> float:
        """
        Calculate skill increase amount based on agent traits and current level.
        
        Args:
            skill_name: Name of the skill to increase
            
        Returns:
            float: Amount to increase the skill by
        """
        base_increase = random.uniform(0.01, 0.05)
        
        # Apply trait modifiers
        if skill_name == 'gathering' and hasattr(self.behavior_traits, 'gathering_efficiency'):
            base_increase *= self.behavior_traits.gathering_efficiency
        elif skill_name == 'combat' and hasattr(self.behavior_traits, 'combat_skill'):
            base_increase *= self.behavior_traits.combat_skill
        elif skill_name == 'social' and hasattr(self.behavior_traits, 'intelligence'):
            base_increase *= self.behavior_traits.intelligence
            
        # Scale based on current skill level (harder to improve at higher levels)
        current_skill = self.skills.get(skill_name, 1.0)
        level_scaling = 1.0 / math.sqrt(current_skill)
        
        return base_increase * level_scaling

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate distance between two positions, considering world wrap-around if enabled.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            float: Distance between positions
        """
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        if self.config.environment.grid.wrap_around:
            dx = min(dx, self.config.environment.grid.width - dx)
            dy = min(dy, self.config.environment.grid.height - dy)

        return math.sqrt(dx ** 2 + dy ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to serializable dictionary."""
        return {
            'name': self.name,
            'position': self.position,
            'state': self.state,
            'inventory': self.inventory,
            'skills': self.skills,
            'status_effects': self.status_effects,
            'agent_type': self.agent_type.name,
            'behavior_traits': self.behavior_traits.to_dict(),
            'goals': self.goals,
            'relationships': self.relationships,
        }

    def __str__(self) -> str:
        """String representation of the agent."""
        return (f"Agent {self.name} (Level {self.state['level']} {self.agent_type.name}) "
                f"at {self.position} | Health: {self.state['health']}/100 | "
                f"Energy: {self.state['energy']}/100 | Mood: {self.state['mood']}")

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return self.__str__()
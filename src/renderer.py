# renderer.py

"""
Enhanced Renderer module for the little-matrix simulation.

This module defines the Renderer class, which handles the visual rendering of the simulation
using Pygame. It provides methods to initialize the display, render agents and objects,
and update the display in sync with the simulation loop with smooth animations and visual effects.
"""

import pygame
import logging
from typing import Tuple, Dict
from dataclasses import dataclass
from agents.agent import Agent
from environment.world import World
from environment.objects import WorldObject

@dataclass
class AnimationState:
    """Tracks animation state for smooth movement transitions."""
    start_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    progress: float = 0.0
    duration: int = 10  # frames

class Renderer:
    """Manages visual rendering of the little-matrix simulation using Pygame."""

    def __init__(self, world: World, cell_size: int = 20, fps: int = 60):
        """Initialize the Renderer instance."""
        self.world = world
        self.cell_size = cell_size
        self.width = world.width * cell_size
        self.height = world.height * cell_size
        self.fps = fps
        self.screen = None
        self.buffer = None
        self.clock = pygame.time.Clock()
        self.colors = self._initialize_colors()
        self.font = None
        self.debug_font = None
        self.logger = logging.getLogger(__name__)
        self.communications: Dict[str, str] = {}
        self.communication_duration = 100
        self.communication_timer: Dict[str, int] = {}
        self.animations: Dict[str, AnimationState] = {}
        self.debug_mode = False
        self.show_grid = True
        self._initialize_pygame()
        
    def _initialize_pygame(self):
        """Initialize Pygame with enhanced display settings."""
        pygame.init()
        pygame.display.set_caption("Little Matrix Simulation - Enhanced!")
        
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        
        self.buffer = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont('Arial', 14)
        self.debug_font = pygame.font.SysFont('Courier New', 12)
        
        self.logger.info("Enhanced Pygame renderer initialized.")

    def _initialize_colors(self) -> dict:
        """Initialize color mappings."""
        return {
            'AgentColors': {
                'default': (0, 255, 0),      # Default green
                'active': (0, 255, 100),     # Bright green
                'inactive': (0, 180, 0),     # Dark green
                'tired': (100, 180, 0),      # Yellow-green
                'low_health': (255, 100, 0), # Orange
                'critical': (255, 0, 0)      # Red
            },
            'Obstacle': (128, 128, 128),
            'Resource': (255, 215, 0),
            'Hazard': (255, 0, 0),
            'Collectible': (0, 0, 255),
            'Tool': (255, 165, 0),
            'TerrainFeature': (139, 69, 19),
            'Default': (255, 255, 255),
            'Grid': (50, 50, 50),
            'Text': (255, 255, 255)
        }

    def _get_agent_color(self, agent: Agent) -> Tuple[int, int, int]:
        """
        Determine agent color based on its state.
        
        Args:
            agent (Agent): The agent to get color for.
            
        Returns:
            Tuple[int, int, int]: RGB color tuple
        """
        color = self.colors['AgentColors']['default']
        
        if hasattr(agent, 'state'):
            if agent.state['energy'] < 30:
                color = self.colors['AgentColors']['tired']
            if agent.state['health'] < 30:
                color = self.colors['AgentColors']['low_health']
            elif agent.state['health'] < 10:
                color = self.colors['AgentColors']['critical']
            
            if agent.state['mood'] == 'energetic':
                color = self.colors['AgentColors']['active']
            elif agent.state['mood'] == 'exhausted':
                color = self.colors['AgentColors']['inactive']
        
        return color

    def render(self):
        """Render the current state of the world."""
        self.buffer .fill((0, 0, 0))  # Clear buffer

        if self.show_grid:
            self._draw_grid()

        # Draw static objects first
        for position, objects in self.world.objects.items():
            for obj in objects:
                self._draw_object(obj)

        # Draw agents with animations
        for agent in self.world.agents.values():
            self._handle_agent_animation(agent)
            self._draw_agent(agent)
            self._draw_agent_communication(agent)

        if self.debug_mode:
            self._draw_debug_info()

        # Swap buffer to screen
        self.screen.blit(self.buffer, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_grid(self):
        """Draw the background grid."""
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.buffer, self.colors['Grid'], 
                           (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.buffer, self.colors['Grid'],
                           (0, y), (self.width, y), 1)

    def _handle_agent_animation(self, agent: Agent):
        """Update animation state for smooth agent movement."""
        if agent.name not in self.animations:
            self.animations[agent.name] = AnimationState(
                agent.position,
                agent.position,
                1.0
            )

        anim = self.animations[agent.name]
        
        if agent.position != anim.target_pos:
            anim.start_pos = (
                anim.start_pos[0] + (anim.target_pos[0] - anim.start_pos[0]) * anim.progress,
                anim.start_pos[1] + (anim.target_pos[1] - anim.start_pos[1]) * anim.progress
            )
            anim.target_pos = agent.position
            anim.progress = 0.0

        if anim.progress < 1.0:
            anim.progress = min(1.0, anim.progress + 1.0 / anim.duration)

    def _draw_agent(self, agent: Agent):
        """Draw an agent with visual effects and state indicators."""
        anim = self.animations.get(agent.name)
        if not anim:
            # Fallback if no animation state exists
            x, y = agent.position
        else:
            x = anim.start_pos[0] + (anim.target_pos[0] - anim.start_pos[0]) * anim.progress
            y = anim.start_pos[1] + (anim.target_pos[1] - anim.start_pos[1]) * anim.progress
        
        rect = pygame.Rect(
            x * self.cell_size, 
            y * self.cell_size,
            self.cell_size ,
            self.cell_size
        )
        
        # Get agent color based on state
        color = self._get_agent_color(agent)
        pygame.draw.rect(self.buffer, color, rect)
        
        # Draw agent identifier
        text_surface = self.font.render(
            agent.name[0] if agent.name else "?",
            True,
            self.colors['Text']
        )
        self.buffer.blit(text_surface, rect.topleft)

    def _draw_object(self, obj: WorldObject):
        """Draw a world object."""
        x, y = obj.position
        rect = pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        
        object_type = type(obj).__name__
        color = self.colors.get(object_type, self.colors['Default'])
        
        pygame.draw.rect(self.buffer, color, rect)
        
        # Draw object symbol
        symbol = getattr(obj, 'symbol', object_type[0])
        text_surface = self.font.render(symbol, True, self.colors['Text'])
        self.buffer.blit(text_surface, rect.topleft)

    def _draw_agent_communication(self, agent: Agent):
        """Draw agent communication bubbles."""
        if agent.name in self.communications and agent.name in self.communication_timer:
            if self.communication_timer[agent.name] > 0:
                message = self.communications[agent.name]
                x, y = agent.position if not hasattr(agent, 'position') else agent.position
                
                text_surface = self.font.render(message, True, self.colors['Text'])
                text_rect = text_surface.get_rect()
                bubble_rect = text_rect.inflate(10, 10)
                bubble_rect.bottomleft = ((x + 1) * self.cell_size, (y - 0.5) * self.cell_size)
                
                pygame.draw.rect(self.buffer, (50, 50, 50), bubble_rect)
                pygame.draw.rect(self.buffer, self.colors['Text'], bubble_rect, 1)
                self.buffer.blit(text_surface, bubble_rect.inflate(-10, -10))
                
                self.communication_timer[agent.name] -= 1
            else:
                self.communications.pop(agent.name)
                self.communication_timer.pop(agent.name)

    def _draw_debug_info(self):
        """Draw debug information."""
        debug_info = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Agents: {len(self.world.agents)}",
            f"Objects: {sum(len(obj) for obj in self.world.objects.values())}"
        ]
        
        for i, info in enumerate(debug_info):
            text_surface = self.debug_font.render(
                info,
                True,
                self.colors['Text']
            )
            self.buffer.blit(text_surface, (5, 5 + i * 15))

    def toggle_debug(self):
        """Toggle debug information display."""
        self.debug_mode = not self.debug_mode

    def toggle_grid(self):
        """Toggle grid display."""
        self.show_grid = not self.show_grid

    def handle_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.logger.info("Pygame window closed by user.")
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.logger.info("Escape key pressed. Exiting simulation.")
                    return False
                elif event.key == pygame.K_d:
                    self.toggle_debug()
                elif event.key == pygame.K_g:
                    self.toggle_grid()
        return True

    def close(self):
        """Clean up resources and close the display."""
        pygame.quit()
        self.logger.info("Enhanced renderer closed.")

    def display_communication(self, agent: Agent, message: str):
        """Display a communication message for an agent."""
        self.communications[agent.name] = message
        self.communication_timer[agent.name] = self.communication_duration
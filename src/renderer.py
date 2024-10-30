# src/renderer/renderer.py

"""
Enhanced Renderer module for the little-matrix simulation.

This module defines the Renderer class, which handles the visual rendering of the simulation
using Pygame. It provides methods to initialize the display, render agents and objects,
and update the display in sync with the simulation loop with smooth animations and visual effects.
"""

import pygame
import logging
from typing import Tuple, Dict, List
from dataclasses import dataclass, field
from src.agents.agent import Agent
from src.environment.world import World
from src.environment.objects import WorldObject
from src.utils.config import Config

@dataclass
class AnimationState:
    """Tracks animation state for smooth movement transitions."""
    start_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    progress: float = 0.0
    duration: int = 10  # frames

class Renderer:
    """Manages visual rendering of the little-matrix simulation using Pygame."""

    def __init__(self, world: World, config: Config):
        """Initialize the Renderer instance."""
        self.world = world
        self.config = config
        self.cell_size = self._calculate_cell_size()
        self.width = world.width * self.cell_size
        self.height = world.height * self.cell_size
        self.fps = config.renderer.display.fps
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
        self.effects_enabled = config.renderer.effects
        self._initialize_pygame()

    def _calculate_cell_size(self) -> int:
        """Calculate the cell size based on display size and world dimensions."""
        display_width, display_height = self.config.renderer.display.size
        cell_width = display_width // self.world.width
        cell_height = display_height // self.world.height
        return min(cell_width, cell_height)

    def _initialize_pygame(self):
        """Initialize Pygame with enhanced display settings."""
        pygame.init()
        pygame.display.set_caption(self.config.simulation.name)

        flags = 0
        if self.config.renderer.display.fullscreen:
            flags |= pygame.FULLSCREEN
        if self.config.renderer.display.resizable:
            flags |= pygame.RESIZABLE

        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            flags
        )

        self.buffer = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont('Arial', int(self.cell_size * 0.6))
        self.debug_font = pygame.font.SysFont('Courier New', 12)

        self.logger.info("Enhanced Pygame renderer initialized.")

    def _initialize_colors(self) -> dict:
        """Initialize color mappings from configuration."""
        colors_config = self.config.renderer.colors
        return {
            'background': tuple(colors_config.background),
            'agent_default': tuple(colors_config.agent_default),
            'resource': tuple(colors_config.resource),
            'terrain': {k: tuple(v) for k, v in colors_config.terrain.items()},
            'environment_effects': {k: tuple(v) for k, v in colors_config.environment_effects.items()},
            'grid': (50, 50, 50),
            'text': (255, 255, 255)
        }

    def render(self):
        """Render the current state of the world."""
        self.buffer.fill(self.colors['background'])  # Clear buffer

        if self.show_grid:
            self._draw_grid()

        # Draw terrain
        self._draw_terrain()

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
            pygame.draw.line(self.buffer, self.colors['grid'],
                             (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.buffer, self.colors['grid'],
                             (0, y), (self.width, y), 1)

    def _draw_terrain(self):
        """Render the terrain grid with optimized color selection and tile drawing."""
        terrain_colors = self.colors.get('terrain', {})
        default_color = (139, 69, 19)  # Default to a brown color for undefined terrain types

        # Precompute rectangle size for tiles to reduce redundant calculations in the loop
        tile_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)

        for y in range(self.world.height):
            for x in range(self.world.width):
                terrain_feature = self.world.terrain[y, x]
                # Get color based on terrain type or fall back to default color
                color = terrain_colors.get(terrain_feature.name, default_color)

                # Adjust tile position without creating a new Rect each time
                tile_rect.topleft = (x * self.cell_size, y * self.cell_size)

                # Draw the terrain tile
                pygame.draw.rect(self.buffer, color, tile_rect)

    def _handle_agent_animation(self, agent: Agent):
        """Update animation state for smooth agent movement."""
        if agent.name not in self.animations:
            self.animations[agent.name] = AnimationState(
                start_pos=agent.position,
                target_pos=agent.position,
                progress=1.0
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
            self.cell_size,
            self.cell_size
        )

        # Get agent color based on type and state
        agent_type_config = next((t for t in self.config.agents.customization.types if t.name == agent.agent_type), None)
        if agent_type_config:
            color = tuple(agent_type_config.color)
        else:
            color = self.colors['agent_default']

        pygame.draw.rect(self.buffer, color, rect)

        # Draw agent symbol
        symbol = agent_type_config.symbol if agent_type_config else 'A'
        text_surface = self.font.render(symbol, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.buffer.blit(text_surface, text_rect)

        # Draw health bar if enabled
        if self.effects_enabled.show_health_bars:
            self._draw_health_bar(agent, rect)

    def _draw_health_bar(self, agent: Agent, rect: pygame.Rect):
        """Draw health bar above the agent."""
        health_percentage = agent.state['health'] / 100.0
        bar_width = rect.width
        bar_height = 5
        bar_x = rect.x
        bar_y = rect.y - bar_height - 2

        health_bar_rect = pygame.Rect(bar_x, bar_y, bar_width * health_percentage, bar_height)
        bg_bar_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)

        pygame.draw.rect(self.buffer, (255, 0, 0), bg_bar_rect)
        pygame.draw.rect(self.buffer, (0, 255, 0), health_bar_rect)

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
        color = self.colors.get(object_type.lower(), self.colors['resource'])

        pygame.draw.rect(self.buffer, color, rect)

        # Draw object symbol
        symbol = getattr(obj, 'symbol', object_type[0])
        text_surface = self.font.render(symbol, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=rect.center)
        self.buffer.blit(text_surface, text_rect)

    def _draw_agent_communication(self, agent: Agent):
        """Draw agent communication bubbles."""
        if agent.name in self.communications and agent.name in self.communication_timer:
            if self.communication_timer[agent.name] > 0:
                message = self.communications[agent.name]
                anim = self.animations.get(agent.name)
                if anim:
                    x = anim.start_pos[0] + (anim.target_pos[0] - anim.start_pos[0]) * anim.progress
                    y = anim.start_pos[1] + (anim.target_pos[1] - anim.start_pos[1]) * anim.progress
                else:
                    x, y = agent.position

                x_pix = x * self.cell_size
                y_pix = y * self.cell_size

                text_surface = self.font.render(message, True, self.colors['text'])
                text_rect = text_surface.get_rect()
                bubble_rect = text_rect.inflate(10, 10)
                bubble_rect.bottomleft = (x_pix + self.cell_size, y_pix - 5)

                pygame.draw.rect(self.buffer, (50, 50, 50), bubble_rect)
                pygame.draw.rect(self.buffer, self.colors['text'], bubble_rect, 1)
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
                self.colors['text']
            )
            self.buffer.blit(text_surface, (5, 5 + i * 15))

    def toggle_debug(self):
        """Toggle debug information display."""
        self.debug_mode = not self.debug_mode

    def toggle_grid(self):
        """Toggle grid display."""
        self.show_grid = not self.show_grid

    def handle_events(self) -> bool:
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

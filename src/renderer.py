# renderer.py

"""
Renderer module for the little-matrix simulation.

This module defines the Renderer class, which handles the visual rendering of the simulation
using Pygame. It provides methods to initialize the display, render agents and objects,
and update the display in sync with the simulation loop.

Classes:
    Renderer: Manages rendering the simulation using Pygame.
"""

import pygame
import sys
import logging
from typing import Tuple
from agents.agent import Agent
from environment.world import World
from environment.objects import WorldObject

class Renderer:
    """
    Manages visual rendering of the little-matrix simulation using Pygame.

    Attributes:
        world (World): The simulation world to render.
        cell_size (int): The size of each grid cell in pixels.
        screen (pygame.Surface): The Pygame display surface.
        clock (pygame.time.Clock): Pygame clock for controlling frame rate.
        colors (dict): Mapping of object types to colors.
        font (pygame.font.Font): Font used for rendering text.
    """

    def __init__(self, world: World, cell_size: int = 20):
        """
        Initializes the Renderer instance.

        Args:
            world (World): The simulation world to render.
            cell_size (int): The size of each grid cell in pixels.
        """
        self.world = world
        self.cell_size = cell_size
        self.width = world.width * cell_size
        self.height = world.height * cell_size
        self.screen = None
        self.clock = pygame.time.Clock()
        self.colors = self._initialize_colors()
        self.font = None
        self.logger = logging.getLogger(__name__)
        self._initialize_pygame()

    def _initialize_pygame(self):
        """
        Initializes Pygame and sets up the display.

        Returns:
            None
        """
        pygame.init()
        pygame.display.set_caption("Little Matrix Simulation")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont(None, 14)
        self.logger.info("Pygame initialized.")

    def _initialize_colors(self) -> dict:
        """
        Initializes the color mapping for different object types.

        Returns:
            dict: Mapping of object types to colors.
        """
        return {
            'Agent': (0, 255, 0),          # Green
            'Obstacle': (128, 128, 128),   # Gray
            'Resource': (255, 215, 0),     # Gold
            'Hazard': (255, 0, 0),         # Red
            'Collectible': (0, 0, 255),    # Blue
            'Tool': (255, 165, 0),         # Orange
            'TerrainFeature': (139, 69, 19),  # Brown
            'Default': (255, 255, 255),    # White
        }

    def render(self):
        """
        Renders the current state of the world to the display.

        Returns:
            None
        """
        self.screen.fill((0, 0, 0))  # Clear screen with black background

        # Render objects
        for position, objects in self.world.objects.items():
            x, y = position
            for obj in objects:
                self._draw_object(obj)

        # Render agents
        for agent in self.world.agents.values():
            self._draw_agent(agent)

        pygame.display.flip()
        self.clock.tick(60)  # Limit to 60 frames per second

    def _draw_agent(self, agent: Agent):
        """
        Draws an agent on the screen.

        Args:
            agent (Agent): The agent to draw.

        Returns:
            None
        """
        x, y = agent.position
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        color = self.colors.get('Agent', self.colors['Default'])
        pygame.draw.rect(self.screen, color, rect)

        # Optionally draw the agent's name or symbol
        text_surface = self.font.render('A', True, (255, 255, 255))
        self.screen.blit(text_surface, rect.topleft)

    def _draw_object(self, obj: WorldObject):
        """
        Draws an object on the screen.

        Args:
            obj (WorldObject): The object to draw.

        Returns:
            None
        """
        x, y = obj.position
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        object_type = type(obj).__name__
        color = self.colors.get(object_type, self.colors['Default'])
        pygame.draw.rect(self.screen, color, rect)

        # Optionally draw the object's symbol
        text_surface = self.font.render(obj.symbol, True, (0, 0, 0))
        self.screen.blit(text_surface, rect.topleft)

    def handle_events(self):
        """
        Handles Pygame events such as quitting the simulation.

        Returns:
            bool: True if the simulation should continue, False if it should exit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.logger.info("Pygame window closed by user.")
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.logger.info("Escape key pressed. Exiting simulation.")
                    pygame.quit()
                    sys.exit()
        return True

    def close(self):
        """
        Closes the Pygame display and quits Pygame.

        Returns:
            None
        """
        pygame.quit()
        self.logger.info("Pygame closed.")

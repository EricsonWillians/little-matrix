# src/data/storage.py

"""
Storage module for the little-matrix simulation.

This module defines the StorageManager class, which handles data persistence using SQLite.
It provides methods for saving and loading agent states, interaction histories, and other
simulation data.

Classes:
    StorageManager: Manages the SQLite database for data persistence.
"""

import sqlite3
import threading
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Self
from contextlib import contextmanager
from ..utils.config import Config
from ..agents.behaviors import AgentTypeBehaviorTraits

class StorageManager:
    """
    Manages data persistence using SQLite for the little-matrix simulation.
    """

    def __init__(self, config: Config):
        """
        Initializes the StorageManager instance.

        Args:
            config (Config): The configuration object loaded from config.yaml.
        """
        self.logger = logging.getLogger(__name__)

        # Load database file path and max connections from the configuration
        self.db_file: str = config.advanced.database_file or 'little_matrix.db'
        self.max_connections: int = config.advanced.performance.max_db_connections or 5

        self.connection_pool: List[sqlite3.Connection] = []
        self.lock = threading.Lock()

        self._initialize_connection_pool()
        self._initialize_database()

    def _initialize_connection_pool(self):
        """
        Initializes a pool of SQLite connections for thread-safe operations.
        """
        for _ in range(self.max_connections):
            conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.connection_pool.append(conn)
        self.logger.info(f"Initialized connection pool with {self.max_connections} connections.")

    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """
        Context manager to safely acquire and release a database connection from the pool.

        Yields:
            sqlite3.Connection: A database connection.
        """
        with self.lock:
            if not self.connection_pool:
                self.logger.warning("No available database connections. Creating a new one.")
                conn = sqlite3.connect(self.db_file, check_same_thread=False)
            else:
                conn = self.connection_pool.pop()
        try:
            yield conn
        finally:
            with self.lock:
                self.connection_pool.append(conn)

    def _initialize_database(self):
        """
        Creates the necessary tables in the database if they do not exist.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Create agents table, add behavior_traits as JSON text if missing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    state TEXT NOT NULL,
                    knowledge_base TEXT,
                    position_x INTEGER NOT NULL,
                    position_y INTEGER NOT NULL,
                    behavior_traits TEXT,  -- Ensure this column exists
                    agent_type TEXT
                )
            """)
            # Create interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            self.logger.info("Database tables initialized.")


    def save_agent_state(self, agent) -> None:
        """Save or update agent state with proper serialization."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Serialize agent data safely
                state_data = {
                    'health': agent.state['health'],
                    'energy': agent.state['energy'],
                    'hunger': agent.state['hunger'],
                    'thirst': agent.state['thirst'],
                    'mood': agent.state['mood'],
                    'experience': agent.state['experience'],
                    'level': agent.state['level'],
                    'action': agent.state['action'],
                    'action_details': agent.state['action_details'],
                    'perceived_agents': agent.state.get('perceived_agents', []),
                    'perceived_objects': agent.state.get('perceived_objects', []),
                    'last_message': agent.state.get('last_message', ''),
                    'strategy_cooldown': agent.state.get('strategy_cooldown', 0),
                }
                
                # Serialize behavior traits
                behavior_traits = {
                    'gathering_efficiency': getattr(agent.behavior_traits, 'gathering_efficiency', 1.0),
                    'combat_skill': getattr(agent.behavior_traits, 'combat_skill', 1.0),
                    'speed_modifier': getattr(agent.behavior_traits, 'speed_modifier', 1.0),
                    'recovery_rate': getattr(agent.behavior_traits, 'recovery_rate', 1.0),
                    'intelligence': getattr(agent.behavior_traits, 'intelligence', 1.0),
                    'risk_tolerance': getattr(agent.behavior_traits, 'risk_tolerance', 0.5),
                    'protective_instinct': getattr(agent.behavior_traits, 'protective_instinct', False),
                    'storage_capacity': getattr(agent.behavior_traits, 'storage_capacity', 1.0),
                    'perception_modifier': getattr(agent.behavior_traits, 'perception_modifier', 0.0)
                }

                # Convert everything to JSON-safe format
                serialized_data = {
                    'state': state_data,
                    'knowledge_base': agent.knowledge_base,
                    'inventory': agent.inventory,
                    'skills': agent.skills,
                    'goals': agent.goals,
                    'relationships': {str(k): float(v) for k, v in agent.relationships.items()},
                    'status_effects': agent.status_effects,
                    'agent_type': agent.agent_type.name if hasattr(agent.agent_type, 'name') else str(agent.agent_type),
                    'behavior_traits': behavior_traits
                }

                # Check if agent exists
                cursor.execute("SELECT id FROM agents WHERE name = ?", (agent.name,))
                result = cursor.fetchone()
                
                if result:
                    # Update existing agent
                    cursor.execute("""
                        UPDATE agents SET
                            state = ?,
                            knowledge_base = ?,
                            position_x = ?,
                            position_y = ?,
                            behavior_traits = ?,
                            agent_type = ?
                        WHERE name = ?
                    """, (
                        json.dumps(serialized_data['state']),
                        json.dumps(serialized_data['knowledge_base']),
                        agent.position[0],
                        agent.position[1],
                        json.dumps(serialized_data['behavior_traits']),
                        serialized_data['agent_type'],
                        agent.name
                    ))
                else:
                    # Insert new agent
                    cursor.execute("""
                        INSERT INTO agents (
                            name, state, knowledge_base, position_x, position_y, 
                            behavior_traits, agent_type
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        agent.name,
                        json.dumps(serialized_data['state']),
                        json.dumps(serialized_data['knowledge_base']),
                        agent.position[0],
                        agent.position[1],
                        json.dumps(serialized_data['behavior_traits']),
                        serialized_data['agent_type']
                    ))
                
                conn.commit()
                self.logger.debug(f"Agent '{agent.name}' state saved successfully")
                
        except Exception as e:
            self.logger.error(f"Error saving agent '{agent.name}' state: {str(e)}")
            raise

    def load_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Load agent state with proper deserialization."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT state, knowledge_base, position_x, position_y, 
                           behavior_traits, agent_type
                    FROM agents 
                    WHERE name = ?
                """, (agent_name,))
                
                result = cursor.fetchone()
                if not result:
                    self.logger.warning(f"Agent '{agent_name}' not found in database")
                    return None
                    
                state_str, kb_str, pos_x, pos_y, traits_str, agent_type = result
                
                # Deserialize all components
                state = json.loads(state_str)
                knowledge_base = json.loads(kb_str)
                behavior_traits = json.loads(traits_str) if traits_str else {}
                
                # Construct behavior traits instance
                behavior_traits_obj = AgentTypeBehaviorTraits(**behavior_traits)
                
                return {
                    'state': state,
                    'knowledge_base': knowledge_base,
                    'position': (pos_x, pos_y),
                    'behavior_traits': behavior_traits_obj,
                    'agent_type': agent_type
                }
                
        except Exception as e:
            self.logger.error(f"Error loading agent '{agent_name}' state: {str(e)}")
            return None

    def save_interaction(self, sender_name: str, recipient_name: str, message: str) -> None:
        """
        Saves an interaction between agents to the database.

        Args:
            sender_name (str): The name of the sending agent.
            recipient_name (str): The name of the receiving agent.
            message (str): The message content.

        Returns:
            None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO interactions (sender, recipient, message)
                    VALUES (?, ?, ?)
                """, (sender_name, recipient_name, message))
                conn.commit()
                self.logger.debug(f"Interaction saved: {sender_name} -> {recipient_name}: {message}")
        except Exception as e:
            self.logger.error(f"Error saving interaction from '{sender_name}' to '{recipient_name}': {e}")

    def get_agent_interactions(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all interactions involving a specific agent.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            List[Dict[str, Any]]: A list of interactions involving the agent.
        """
        interactions = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sender, recipient, message, timestamp
                    FROM interactions
                    WHERE sender = ? OR recipient = ?
                    ORDER BY timestamp ASC
                """, (agent_name, agent_name))
                rows = cursor.fetchall()
                for row in rows:
                    sender, recipient, message, timestamp = row
                    interactions.append({
                        'sender': sender,
                        'recipient': recipient,
                        'message': message,
                        'timestamp': timestamp
                    })
                self.logger.debug(f"Retrieved {len(interactions)} interactions for agent '{agent_name}'.")
        except Exception as e:
            self.logger.error(f"Error retrieving interactions for agent '{agent_name}': {e}")
        return interactions

    def save_simulation_state(self, timestep: int, agents: List[Any], world_state: Dict[str, Any]) -> None:
        """
        Saves the entire simulation state at a given timestep.

        Args:
            timestep (int): The current timestep of the simulation.
            agents (List[Any]): List of agent instances.
            world_state (Dict[str, Any]): The current state of the world.

        Returns:
            None
        """
        # Implement as needed, possibly saving to a separate table or file.
        pass

    def delete_agent_state(self, agent_name: str) -> None:
        """
        Deletes the agent's state from the database.

        Args:
            agent_name (str): The name of the agent to delete.

        Returns:
            None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM agents WHERE name = ?", (agent_name,))
                conn.commit()
                self.logger.info(f"Agent '{agent_name}' state deleted from the database.")
        except Exception as e:
            self.logger.error(f"Error deleting agent '{agent_name}' state: {e}")

    def save_world_state(self, agents: list, world: Self):
        """Saves the entire simulation state, including the world and agents."""
        try:
            simulation_state = {
                'world': self._serialize_world(world),
                'agents': [agent.to_dict() for agent in agents],
            }
            with open('simulation_state.json', 'w') as f:
                json.dump(simulation_state, f)
            self.logger.info("Simulation state saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save simulation state: {e}")


    def _serialize_world(self, world: Self) -> dict:
        """Serializes the world object to a dictionary."""
        return {
            'current_time': world.current_time,
            'objects': self._serialize_objects(world.objects),
            # Add other world attributes if necessary
        }

    def _serialize_objects(self, objects) -> dict:
        """Serializes world objects."""
        serialized_objects = {}
        for position, obj_list in objects.items():
            serialized_objects[str(position)] = [obj.to_dict() for obj in obj_list]
        return serialized_objects

    def close(self):
        """
        Closes all database connections in the pool.

        Returns:
            None
        """
        with self.lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()
            self.logger.info("All database connections have been closed.")

    def __del__(self):
        """
        Destructor to ensure the database connections are closed.
        """
        try:
            self.close()
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

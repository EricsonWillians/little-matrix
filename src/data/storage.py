# storage.py

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
from typing import Dict, Any, List

class StorageManager:
    """
    Manages data persistence using SQLite for the little-matrix simulation.

    Attributes:
        db_file (str): Path to the SQLite database file.
        connection (sqlite3.Connection): Connection object to the SQLite database.
        lock (threading.Lock): Ensures thread-safe database operations.
    """

    def __init__(self, db_file: str = "little_matrix.db"):
        """
        Initializes the StorageManager instance.

        Args:
            db_file (str): Path to the SQLite database file.
        """
        self.db_file = db_file
        self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._initialize_database()

    def _initialize_database(self):
        """
        Creates the necessary tables in the database if they do not exist.

        Returns:
            None
        """
        with self.lock:
            cursor = self.connection.cursor()
            # Create agents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    state TEXT NOT NULL,
                    knowledge_base TEXT,
                    position_x INTEGER NOT NULL,
                    position_y INTEGER NOT NULL
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
            self.connection.commit()

    def save_agent_state(self, agent):
        """
        Saves or updates the state of an agent in the database.

        Args:
            agent (Agent): The agent whose state is to be saved.

        Returns:
            None
        """
        with self.lock:
            cursor = self.connection.cursor()
            # Serialize state and knowledge_base to JSON strings
            state_str = json.dumps(agent.state)
            knowledge_base_str = json.dumps(agent.knowledge_base)
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
                        position_y = ?
                    WHERE name = ?
                """, (state_str, knowledge_base_str, agent.position[0], agent.position[1], agent.name))
            else:
                # Insert new agent
                cursor.execute("""
                    INSERT INTO agents (name, state, knowledge_base, position_x, position_y)
                    VALUES (?, ?, ?, ?, ?)
                """, (agent.name, state_str, knowledge_base_str, agent.position[0], agent.position[1]))
            self.connection.commit()

    def load_agent_state(self, agent_name: str) -> Dict[str, Any]:
        """
        Loads the state of an agent from the database.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            dict: A dictionary containing the agent's state, knowledge_base, and position.
        """
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT state, knowledge_base, position_x, position_y
                FROM agents WHERE name = ?
            """, (agent_name,))
            result = cursor.fetchone()
            if result:
                state_str, knowledge_base_str, position_x, position_y = result
                state = json.loads(state_str)  # Deserialize JSON string to dict
                knowledge_base = json.loads(knowledge_base_str)  # Deserialize JSON string to list
                position = (position_x, position_y)
                return {
                    'state': state,
                    'knowledge_base': knowledge_base,
                    'position': position
                }
            else:
                self.logger.warning(f"Agent '{agent_name}' not found in database.")
                return {}

    def save_interaction(self, sender_name: str, recipient_name: str, message: str):
        """
        Saves an interaction between agents to the database.

        Args:
            sender_name (str): The name of the sending agent.
            recipient_name (str): The name of the receiving agent.
            message (str): The message content.

        Returns:
            None
        """
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO interactions (sender, recipient, message)
                VALUES (?, ?, ?)
            """, (sender_name, recipient_name, message))
            self.connection.commit()

    def get_agent_interactions(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all interactions involving a specific agent.

        Args:
            agent_name (str): The name of the agent.

        Returns:
            List[dict]: A list of interactions involving the agent.
        """
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT sender, recipient, message, timestamp
                FROM interactions
                WHERE sender = ? OR recipient = ?
                ORDER BY timestamp ASC
            """, (agent_name, agent_name))
            rows = cursor.fetchall()
            interactions = []
            for row in rows:
                sender, recipient, message, timestamp = row
                interactions.append({
                    'sender': sender,
                    'recipient': recipient,
                    'message': message,
                    'timestamp': timestamp
                })
            return interactions

    def close(self):
        """
        Closes the database connection.

        Returns:
            None
        """
        with self.lock:
            self.connection.close()

    def __del__(self):
        """
        Destructor to ensure the database connection is closed.
        """
        try:
            self.close()
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")

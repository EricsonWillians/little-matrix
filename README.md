Here’s a polished and professional README for the **Little Matrix Simulation** project, considering all recent updates and features:

---

# Little Matrix Simulation

![Little Matrix Simulation Logo](src/assets/logo.webp)

Welcome to the **Little Matrix Simulation**, a sophisticated, autonomous agent-based simulation powered by advanced Language Learning Models (LLMs) from Hugging Face. This project demonstrates how agents perceive their environment, make intelligent decisions, and interact within a simulated world using natural language processing capabilities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The **Little Matrix Simulation** models a dynamic environment where multiple agents operate autonomously. Each agent leverages the Hugging Face Inference API to make informed decisions based on its state and perceptions of the environment. The simulation includes:

- **Autonomous Agents:** Intelligent entities capable of perception, decision-making, actions, and communication.
- **Dynamic Environment:** A grid-based world populated with various objects, resources, and terrain types.
- **Advanced Decision-Making:** Agents use LLMs to generate actions and communicate effectively.
- **Real-Time Visualization:** Pygame-based rendering for a live view of simulation dynamics.

This project demonstrates LLM integration with agent-based modeling, allowing for complex, interactive simulations.

## Features

- **Agent Perception:** Agents detect nearby entities and objects within their perception radius.
- **LLM-Powered Decisions:** Hugging Face's Phi-3 model supports reasoning, action selection, and language-based interactions.
- **Inter-Agent Communication:** Agents communicate using natural language, allowing for collaboration and complex interactions.
- **Resource Management:** Agents collect resources to maintain energy and health.
- **Real-Time Visualization:** A Pygame-based renderer provides a graphical simulation display.
- **Extensible Architecture:** Easily add new agent behaviors, object types, and environmental features.

## Architecture

The project is organized into several core components:

1. **LLM Client (`llm/client.py`):**
   - Interfaces with Hugging Face's Inference API.
   - Manages prompt generation, response handling, and secure API interactions.

2. **Prompt Manager (`llm/prompts.py`):**
   - Manages all prompt templates and dynamically generates prompts based on agent states.

3. **Agent (`agents/agent.py`):**
   - Represents individual agents within the simulation.
   - Handles perception, decision-making, action execution, and communication.

4. **Renderer (`renderer.py`):**
   - Uses Pygame for real-time simulation rendering.
   - Displays agents, objects, and environmental interactions.

5. **Environment (`environment/world.py` and `environment/objects.py`):**
   - Defines the simulation world and its objects.
   - Manages state and interactions of objects and agents.

6. **Main Simulation (`src/main.py`):**
   - Initializes and runs the main simulation loop.
   - Manages interactions between agents and the environment.

## Getting Started

Follow these steps to set up and run the **Little Matrix Simulation** on your local machine.

### Prerequisites

Ensure you have the following installed:

- **Python 3.8 or higher**
- **Poetry** (for dependency management)
- **Git** (for cloning the repository)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/little-matrix-simulation.git
   cd little-matrix-simulation
   ```

2. **Install Dependencies with Poetry**

   Ensure [Poetry](https://python-poetry.org/docs/#installation) is installed on your system.

   ```bash
   poetry install
   ```

3. **Activate the Virtual Environment**

   Use Poetry to activate the environment:

   ```bash
   poetry shell
   ```

### Configuration

1. **Hugging Face API Key**

   Obtain a Hugging Face API key by creating an account on [Hugging Face](https://huggingface.co/) and navigating to your account settings.

2. **Set Environment Variables**

   To securely store your API keys, set them as environment variables:

   ```bash
   export HUGGINGFACE_API_KEY="your_huggingface_api_key"
   export HUGGINGFACE_MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
   ```

   Alternatively, create a `.env` file in the project root with:

   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   HUGGINGFACE_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
   ```

   Use `python-dotenv` to load these variables if you prefer this method.

## Usage

To run the simulation, execute the main script with:

```bash
poetry run python src/main.py --render
```

### Command-Line Arguments

- `--render`: Launches the Pygame renderer to visualize the simulation.
- `--no-render`: Runs the simulation without rendering (ideal for headless environments).
- `--timesteps`: Specifies the number of simulation steps (default: 1000).

**Example:**

```bash
poetry run python src/main.py --render --timesteps 200
```

### Simulation Controls

- **Quit Simulation:** Close the Pygame window or press `Escape` to exit.

## Project Structure

```plaintext
├── config.yaml
├── LICENSE
├── little_matrix.db
├── poetry.lock
├── pyproject.toml
├── README.md
├── setup.py
├── src
│   ├── agents
│   │   ├── agent.py
│   │   ├── behaviors.py
│   │   ├── __init__.py
│   ├── assets
│   │   └── logo.webp
│   ├── data
│   │   └── storage.py
│   ├── environment
│   │   ├── objects.py
│   │   └── world.py
│   ├── llm
│   │   ├── client.py
│   │   └── prompts.py
│   ├── main.py
│   ├── renderer.py
│   └── utils
│       ├── config.py
│       └── logger.py
└── tests
    ├── test_agents.py
    ├── test_environment.py
    ├── test_llm.py
    └── test_utils.py
```

- **agents/**: Contains the `Agent` class and related behaviors.
- **environment/**: Defines the world and its objects.
- **llm/**: Manages LLM interactions and prompt templates.
- **renderer.py**: Handles the simulation's graphical rendering.
- **utils/**: Contains utility files, such as configurations and logging.
- **assets/**: Holds assets, like logos.
- **tests/**: Unit tests for various modules.

## Contributing

Contributions are highly encouraged! To contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your message here"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

   Describe your changes and submit for review.

### Guidelines

- **Code Quality:** Follow PEP 8 standards.
- **Documentation:** Ensure code is well-documented.
- **Testing:** Include tests for new features or fixes.
- **Issues:** Report bugs or request features via GitHub Issues.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- **[Hugging Face](https://huggingface.co/):** For LLM support.
- **[LangChain](https://github.com/hwchase17/langchain):** Facilitating LLM integrations.
- **[Pygame](https://www.pygame.org/):** For real-time visualization.
- **Community Contributors:** Thanks to all contributors who’ve helped improve this project.

---

For questions or support, reach out via [email](mailto:your.email@example.com) or open a GitHub issue.

---

## Sample Configuration (config.yaml)

```yaml
# =========================================
# 🌌 Little Matrix Simulation Configuration
# =========================================

# General Settings
simulation:
  name: "Little Matrix Ultimate Simulation"          # Simulation name
  description: "An expansive, highly configurable world where agents interact, evolve, and explore using advanced AI-driven behaviors and environmental dynamics."
  render: true                                       # Enable Pygame rendering for visuals
  timesteps: 1000                                    # Default number of steps per simulation run
  save_state_interval: 50                            # Interval at which the simulation state is saved
  seed: 42                                           # Random seed for reproducibility

# Environment Settings
environment:
  grid:
    width: 200                                       # Width of the simulation grid (much larger world)
    height: 200                                      # Height of the simulation grid (much larger world)
    wrap_around: true                                # Whether the grid wraps around edges (toroidal)
    terrain:
      types:
        - name: "Plains"
          symbol: "."
          movement_cost: 1
          color: [34, 139, 34]
        - name: "Forest"
          symbol: "F"
          movement_cost: 2
          color: [0, 100, 0]
        - name: "Mountain"
          symbol: "^"
          movement_cost: 3
          impassable: false
          color: [139, 137, 137]
        - name: "Water"
          symbol: "~"
          movement_cost: 9999
          impassable: true
          color: [28, 107, 160]
      distribution:
        Plains: 50                                   # Percentage of grid cells
        Forest: 30
        Mountain: 15
        Water: 5
  resource:
    spawn_rate: 0.02                                 # Probability of resources spawning
    max_resources: 500                               # Increased resource cap for larger environment
    regeneration_rate: 0.01                         # Slower regeneration to simulate scarcity
    types:
      - name: "Food"
        symbol: "*"
        color: [255, 215, 0]
        spawn_on: ["Plains", "Forest"]
        quantity_range: [10, 50]
      - name: "Water"
        symbol: "~"
        color: [28, 107, 160]
        spawn_on: ["Water"]
        quantity_range: [50, 100]
      - name: "Metal"
        symbol: "M"
        color: [192, 192, 192]
        spawn_on: ["Mountain"]
        quantity_range: [5, 20]
  weather:
    enabled: true
    change_interval: 50                              # Weather changes every 50 timesteps
    patterns:
      - name: "Sunny"
        duration_range: [30, 70]
        effects: []
      - name: "Rainy"
        duration_range: [20, 50]
        effects:
          - type: "movement_cost_modifier"
            value: 1.1                               # Increase movement cost by 10%
          - type: "resource_regeneration_modifier"
            value: 1.2                               # Increase resource regeneration by 20%
      - name: "Foggy"
        duration_range: [10, 30]
        effects:
          - type: "visibility_reduction"
            value: 0.5                               # Reduce visibility by 50%
      - name: "Stormy"
        duration_range: [5, 15]
        effects:
          - type: "movement_cost_modifier"
            value: 1.5                               # Increase movement cost by 50%
          - type: "agent_energy_cost"
            value: 1.5                               # Increase agent energy cost by 50%
          - type: "hazard_spawn_rate"
            value: 0.1                               # Increase hazard spawn rate
    transitions:
      Sunny:
        Rainy: 0.2
        Foggy: 0.1
        Stormy: 0.05
        Sunny: 0.65
      Rainy:
        Sunny: 0.3
        Foggy: 0.2
        Stormy: 0.1
        Rainy: 0.4
      Foggy:
        Sunny: 0.4
        Rainy: 0.3
        Foggy: 0.2
        Stormy: 0.1
      Stormy:
        Sunny: 0.5
        Rainy: 0.3
        Foggy: 0.1
        Stormy: 0.1

llm:
  api_key: "YOUR_API_KEY_HERE"
  model: "microsoft/Phi-3-mini-4k-instruct"

# Agent Settings
agents:
  count: 10                                         # Increased number of agents for a much bigger world
  perception:
    base_radius: 6                                   # Base perception radius
    modifiers:
      'Explorer': 2                                  # Explorers have increased perception
      'Defender': -1                                 # Defenders have decreased perception
  sight_range:
    base_range: 8
    modifiers:
      'Explorer': 2
      'Gatherer': 0
      'Defender': -2
  behavior:
    initial_energy: 200                              # Higher energy to allow more exploration
    communication_enabled: true                      # Enable inter-agent communication for complex interactions
    aggressiveness_base: 0.1                         # Base aggressiveness
    aggressiveness_modifiers:
      'Defender': 0.2                                # Defenders are more aggressive
      'Explorer': 0.05                               # Explorers are less aggressive
    energy_consumption_rate: 1.0                     # Base energy cost per action
  customization:
    types:
      - name: "Explorer"
        color: [0, 128, 255]                         # Blue for explorers
        symbol: "E"
        behavior_traits:
          resource_preference: "knowledge"           # Prefer resources related to "knowledge"
          risk_tolerance: 0.8                        # High risk tolerance for exploration
          speed_modifier: 1.2                        # Move faster
          intelligence: 1.5                          # Higher intelligence
      - name: "Gatherer"
        color: [255, 215, 0]                         # Gold for gatherers
        symbol: "G"
        behavior_traits:
          resource_preference: "food"                # Prefer food-based resources
          risk_tolerance: 0.3                        # Low risk tolerance
          gathering_efficiency: 1.5                  # Better at gathering
          storage_capacity: 2.0                      # Larger inventory
      - name: "Defender"
        color: [255, 0, 0]                           # Red for defenders
        symbol: "D"
        behavior_traits:
          protective_instinct: true                  # Guards resources and other agents
          risk_tolerance: 0.6                        # Moderate risk tolerance
          combat_skill: 1.5                          # Better at combat
          armor: 1.2                                 # More resistant to damage
    status_effects:
      enabled: true
      effects:
        - name: "Poisoned"
          duration: [5, 15]
          health_decrease_per_tick: 2
        - name: "Exhausted"
          duration: [10, 20]
          energy_recovery_modifier: 0.5
        - name: "Inspired"
          duration: [5, 10]
          action_success_modifier: 1.2
    social:
      friendships:
        formation_probability: 0.05
        decay_rate: 0.01
      rivalries:
        formation_probability: 0.03
        aggression_multiplier: 1.5
    learning:
      enabled: true
      methods:
        - name: "Reinforcement Learning"
          parameters:
            learning_rate: 0.1
        - name: "Imitation Learning"
          parameters:
            imitation_probability: 0.2

# Renderer (Pygame) Configuration
renderer:
  display:
    size: [1600, 1200]                               # Larger display size for a huge world view
    fps: 60                                          # Smooth frame rate for rendering
    fullscreen: false                                # Option to run in fullscreen mode
    resizable: true                                  # Allow window resizing
  colors:
    background: [10, 10, 40]                         # Darker background for contrast
    agent_default: [255, 255, 255]                   # Default white color for agents without specific type
    resource: [34, 139, 34]                          # Green for resources
    terrain:
      Plains: [50, 205, 50]
      Forest: [34, 139, 34]
      Mountain: [139, 137, 137]
      Water: [28, 107, 160]
    environment_effects:
      fog: [200, 200, 200, 180]                      # Dense fog overlay
      rain: [70, 130, 180, 180]                      # Dark rain overlay for heavy weather
      storm: [75, 0, 130, 150]                       # Stormy purple for dramatic effect
  effects:
    show_shadows: true                               # Enable shadows for a more realistic feel
    particle_effects: true                           # Enable particle effects for agents and weather
    weather_effects: true                            # Toggle for weather visualization
    smooth_movement: true                            # Smooth animations for agent movement
    display_hud: true                                # Display Heads-Up Display with simulation stats
    show_agent_names: false                          # Option to display agent names
    show_health_bars: true                           # Display health bars above agents

# Logging Configuration
logging:
  level: "INFO"                                      # Set log level (DEBUG, INFO, WARNING, ERROR)
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "simulation.log"                             # Log file output, or leave blank for console-only
  rotation:
    enabled: true
    max_bytes: 10485760                              # 10 MB per log file
    backup_count: 5                                  # Keep up to 5 backup log files

# Advanced Settings (Optional)
advanced:
  performance:
    max_agent_threads: 16                            # Increased threads for processing in huge world
    tick_rate: 0.05                                  # Delay between ticks for pacing
    use_multiprocessing: true                        # Enable multiprocessing for better performance
  experiment_mode:
    enabled: true                                    # Enable experimental features
    data_collection: true                            # Collect agent behavior data
    save_interval: 50                                # Interval to save collected data (in timesteps)
    data_fields:
      - "agent_positions"
      - "agent_states"
      - "resource_levels"
      - "weather_conditions"
  ai_debugging:
    enabled: true                                     # AI diagnostics for testing agent decisions
    log_decisions: true                               # Log detailed agent decisions and actions
    visualize_decision_trees: true                    # Generate visualizations of agent decision trees
    profiling:
      enabled: true
      output_file: "profiling.log"
  compatibility_mode:
    enabled: false                                    # Enable if running on older hardware
    reduce_graphics: true                             # Reduce graphics quality for performance
    limit_agents: 50                                  # Limit number of agents for performance

```

Happy Simulating!
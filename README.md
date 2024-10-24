# Little Matrix Simulation

![Little Matrix Simulation Logo](assets/logo.png)

Welcome to the **Little Matrix Simulation**, a sophisticated and autonomous agent-based simulation powered by advanced Language Learning Models (LLMs) from Hugging Face. This project demonstrates how agents can perceive their environment, make intelligent decisions, and interact within a simulated world using natural language processing capabilities.

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

The **Little Matrix Simulation** is designed to model a dynamic environment where multiple agents operate autonomously. Each agent leverages the Hugging Face Inference API to make informed decisions based on its current state and perceptions of the environment. The simulation includes:

- **Autonomous Agents:** Intelligent entities that can perceive, decide, act, and communicate.
- **Dynamic Environment:** A grid-based world populated with various objects and resources.
- **Advanced Decision-Making:** Agents utilize LLMs to generate actions and communicate effectively.
- **Visual Rendering:** Real-time visualization of the simulation using Pygame.

This project showcases the integration of LLMs with agent-based modeling to create complex, interactive simulations.

## Features

- **Agent Perception:** Agents can detect nearby agents and objects within their perception radius.
- **LLM-Powered Decision Making:** Leveraging Hugging Face's Phi-3 model for advanced reasoning and action selection.
- **Inter-Agent Communication:** Agents can communicate with each other using natural language.
- **Resource Management:** Agents can collect resources to maintain their energy and health.
- **Real-Time Visualization:** Monitor the simulation through a Pygame-based renderer.
- **Extensible Architecture:** Easily add new agent behaviors, object types, and environmental features.

## Architecture

The project is structured into several key components:

1. **LLM Client (`llm/client.py`):**
   - Interfaces with the Hugging Face Inference API.
   - Manages prompt generation and response handling.
   - Ensures secure API interactions.

2. **Prompt Manager (`llm/prompts.py`):**
   - Stores and manages all prompt templates.
   - Facilitates dynamic prompt generation based on agent states.

3. **Agent (`agents/agent.py`):**
   - Represents individual agents in the simulation.
   - Handles perception, decision-making, action execution, and communication.

4. **Renderer (`renderer.py`):**
   - Utilizes Pygame to visually render the simulation.
   - Displays agents, objects, and their interactions in real-time.

5. **Environment (`environment/world.py` and `environment/objects.py`):**
   - Defines the simulation world and various objects within it.
   - Manages the state and interactions of objects and agents.

6. **Main Simulation (`src/main.py`):**
   - Initializes and runs the simulation loop.
   - Coordinates interactions between agents and the environment.

## Getting Started

Follow these instructions to set up and run the **Little Matrix Simulation** on your local machine.

### Prerequisites

Ensure you have the following installed:

- **Python 3.8 or higher**
- **Pip** (Python package installer)
- **Git** (for cloning the repository)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/little-matrix-simulation.git
   cd little-matrix-simulation
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Note:** Ensure that `requirements.txt` includes all necessary packages such as `pygame`, `huggingface_hub`, `langchain`, etc.

### Configuration

1. **Hugging Face API Key**

   Obtain your Hugging Face API key by creating an account on [Hugging Face](https://huggingface.co/) and navigating to your account settings.

2. **Set Environment Variables**

   It's essential to keep your API keys secure. Set them as environment variables:

   ```bash
   export HUGGINGFACE_API_KEY="your_huggingface_api_key"
   export HUGGINGFACE_MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
   ```

   **On Windows:**

   ```cmd
   set HUGGINGFACE_API_KEY=your_huggingface_api_key
   set HUGGINGFACE_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
   ```

   **Alternatively**, create a `.env` file in the project root and add:

   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   HUGGINGFACE_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
   ```

   Ensure to load these variables in your application using packages like `python-dotenv` if you choose this method.

## Usage

Run the simulation using the main script:

```bash
python3 src/main.py --render
```

### Command-Line Arguments

- `--render`: Launches the Pygame renderer to visualize the simulation.
- `--no-render`: Runs the simulation without rendering (useful for headless environments).
- `--timesteps`: Specifies the number of simulation steps (default: 100).

**Example:**

```bash
python3 src/main.py --render --timesteps 200
```

### Simulation Controls

- **Quit Simulation:** Close the Pygame window or press the `Escape` key to exit.

## Project Structure

```
little-matrix-simulation/
├── agents/
│   └── agent.py
├── environment/
│   ├── objects.py
│   └── world.py
├── llm/
│   ├── client.py
│   └── prompts.py
├── src/
│   └── main.py
├── assets/
│   └── logo.png
├── renderer.py
├── requirements.txt
├── README.md
└── .env
```

- **agents/**: Contains the `Agent` class definition.
- **environment/**: Defines the simulation world and various objects.
- **llm/**: Manages LLM interactions and prompt templates.
- **src/**: Entry point for running the simulation.
- **assets/**: Contains assets like logos and images.
- **renderer.py**: Handles the visualization of the simulation.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation.
- **.env**: Stores environment variables (optional).

## Contributing

Contributions are welcome! Follow these steps to contribute:

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

   Describe your changes and submit the pull request for review.

### Guidelines

- **Code Quality:** Ensure your code follows PEP 8 standards.
- **Documentation:** Update or add documentation where necessary.
- **Testing:** Include tests for new features or bug fixes.
- **Issues:** Report any bugs or feature requests via GitHub Issues.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- **[Hugging Face](https://huggingface.co/):** For providing powerful LLMs and the Inference API.
- **[LangChain](https://github.com/hwchase17/langchain):** For facilitating the integration with LLMs.
- **[Pygame](https://www.pygame.org/):** For enabling real-time visualization of the simulation.
- **Community Contributors:** Special thanks to users on forums who helped troubleshoot issues and improve the project.

---

Feel free to reach out via [email](mailto:your.email@example.com) or open an issue on GitHub if you have any questions or need further assistance.

Happy Simulating!
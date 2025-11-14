# Flappy-Bird-with-AI

![flappy (1)](https://github.com/user-attachments/assets/b48163fb-e543-470a-9e6c-4a45abe5957e)


A compact Flappy Bird clone where a simple neuroevolution setup trains agents to play. The project mixes a lightweight Pygame front-end with a minimal genetic algorithm and tiny feed‑forward neural agents — useful as a learning toy or baseline for experiments in evolving controllers.

## Highlights
- Agents are small neural networks (configurable layers) evolved over generations.
- Simulation runs entire populations and breeds the next generation based on fitness.
- Basic rendering with Pygame: background, pipes, scrolling floor and multiple birds (limited render cap).
- Best brain is exported to JSON when a new record is found.

## Quick requirements
- Python 3.8+
- numpy
- pygame

Install with:
pip install numpy pygame

## How to run
From the project root:
python main.py

The simulation runs continuously and evolves the agent population. Close the window or use Ctrl+C to stop.

## Project layout (important files)
- Flappy Bird with RL/main.py — orchestrates simulation and evolution loop
- game_objects/bird.py — bird physics, animation, rendering
- game_objects/obstacle.py — pipe/tube behavior and collision
- game_objects/ground.py — scrolling floor
- game_objects/agent.py — neural agent, mutation, decision logic
- game_objects/assets — sprites used by the game

## Notes & tips
- Configurable genetic params live in main.py (population size, mutation rates, hidden layer sizes).
- Rendering is limited to a small number of active birds for performance; fitness is tracked per agent.
- Exported champion_brain.json contains layer sizes, weights and biases for later analysis or reuse.

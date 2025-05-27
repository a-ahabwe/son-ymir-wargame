# Veto Game

A tactical exploration game with AI and human veto capabilities for studying human-AI collaboration.

## Overview

This project implements a tactical exploration game where an AI agent navigates a procedurally generated environment, but allows human intervention through a veto mechanism. The goal is to provide a research platform for studying effective human-AI collaboration in high-stakes, time-constrained decision environments.

### Research Questions

This project explores:
1. How do different veto mechanism designs affect task performance and trust?
2. What uncertainty representation methods most effectively support human intervention decisions?
3. When does human intervention improve vs. degrade AI performance?

## Key Features

- **Tactical Exploration Game**: Grid-based environment with various terrain types, resources, and enemies
- **Deep Reinforcement Learning Agent**: Uses Dueling DQN architecture for decision making
- **Human Veto Mechanism**: Allows humans to approve or veto AI actions
- **Uncertainty Estimation**: Multiple methods for estimating AI uncertainty
- **Experiment Framework**: Tools for running controlled experiments and collecting data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/a-ahabwe/son-ymir-wargame.git
cd son-ymir-wargame
chmod +x run_improved_game.sh
./run_improved_game.sh
```
Or if you are using Windows
```bash
git clone https://github.com/a-ahabwe/son-ymir-wargame.git
cd son-ymir-wargame
bash run_improved_game.sh
```

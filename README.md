# Deep Reinforcement Learning Experimentation Platform

An interactive reinforcement learning platform built with **Python, PyTorch, Gym, and Streamlit** for training, monitoring, and experimenting with deep reinforcement learning algorithms across discrete- and continuous-action environments.

The platform provides a unified interface for configuring RL hyperparameters, running training experiments, and visualizing episode rewards through a Streamlit web application.

---

## Overview

Reinforcement learning algorithms differ in their learning mechanisms, network architectures, exploration strategies, optimization procedures, and supported action spaces.

This project provides a common experimentation framework for implementing and experimenting with multiple deep reinforcement learning algorithms while monitoring their training behavior interactively.

### Supported Algorithms

- **DQN** — Deep Q-Network
- **A2C** — Advantage Actor-Critic
- **PPO** — Proximal Policy Optimization
- **DDPG** — Deep Deterministic Policy Gradient
- **TD3** — Twin Delayed Deep Deterministic Policy Gradient

The implementations use **PyTorch** for neural-network models and optimization.

---

## Features

- Interactive **Streamlit** interface
- Configurable reinforcement learning hyperparameters
- Support for discrete and continuous action spaces
- Multiple deep reinforcement learning algorithms
- Neural-network-based policy and value models
- Experience replay for off-policy algorithms
- Target networks for stable value-based learning
- Training reward monitoring
- Episode-level training visualization
- Modern Gym environment compatibility
- Modular agent, model, and utility implementations

---

## Supported Environments

The platform includes support for environments such as:

- `CartPole-v1`
- `Pendulum-v1`
- `LunarLander-v3`
- `LunarLanderContinuous-v3`
- `BipedalWalker-v3`

These environments provide both discrete-action and continuous-action reinforcement learning tasks.

---

## Project Architecture

```text
                         Streamlit Interface
                                  |
                                  v
                           Training Engine
                                  |
             +--------------------+--------------------+
             |         |           |          |         |
             v         v           v          v         v
            DQN       A2C         PPO        DDPG      TD3
             |         |           |          |         |
             +---------+-----------+----------+---------+
                                  |
                                  v
                             PyTorch Models
                                  |
                                  v
                          Gym Environments
                                  |
                                  v
                        Training / Evaluation
                                  |
                                  v
                         Reward Visualization




Algorithms
DQN

Deep Q-Network approximates the action-value function using a neural network.

The implementation supports:

Experience replay
Target-network-based learning
Discounted Bellman targets
Epsilon-based action selection

DQN is primarily used with discrete-action environments such as CartPole-v1.

A2C

Advantage Actor-Critic combines:

A policy network (Actor)
A value network (Critic)

The actor learns the policy while the critic estimates state values. The advantage signal is used to guide policy updates.

PPO

Proximal Policy Optimization performs policy-gradient updates while constraining policy changes through a clipped objective.

The implementation includes:

Policy/value networks
Advantage-based updates
Clipped policy objective
Value-function optimization
Trajectory collection

PPO can be applied to discrete and continuous control tasks depending on the environment and policy architecture.

DDPG

Deep Deterministic Policy Gradient is an actor-critic algorithm designed for continuous action spaces.

The implementation uses:

Actor network
Critic network
Target actor
Target critic
Experience replay
Exploration noise

A representative environment is:

Pendulum-v1
TD3

Twin Delayed Deep Deterministic Policy Gradient extends DDPG using techniques designed to reduce value overestimation and improve training stability.

The implementation includes:

Twin critic networks
Target policy smoothing
Delayed policy updates
Target networks
Experience replay
Technology Stack
Component	Technology
Programming Language	Python
Deep Learning	PyTorch
Reinforcement Learning Environments	Gym
Web Interface	Streamlit
Numerical Computing	NumPy
Data Processing	Pandas
Experimentation	Jupyter Notebook
Version Control	Git / GitHub
Project Structure
reinforcement-learning-platform/
│
├── app.py                    # Streamlit application
├── agent.py                  # RL agent implementations
├── model.py                  # Neural-network architectures
├── utils.py                  # Replay buffers and training utilities
│
├── notebook/                 # Development and experimentation notebooks
├── ppo_play_boxing.ipynb     # PPO experimentation notebook
│
├── img/
│   └── demo.gif              # Application demonstration
│
├── README.md
└── .gitignore
Installation
1. Clone the repository
git clone https://github.com/Tushar2021s/reinforcement-learning-platform.git
cd reinforcement-learning-platform
2. Create a virtual environment
python3 -m venv .venv

Activate it on macOS/Linux:

source .venv/bin/activate
3. Install dependencies
pip install --upgrade pip
pip install numpy==1.26.4
pip install torch gym streamlit pandas matplotlib
4. Start the application
streamlit run app.py

The application will be available at:

http://localhost:8501
Using the Platform
Start the Streamlit application.
Select an RL algorithm from the sidebar.
Select an environment.
Configure the available hyperparameters.
Start training.
Monitor episode rewards through the Streamlit interface.

Example workflow:

Select Environment
       |
       v
Select RL Algorithm
       |
       v
Configure Hyperparameters
       |
       v
Start Training
       |
       v
Collect Experiences
       |
       v
Update Neural Networks
       |
       v
Monitor Episode Rewards
Example Experiments
Discrete Control

CartPole-v1 can be used to experiment with:

DQN
A2C
PPO
Continuous Control

Pendulum-v1 can be used to experiment with:

DDPG
TD3

These experiments provide a basis for studying:

Learning dynamics
Exploration strategies
Policy optimization
Value estimation
Convergence behavior
Hyperparameter sensitivity
Research and Experimentation

The platform is designed as an experimentation framework rather than only a training script.

Potential experimental dimensions include:

Hyperparameter Studies
Learning rate
Discount factor
Batch size
Replay-buffer size
Exploration parameters
Policy-update parameters
Algorithmic Comparison

Different algorithms can be evaluated using common environments and training configurations.

Relevant measurements include:

Episode reward
Rolling average reward
Training stability
Convergence behavior
Sample efficiency
Training time

For reproducible comparisons, experiments should use consistent environment configurations, random seeds, training budgets, and evaluation procedures.

Compatibility Improvements

The original project was developed around older versions of the Gym ecosystem.

The current version has been updated to work with newer environment APIs, including:

Updated environment versions such as CartPole-v1 and Pendulum-v1
Modern environment reset handling
Modern five-value step() API handling
terminated / truncated episode handling
Modern rendering through render_mode="rgb_array"
Updated Streamlit chart APIs
NumPy/Gym compatibility adjustments

These changes allow the platform to run successfully with the current development environment used for this project.

Demo

The Streamlit application provides an interactive interface for selecting algorithms, environments, hyperparameters, and monitoring training.

Future Work

Potential extensions include:

Automated hyperparameter sweeps
Experiment configuration files
Reproducible random seeds
Training-result persistence
Algorithm comparison dashboards
TensorBoard integration
Additional reinforcement learning environments
Multi-run statistical evaluation
Confidence intervals for reward curves
Automated experiment reports
Distributed training experiments
Learning Objectives

This project provides practical experience with:

Reinforcement learning
Deep neural networks
Policy-gradient methods
Actor-critic architectures
Value-function approximation
Experience replay
Target networks
Optimization
Hyperparameter experimentation
Numerical experimentation
Python-based ML development
Interactive ML prototyping with Streamlit
# UTPB-COSC-6389-Project2: Neural Network Visualizer

## Overview
This project implements a neural network from scratch for Project 2 of the Computational Biomimicry class. The application generates, trains, and visualizes networks in real time, allowing users to load datasets, configure hidden layers, and select activation functions.

## Features
1. **Dynamic Network Generation**:
   - Creates networks based on input/output dimensions from datasets.
   - Configurable hidden layers and neuron counts.

2. **Activation Functions**:
   - User-selectable: **Sigmoid**, **Tanh**, **ReLU**.

3. **Real-Time Visualization**:
   - Displays network structure and updates weights dynamically during training.

4. **Training and Loss Plot**:
   - Implements forward/backward propagation and plots training loss upon completion.

## How to Use
1. **Run**: `python neural_network_ui.py`
2. **Load Dataset**: Select a `.csv` file with inputs and target values.
3. **Train**: Click **"Start Training"** and observe the real-time updates.
4. **Save Visualization**: Option to save the final network diagram.

## Constraints
- Neural network functionality (training, propagation) is implemented without libraries.
- The application dynamically visualizes and updates weights during training.
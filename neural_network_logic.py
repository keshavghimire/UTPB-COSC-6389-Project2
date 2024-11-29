import numpy as np
import pandas as pd


# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.weights = []
        self.biases = []
        self.a = []
        self.activations = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "tanh": (tanh, tanh_derivative),
            "relu": (relu, relu_derivative),
        }
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights and biases using Xavier or He initialization.
        """
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            # He Initialization for ReLU (better for deeper networks)
            limit = np.sqrt(2 / layers[i])
            self.weights.append(np.random.uniform(-limit, limit, (layers[i], layers[i + 1])))
            self.biases.append(np.zeros(layers[i + 1]))
            self.a.append(np.zeros(layers[i]))
        self.a.append(np.zeros(layers[-1]))

    def forward(self, x):
        self.a[0] = x
        activation_func = self.activations[self.activation][0]
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.a[i + 1] = activation_func(z)
        return self.a[-1]

    def backward(self, x, y, learning_rate):
        """
        Perform backpropagation with gradient clipping.

        Args:
            x: Input data.
            y: Target data.
            learning_rate: Learning rate for gradient descent.
        """
        m = x.shape[0]
        derivative_func = self.activations[self.activation][1]
        delta = self.a[-1] - y

        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.a[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            # Clip gradients to prevent exploding values
            clip_value = 1.0
            dW = np.clip(dW, -clip_value, clip_value)
            db = np.clip(db, -clip_value, clip_value)

            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            # Log gradients for debugging
            print(f"Layer {i + 1}, Gradient dW Mean: {np.mean(dW):.6f}, Bias Gradient Mean: {np.mean(db):.6f}")

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * derivative_func(self.a[i])

    def train(self, x, y, epochs, learning_rate, update_callback=None):
        self.losses = []
        for epoch in range(epochs):
            output = self.forward(x)
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            self.backward(x, y, learning_rate)

            # Call visualization callback at intervals
            if update_callback and (epoch % 10 == 0 or epoch == epochs - 1):
                update_callback()

            # Log training progress to the console every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    def plot_loss(self):
        """
        Plot the training loss over epochs.
        """
        import matplotlib.pyplot as plt

        if hasattr(self, 'losses') and self.losses:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.losses) + 1), self.losses, label="Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Epochs")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No loss data available to plot. Train the network first.")



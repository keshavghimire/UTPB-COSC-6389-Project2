import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.random.randn(layers[i + 1]) * 0.1)
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
        m = x.shape[0]
        derivative_func = self.activations[self.activation][1]

        delta = self.a[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.a[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * derivative_func(self.a[i])

    def train(self, x, y, epochs, learning_rate, update_callback=None):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            if update_callback:
                update_callback()
            if epoch % 100 == 0:
                loss = np.mean((self.a[-1] - y) ** 2)  # MSE for regression
                print(f"Epoch {epoch}, Loss: {loss}")

class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualizer")
        self.create_ui()

    def create_ui(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Number of Inputs:").grid(row=0, column=0)
        self.input_size = tk.IntVar(value=2)
        ttk.Entry(frame, textvariable=self.input_size).grid(row=0, column=1)

        ttk.Label(frame, text="Hidden Layers (comma-separated):").grid(row=1, column=0)
        self.hidden_layers = tk.StringVar(value="4,4")
        ttk.Entry(frame, textvariable=self.hidden_layers).grid(row=1, column=1)

        ttk.Label(frame, text="Number of Outputs:").grid(row=2, column=0)
        self.output_size = tk.IntVar(value=1)
        ttk.Entry(frame, textvariable=self.output_size).grid(row=2, column=1)

        ttk.Label(frame, text="Activation Function:").grid(row=3, column=0)
        self.activation = tk.StringVar(value="sigmoid")
        ttk.Combobox(frame, textvariable=self.activation, values=["sigmoid", "tanh", "relu"]).grid(row=3, column=1)

        ttk.Button(frame, text="Generate Network", command=self.generate_network).grid(row=4, column=0, padx=5)
        self.start_training_button = ttk.Button(frame, text="Start Training", command=self.start_training_with_dataset)
        self.start_training_button.grid(row=4, column=1, padx=5)
        ttk.Button(frame, text="Save Visualization", command=self.save_visualization).grid(row=4, column=2, padx=5)
        ttk.Button(frame, text="Load Dataset", command=self.load_dataset).grid(row=4, column=3, padx=5)

        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)
            X = data.drop(columns=["quality"]).values  # All columns except the 'quality' column
            y = data["quality"].values.reshape(-1, 1)  # 'quality' as target

            # Normalize input features
            X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

            # Update input size and output size based on the loaded data
            self.input_size.set(X.shape[1])
            self.output_size.set(1)  # Output is always one (wine quality)

            # Generate network if not already initialized
            if not hasattr(self, 'network'):
                self.generate_network()

            self.X = X
            self.y = y

            print("Dataset loaded successfully")
            print(f"Input features: {X.shape[1]}, Output size: {y.shape[1]}")

            # Success message
            messagebox.showinfo("Success", "Dataset loaded successfully!")

    def start_training_with_dataset(self):
        if not hasattr(self, 'network'):
            print("Network has not been initialized yet!")
            return
        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            print("Dataset has not been loaded yet!")
            return

        def update_visualization():
            self.visualize_network()

        def train_step(epoch=0):
            if epoch < 1000:  # Train for 1000 epochs
                self.network.train(self.X, self.y, epochs=1, learning_rate=0.01, update_callback=update_visualization)
                self.root.after(10, train_step, epoch + 1)
            else:
                # Display success message after training is completed
                messagebox.showinfo("Success", "Training completed successfully!")

        train_step()

    def generate_network(self):
        input_size = self.input_size.get()
        hidden_layers = list(map(int, self.hidden_layers.get().split(",")))
        output_size = self.output_size.get()
        activation = self.activation.get()
        self.network = NeuralNetwork(input_size, hidden_layers, output_size, activation)
        self.visualize_network()

    def visualize_network(self):
        self.canvas.delete("all")
        layers = [self.network.input_size] + self.network.hidden_layers + [self.network.output_size]
        max_nodes = max(layers)
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        layer_gap = width / len(layers)
        node_gap = height / max(2, min(max_nodes, 20))

        layer_positions = []
        for i, nodes in enumerate(layers):
            x = layer_gap * i + layer_gap / 2
            layer_positions.append([(x, (height / 2) - (nodes / 2) * node_gap + j * node_gap) for j in range(nodes)])

        for i, layer in enumerate(layer_positions):
            for j, (x, y) in enumerate(layer):
                activation = 0.5  # Default value in case of errors
                if hasattr(self.network, 'a') and i < len(self.network.a):
                    if isinstance(self.network.a[i], np.ndarray):
                        # Safely extract the scalar value for the current node
                        if j < len(self.network.a[i]):
                            activation = self.network.a[i].flatten()[j]
                    else:
                        activation = self.network.a[i]

                # Ensure activation is a scalar and normalize between 0 and 1
                try:
                    activation = float(np.clip(activation, 0, 1))
                except (TypeError, ValueError):
                    activation = 0.5  # Fallback in case of errors

                # Generate color based on activation value
                node_color = f"#{int(activation * 255):02x}00{255 - int(activation * 255):02x}"
                node_id = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill=node_color)
                self.canvas.tag_bind(node_id, "<Enter>",
                                     lambda e, txt=f"Activation: {activation:.2f}": self.show_tooltip(e, txt))
                self.canvas.tag_bind(node_id, "<Leave>", self.hide_tooltip)

        for i in range(len(layer_positions) - 1):
            for start_idx, node_start in enumerate(layer_positions[i]):
                for end_idx, node_end in enumerate(layer_positions[i + 1]):
                    weight = self.network.weights[i][start_idx, end_idx]
                    normalized_weight = abs(weight) / max(1, np.max(np.abs(self.network.weights[i])))
                    line_width = max(1, int(normalized_weight * 5))
                    line_color = f"#{255 - int(normalized_weight * 255):02x}{int(normalized_weight * 255):02x}00"
                    line_id = self.canvas.create_line(node_start[0], node_start[1], node_end[0], node_end[1],
                                                      fill=line_color, width=line_width)
                    self.canvas.tag_bind(line_id, "<Enter>",
                                         lambda e, txt=f"Weight: {weight:.2f}": self.show_tooltip(e, txt))
                    self.canvas.tag_bind(line_id, "<Leave>", self.hide_tooltip)

    def save_visualization(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".ps", filetypes=[("PostScript files", "*.ps")])
        if file_path:
            self.canvas.postscript(file=file_path)

    def show_tooltip(self, event, text):
        self.tooltip = tk.Label(self.root, text=text, bg="black", font=("Arial", 8))
        self.tooltip.place(x=event.x_root, y=event.y_root)

    def hide_tooltip(self, event=None):
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

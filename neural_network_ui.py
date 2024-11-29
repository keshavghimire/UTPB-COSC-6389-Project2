import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from neural_network_logic import NeuralNetwork
import pandas as pd
import numpy as np
import threading


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
        self.activation = tk.StringVar(value="relu")
        ttk.Combobox(frame, textvariable=self.activation, values=["relu", "sigmoid", "tanh"]).grid(row=3, column=1)

        # Button to load dataset
        ttk.Button(frame, text="Load Dataset", command=self.load_dataset).grid(row=4, column=0, padx=5)

        # Start Training button (ensure it's an instance attribute)
        self.start_training_button = ttk.Button(
            frame, text="Start Training", command=self.start_training_with_dataset
        )
        self.start_training_button.grid(row=4, column=1, padx=5)

        # Button to save visualization
        ttk.Button(frame, text="Save Visualization", command=self.save_visualization).grid(row=4, column=2, padx=5)

        # Canvas for visualization
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def load_dataset(self):
        """
        Load a dataset from a CSV file, preprocess it, and update the UI.
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Load dataset
                data = pd.read_csv(file_path)
                X = data.iloc[:, :-1].values  # Features
                y = data.iloc[:, -1].values.reshape(-1, 1)  # Labels

                # Normalize inputs and outputs
                X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                y = (y - y.min(axis=0)) / (y.max(axis=0) - y.min(axis=0))

                # Update UI
                self.input_size.set(X.shape[1])  # Update input size
                self.output_size.set(y.shape[1])  # Update output size
                self.X, self.y = X, y  # Store dataset for training

                # Regenerate network structure
                self.generate_network()

                # Confirmation Message
                messagebox.showinfo(
                    "Dataset Loaded",
                    f"Dataset loaded successfully!\n\n"
                    f"Features: {X.shape[1]}\nSamples: {X.shape[0]}"
                )

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def start_training_with_dataset(self):
        """
        Start training the neural network in a separate thread and update visualization live.
        """
        if not hasattr(self, 'network'):
            messagebox.showerror("Error", "Please load the dataset first!")
            return

        # Disable the Start Training button to prevent multiple clicks
        self.start_training_button.config(state='disabled')

        print("Starting training...")

        # Start the training in a separate thread
        training_thread = threading.Thread(target=self.train_network)
        training_thread.start()

    def train_network(self):
        """
        Train the neural network and update visualization in a thread-safe manner.
        """
        epochs = 1000
        learning_rate = 0.01
        self.network.train(
            self.X,
            self.y,
            epochs=epochs,
            learning_rate=learning_rate,
            update_callback=self.update_visualization_thread_safe
        )
        print("Training completed!")

        # Schedule the messagebox and re-enable button on the main thread
        self.root.after(0, self.on_training_complete)

    def generate_network(self):
        input_size = self.input_size.get()
        hidden_layers = list(map(int, self.hidden_layers.get().split(",")))
        output_size = self.output_size.get()
        activation = self.activation.get()
        self.network = NeuralNetwork(input_size, hidden_layers, output_size, activation)
        self.visualize_network()

    def visualize_network(self):
        """
        Visualize the neural network structure with all connections thin initially.
        """
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
                self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")

        # Initially set all lines thin
        for i in range(len(layer_positions) - 1):
            for start_idx, node_start in enumerate(layer_positions[i]):
                for end_idx, node_end in enumerate(layer_positions[i + 1]):
                    self.canvas.create_line(
                        node_start[0], node_start[1], node_end[0], node_end[1], fill="gray", width=1
                    )

    def update_visualization_during_training(self):
        """
        Dynamically update the thickness and color of the lines based on weight values during training.
        """
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
                self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")

        for i in range(len(layer_positions) - 1):
            max_weight = np.max(np.abs(self.network.weights[i])) if np.max(np.abs(self.network.weights[i])) != 0 else 1
            for start_idx, node_start in enumerate(layer_positions[i]):
                for end_idx, node_end in enumerate(layer_positions[i + 1]):
                    weight = self.network.weights[i][start_idx, end_idx]
                    normalized_weight = abs(weight) / max_weight
                    line_width = max(1, int(normalized_weight * 5))  # Scale thickness
                    color_intensity = int(normalized_weight * 255)
                    if weight >= 0:
                        line_color = f"#{255 - color_intensity:02x}{color_intensity:02x}00"  # Green for positive
                    else:
                        line_color = f"#{color_intensity:02x}00{255 - color_intensity:02x}"  # Red for negative
                    self.canvas.create_line(
                        node_start[0], node_start[1], node_end[0], node_end[1],
                        fill=line_color, width=line_width
                    )

    def save_visualization(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".ps", filetypes=[("PostScript files", "*.ps")])
        if file_path:
            self.canvas.postscript(file=file_path)

    def update_visualization_thread_safe(self):
        """
        Schedule the visualization update on the main thread using root.after().
        This ensures thread-safe updates of the Tkinter UI.
        """
        self.root.after(0, self.update_visualization_during_training)

    def on_training_complete(self):
        """
        Handle UI updates after training completes.
        """
        # Notify the user that training is complete
        messagebox.showinfo("Training Complete", "Training has completed successfully!")

        # Re-enable the "Start Training" button
        self.start_training_button.config(state='normal')

        # Plot the training loss
        self.network.plot_loss()


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

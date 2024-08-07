from collections import deque
import json
from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib
from scipy.spatial import distance_matrix
from configs.basic2x5 import node_config, node_positions, input_indices, output_indices
import pickle
from datetime import datetime
import tkinter as tk
from tkinter.filedialog import askopenfilename
import sys
from pynput import keyboard

root = tk.Tk()
root.withdraw()

# config
auto_start_training = True
enable_spatial_attention = False
show_weights_biases = False
show_distances = False
show_activations = False # show activations of last iteration

# the weights and biases of the network will be generated randomly around this seed
seed_weights : bool = True
seed_checkpoint_path = "./checkpoints/checkpoint_2024-08-07_14-00-39.atmn"
seed_weights_bias : Any = None
if seed_weights:
    with open(seed_checkpoint_path, 'rb') as f:
        seed_weights_bias = pickle.load(f)
        
class SpatialNeuron:
    def __init__(self, index, position, activation="sigmoid", num_connections=3):
        self.index = index
        self.position = np.array(position)
        self.is_input = False
        self.is_output = False
        self.is_hidden = False
        self.num_connections = num_connections
        self.weights = np.ndarray([])
        self.bias = np.zeros(1)
        self.outgoing_connections : List[SpatialNeuron] = []
        self.incoming_connections : List[SpatialNeuron] = []
        self.activation_function = activation

    def initialize_weights_and_biases(self):
        # Number of incoming and outgoing connections
        # fan_in = len(self.incoming_connections)
        # fan_out = len(self.outgoing_connections)

        # # Xavier initialization
        # if fan_in > 0 and fan_out > 0:
        #     scale = np.sqrt(2 / (fan_in + fan_out))
        #     self.weights = np.random.randn(fan_in) * scale
        # else:
        # self.weights = np.random.randn(len(self.incoming_connections))
        self.weights : np.ndarray = np.ndarray(shape=(len(self.incoming_connections)))
        if seed_weights and self.index < len(seed_weights_bias):
            self.weights = seed_weights_bias[self.index][0][:len(self.incoming_connections)]
            self.bias = seed_weights_bias[self.index][1]
            # add some noise
            self.weights += np.random.randn(len(self.incoming_connections)) * 0.01
            self.bias += np.random.randn(1) * 0.01
        else:
            for i in range(len(self.incoming_connections)):
                self.weights[i] = np.random.randn(1)
            self.bias = np.random.randn(1)
        print("Neuron at", self.position, "has weights", self.weights, "and bias", self.bias)

    def forward(self, inputs, activation=True):
        # print("Forwarding neuron at", self.position, "with inputs", inputs, "and weights", self.weights)
        z = np.dot(inputs, self.weights) + self.bias
        # print("Z", z)
        if activation:
            a = self.activation(z)
        else:
            a = z
        # output nodes don't have spatial attention
        if self.is_output:
            return a
        if enable_spatial_attention:
            # distance attention
            output_distances = [np.linalg.norm(self.position - conn.position) for conn in self.outgoing_connections]
            output_distances = np.array(output_distances).reshape(-1, 1)
            a = a * (1 / output_distances)
        else:
            a = a * [[1] for conn in self.outgoing_connections]
        return a

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_function == "tanh":
            return self.tanh(x)
        elif self.activation_function == "relu":
            return self.relu(x)
        elif self.activation_function == "softmax":
            return self.softmax(x)
        elif self.activation_function == "linear":
            return x
        else:
            raise ValueError(f"Invalid activation function {self.activation_function}")

    def activation_derivative(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid_derivative(x)
        elif self.activation_function == "tanh":
            return self.tanh_derivative(x)
        elif self.activation_function == "relu":
            return self.relu_derivative(x)
        elif self.activation_function == "softmax":
            return self.softmax_derivative(x)
        elif self.activation_function == "linear":
            return 1
        else:
            raise ValueError(f"Invalid activation function {self.activation_function}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def softmax_derivative(self, x):
        return self.softmax(x) * (1 - self.softmax(x))

class SpatialNetwork:
    def __init__(self, node_config, input_indices, output_indices):
        self.neurons = [SpatialNeuron(ind, pos, activation=node_config[pos]) for ind, pos in enumerate(node_config.keys())]
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.dmin = 1.1  # minimum distance for spatial attention
        self.dmax = 2  # maximum distance for spatial attention
        self.hidden_indices = [i for i in range(len(self.neurons)) if i not in input_indices and i not in output_indices]

        for i in input_indices:
            self.neurons[i].is_input = True
            # self.neurons[i].num_connections = 20
        for i in output_indices:
            self.neurons[i].is_output = True
        for i in self.hidden_indices:
            self.neurons[i].is_hidden = True

        self.initialize_connections()
        # activations is a matrix representing activation value sent from ith neuron to jth neuron
        self.activations = np.zeros((len(self.neurons), len(self.neurons)))
        self.learning_rate = 0.0005
        self.position_learning_rate = 0.5
        self.current_iteration = 0
        self.loss_history = []
        self.distances = np.zeros((len(self.neurons), len(self.neurons)))

    def initialize_connections(self):
        self.distances = distance_matrix([n.position for n in self.neurons], [n.position for n in self.neurons])

        # Connect input nodes to nearest hidden nodes
        # only 1 hidden neuron connected per input node
        for i in self.input_indices:
            nearest_hidden = sorted(self.hidden_indices, key=lambda j: self.distances[i][j])[:self.neurons[i].num_connections]
            self.neurons[i].outgoing_connections = [self.neurons[j] for j in nearest_hidden]
            for j in nearest_hidden:
                self.neurons[j].incoming_connections.append(self.neurons[i])

        # Connect hidden nodes to nearest hidden or output nodes
        # ensure no cycles are created
        for i in self.hidden_indices:
            valid_targets = self.hidden_indices + self.output_indices
            valid_targets.remove(i)  # Exclude self-connection
            nearest = sorted(valid_targets, key=lambda j: self.distances[i][j])
            for j in nearest:
                # connect only if the node is at the same or more X position
                # also if the node is not already connected
                # print(f"Connecting {self.neurons[i].position} to {self.neurons[j].position}")
                from_neuron : SpatialNeuron = self.neurons[i]
                to_neuron : SpatialNeuron = self.neurons[j]
                if to_neuron.position[0] >= from_neuron.position[0] and to_neuron not in from_neuron.incoming_connections:
                    to_neuron.incoming_connections.append(from_neuron)
                    from_neuron.outgoing_connections.append(to_neuron)
                if len(from_neuron.outgoing_connections) == from_neuron.num_connections:
                    break

        # Initialize weights for hidden and output nodes
        for i in self.hidden_indices + self.output_indices:
            neuron : SpatialNeuron = self.neurons[i]
            neuron.initialize_weights_and_biases()

    def update_distances(self):
        self.distances = distance_matrix([n.position for n in self.neurons], [n.position for n in self.neurons])
        # self distances are inf
        np.fill_diagonal(self.distances, np.inf)

    def forward(self, inputs):
        # Reset activations, default value is -1
        self.activations.fill(-1)

        # Set the activations for input neurons
        for index, inp_node_index in enumerate(self.input_indices):
            inp_value = inputs[index][0] if isinstance(inputs[index], np.ndarray) else inputs[index]
            for conn in self.neurons[inp_node_index].outgoing_connections:
                self.activations[inp_node_index][conn.index] = inp_value

        # Topological sort of neurons
        sorted_neurons = self.topological_sort()

        # Activate neurons in topological order
        for neuron_index in sorted_neurons:
            neuron = self.neurons[neuron_index]
            if not neuron.is_input:
                inputs = np.array([self.activations[conn.index][neuron_index] for conn in neuron.incoming_connections])
                output_activations = neuron.forward(inputs)

                if neuron.is_output:
                    self.activations[neuron_index][neuron_index] = output_activations[0]
                else:
                    for idx, conn in enumerate(neuron.outgoing_connections):
                        self.activations[neuron_index][conn.index] = output_activations[idx][0]

        # Return the activations of the output nodes
        return np.array([self.activations[i][i] for i in self.output_indices])

    def topological_sort(self):
        # Perform topological sort of the neurons
        in_degree = {i: 0 for i in range(len(self.neurons))}
        for neuron in self.neurons:
            for conn in neuron.outgoing_connections:
                in_degree[conn.index] += 1

        queue = deque(self.input_indices)
        sorted_neurons = []

        while queue:
            node = queue.popleft()
            sorted_neurons.append(node)
            for conn in self.neurons[node].outgoing_connections:
                in_degree[conn.index] -= 1
                if in_degree[conn.index] == 0:
                    queue.append(conn.index)

        return sorted_neurons

    def backward(self, x, y_true, y_pred):
        gradients = {i: {'weights': np.zeros_like(n.weights) if n.weights is not None else 0,
                         'bias': 0,
                         'position': np.zeros_like(n.position)}
                     for i, n in enumerate(self.neurons)}
        deltas = {i: np.ndarray([]) for i in range(len(self.neurons))}

        # Compute output gradients
        output_deltas = {}
        for i, (y_true, y_pred) in enumerate(zip(y_true, y_pred)):
            output_index = self.output_indices[i]
            d_loss = y_pred - y_true
            # print("Loss", d_loss, "y_true", y_true, "y_pred", y_pred)
            d_activation_derivative = self.neurons[output_index].activation_derivative(y_pred)
            output_deltas[output_index] = d_loss * d_activation_derivative

        def compute_gradient(curr_node):
            neuron : SpatialNeuron = self.neurons[curr_node]
            # print("Computing gradient for neuron", neuron.position, " i", curr_node)
            if curr_node in output_deltas:
                # print("got value from output deltas")
                delta = output_deltas[curr_node]
                deltas[curr_node] = delta
            else:
                delta_l_plus_1 = np.zeros(len(neuron.outgoing_connections))
                weights_l_plus_1 = np.zeros(len(neuron.outgoing_connections))

                for outgoing_connection_number, conn in enumerate(neuron.outgoing_connections):
                    weight_index = conn.incoming_connections.index(neuron)
                    delta_l_plus_1[outgoing_connection_number] = deltas[conn.index] # delta^(l+1)
                    weights_l_plus_1[outgoing_connection_number] = conn.weights[weight_index] # w^(l+1)

                delta_l = delta_l_plus_1 @ weights_l_plus_1 # w^(l+1) * delta^(l+1)
                # print("Delta l", delta_l, "delta_l_plus_1", delta_l_plus_1, "weights_l_plus_1", weights_l_plus_1)
                deltas[curr_node] = delta_l

                z_l = np.average([self.activations[conn.index][curr_node] for conn in neuron.incoming_connections])

                activation_derivative = neuron.activation_derivative(z_l)
                delta = delta_l * activation_derivative # delta^(l) = delta^(l+1) @ w^(l+1) * f'(z^(l))

            # print("Delta", delta)
            if not neuron.is_input:
                # Bias gradient = delta
                gradients[curr_node]['bias'] = delta
                # Weights gradient = a^(l-1) * delta
                gradients[curr_node]['weights'] = np.array([self.activations[n.index][curr_node] for n in neuron.incoming_connections]) * delta

                if enable_spatial_attention:
                    # Position gradient calculation remains the same
                    # All nodes pull their incoming connections, if possible and benefitial
                    for conn in neuron.incoming_connections:
                        distance = self.distances[curr_node][conn.index]
                        if distance < self.dmin:
                            d_vector = -1 * 0.01 * (neuron.position - conn.position) / distance  # Repulsive force
                            # print(f"neuron {conn.position} repulsive")
                        elif distance > self.dmax:
                            d_vector = 0.01 * (neuron.position - conn.position) / distance  # Attractive force
                            # print(f"neuron {conn.position} attractive")
                        else:
                            d_vector = -1*(delta * self.activations[conn.index][neuron.index]) * (neuron.position - conn.position) / (1 + distance)**3
                            # print(f"neuron {neuron.position} conn {conn.position} d_vector {d_vector}")
                        gradients[conn.index]['position'] = gradients[conn.index]['position'] + d_vector

            return delta

        for i in reversed(range(len(self.neurons))):
            if not self.neurons[i].is_input:
                compute_gradient(i)

        return gradients

    def train_step(self, train_dataset, test_dataset):
        total_loss = 0
        total_correct = 0
        gradients = {i: {'weights': np.zeros_like(n.weights) if n.weights is not None else 0,
                         'bias': 0,
                         'position': np.zeros_like(n.position)}
                     for i, n in enumerate(self.neurons)}

        for x, y_true in train_dataset:
            y_pred = self.forward(x)
            if enable_spatial_attention:
                self.update_distances()
            # print("Predicted", y_pred)
            # print(np.max(self.activations))
            single_row_gradients = self.backward(x, y_true, y_pred)
            # for i, grad in single_row_gradients.items():
            #     if i == 2:
                # print(f"p={grad['position']}")
                        #   , b={grad['bias']}, p={grad['position']}")
            for i, grad in single_row_gradients.items():
                for key in grad:
                    gradients[i][key] = gradients[i][key] + grad[key]


        class NumpyEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)
        # print(json.dumps(gradients, cls=NumpyEncoder, indent=2))

        for x, y_true in test_dataset.copy():
            y_pred = self.forward(x)
            # change shape of y_pred to match y_true
            y_pred = np.array(y_pred).reshape(-1)

            # Compute loss and accuracy
            prediction_loss = np.mean((y_pred - y_true) ** 2)
            loss = prediction_loss
            total_loss += loss
            # if self.current_iteration % 10 == 0:
            # print("Predictions : ", np.round(y_pred, 2), "True : ", np.round(y_true, 2))
            total_correct += np.all(np.round(y_pred, 1) == np.round(y_true, 1))

        # Average gradients
        for i in gradients:
            for key in gradients[i]:
                # if key != 'position':
                gradients[i][key] = gradients[i][key] / len(train_dataset)

        for i, neuron in enumerate(self.neurons):
            if not neuron.is_input:
                neuron.weights -= self.learning_rate * gradients[i]['weights']
                neuron.bias -= self.learning_rate * gradients[i]['bias']
                neuron.position = neuron.position + (gradients[i]['position'] * self.position_learning_rate)

        accuracy = total_correct / len(test_dataset)
        loss_percentage = total_loss / len(test_dataset)
        self.loss_history.append(loss_percentage)
        self.current_iteration += 1

        # print(f"Iteration {self.current_iteration}: Loss {loss_percentage:.4f}, Accuracy {accuracy:.2f}")
        return loss_percentage, accuracy
    
    def save_checkpoint(self, path):
        print(f"Saving checkpoint to {path}")
        with open(path, 'wb') as f:
            pickle.dump([(neuron.weights, neuron.bias) for neuron in self.neurons], f)
            print("Checkpoint saved")
        return
    
    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        with open(path, 'rb') as f:
            loaded_network = pickle.load(f)
        for i, (weights, bias) in enumerate(loaded_network):
            self.neurons[i].weights = weights
            self.neurons[i].bias = bias
        print("Checkpoint loaded")
        return


# Visualization setup
# Correcting the one-liner to create the desired subplot layout
fig, axs = plt.subplots(2, 2, figsize=(3, 2), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1], 'wspace': 0.1}, squeeze=False)
# Merging the first row to create a single plot spanning two columns
fig.delaxes(axs[0, 0])
fig.delaxes(axs[0, 1])
ax1 = fig.add_subplot(2, 2, (1, 2))

# def move_figure(f, x, y):
#     """Move figure's upper left corner to pixel (x, y)"""
#     backend = matplotlib.get_backend()
#     if backend == 'TkAgg':
#         f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
#     elif backend == 'WXAgg':
#         f.canvas.manager.window.SetPosition((x, y))
#     elif backend == 'macosx':
#         mgr = plt.get_current_fig_manager()
#         if mgr:
#             mgr.window.wm_geometry("+" + str(x) + "+" + str(y))
#     else:
#         # This works for QT and GTK
#         # You can also use window.setGeometry
#         f.canvas.manager.window.move(x, y)

# Movie the figure to the x and y arguments passed to run the code
arg_x = sys.argv[1] if len(sys.argv) > 1 else 0
arg_y = sys.argv[2] if len(sys.argv) > 2 else 0
# move_figure(fig, int(arg_x), int(arg_y))

plt.subplots_adjust(bottom=0.3)
# Define node positions for a spatial network to predict sine waves
# Input layer (1 node), two hidden layers (3 nodes each), output layer (1 node)

# Create the SpatialNetwork instance
network = SpatialNetwork(node_config, input_indices, output_indices)

# Generate datasets
def generate_sine_data(num_samples, start, end):
    x = np.linspace(start, end, num_samples)
    y = np.sin(x)
    # normalize y to be between 0 and 1
    y = (y + 1) / 2
    return np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))

# Training dataset: Cover the full range of sine wave
train_dataset = generate_sine_data(60, 0, 2*np.pi)
train_dataset = np.expand_dims(train_dataset, axis=2)

# Test dataset: Include values outside the training range to test generalization
test_dataset = generate_sine_data(40, 0, 2*np.pi)
test_dataset = np.expand_dims(test_dataset, axis=2)

# # Normalize the datasets
# max_val = max(np.max(np.abs(train_dataset)), np.max(np.abs(test_dataset)))
# train_dataset /= max_val
# test_dataset /= max_val

# Example usage
print("Network configuration:")
print(f"Node positions: {node_positions}")
print(f"Input indices: {input_indices}")
print(f"Output indices: {output_indices}")

print("\nTraining dataset shape:", train_dataset.shape)
print("Test dataset shape:", test_dataset.shape)

# Display a few samples from each dataset
print("\nSample from training dataset:")
print(train_dataset[:5])
print("\nSample from test dataset:")
print(test_dataset[:5])

# Main plot
input_nodes = [node_positions[i] for i in input_indices]
input_plot = ax1.scatter([p[0] for p in input_nodes], [p[1] for p in input_nodes], c='blue', s=50, zorder=2)
output_nodes = [node_positions[i] for i in output_indices]
output_plot = ax1.scatter([p[0] for p in output_nodes], [p[1] for p in output_nodes], c='green', s=50, zorder=2)
neuron_nodes = [node_positions[i] for i in network.hidden_indices]
neuron_plot = ax1.scatter([p[0] for p in neuron_nodes], [p[1] for p in neuron_nodes], c='orange', s=50, zorder=3)
lines = []

def add_arrow(line, position=None, direction='right', size=10, color='gray'):
    if position is None:
        position = line.get_data()
    xdata, ydata = position
    arrowprops = dict(facecolor=color, edgecolor=color, arrowstyle='-|>', mutation_scale=size)
    return ax1.annotate('', xy=(xdata[1], ydata[1]), xytext=(xdata[0], ydata[0]), arrowprops=arrowprops)

for neuron in network.neurons:
    for conn in neuron.outgoing_connections:
        line, = ax1.plot([neuron.position[0], conn.position[0]],
                         [neuron.position[1], conn.position[1]],
                         'gray', linewidth=1, zorder=1)
        add_arrow(line)
        lines.append(line)

ax1.set_xlim(min([p[0] for p in node_positions]) - 0.5, max([p[0] for p in node_positions]) + 0.5)
ax1.set_ylim(min([p[1] for p in node_positions]) - 0.5, max([p[1] for p in node_positions]) + 0.5)
ax1.axis('off')
for i, neuron in enumerate(network.neurons):
    # calculate incoming connection distances
    incoming_distances = []
    for conn in neuron.incoming_connections:
        incoming_distances.append(round(float(network.distances[i][conn.index]), 2))

    neuron_label_text = f'N{i}'
    if show_weights_biases:
        neuron_label_text += f'\nw={neuron.weights}\nb={neuron.bias}'
    if show_distances:
        neuron_label_text += f'\nd={incoming_distances}'
    if show_activations:
        acts = [round(float(network.activations[conn.index][i]), 2) for conn in neuron.incoming_connections]
        neuron_label_text += f'\na={acts}'
    ax1.text(neuron.position[0]-.1, neuron.position[1]+.1, neuron_label_text, ha='center', va='center')

# Loss plot
ax2 = axs[1, 0]
loss_line, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 0.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.grid(True)
# also show the current loss value
loss_accuracy_text = ax2.text(0.02, 0.95, '',
                        verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes)

# Function plot
ax3 = axs[1, 1]
ax3.plot(train_dataset[:, 0][:, 0], train_dataset[:, 1][:, 0], label='Training data')
ax3.set_xlim(-0.5, 2*np.pi + 0.5)
ax3.set_ylim(-0.5, 1.5)

def update_plot():
    for annotation in ax1.texts:
        annotation.remove()

    for i, neuron in enumerate(network.neurons):
        # calculate incoming connection distances
        incoming_distances = []
        for conn in neuron.incoming_connections:
            incoming_distances.append(round(float(network.distances[i][conn.index]), 2))

        neuron_label_text = f'N{i}'
        if show_weights_biases:
            neuron_label_text += f'\nw={neuron.weights}\nb={neuron.bias}'
        if show_distances:
            neuron_label_text += f'\nd={incoming_distances}'
        if show_activations:
            acts = [round(float(network.activations[conn.index][i]), 2) for conn in neuron.incoming_connections]
            neuron_label_text += f'\na={acts}'
        ax1.texts[i].set_position(tuple(neuron.position))
        ax1.texts[i].set_text(neuron_label_text)

    neuron_plot.set_offsets([n.position for n in network.neurons if n.is_hidden])

    line_index = 0
    for neuron in network.neurons:
        for conn in neuron.outgoing_connections:
            lines[line_index].set_data([neuron.position[0], conn.position[0]], [neuron.position[1], conn.position[1]])
            add_arrow(lines[line_index])
            line_index += 1

    plt.draw()

curr_max_loss = 0.3
def update_loss_plot(loss, accuracy):
    global curr_max_loss
    curr_max_loss = max(curr_max_loss, loss)
    loss_accuracy_text.set_text(f'Iteration: {network.current_iteration}, \nLoss: {loss:.4f}, \nAccuracy: {(accuracy*100):.2f}%')
    loss_line.set_data(range(len(network.loss_history)), network.loss_history)
    ax2.set_xlim(0, max(100, len(network.loss_history)))
    ax2.set_ylim(0, curr_max_loss)
    plt.draw()


def update_function_plot():
    # plot the predicted sine wave
    x, y = test_dataset[:, 0][:, 0].copy(), test_dataset[:, 1][:, 0].copy()
    for i, x_val in enumerate(x):
        y[i] = network.forward(np.array([[x_val]]))[0]
        # print(f"Predicted {y[i]:.2f} for {x_val:.2f}, True {np.sin(x_val):.2f}")
    if len(ax3.lines) > 1:
        ax3.lines[1].set_data(x, y)
    else:
        ax3.plot(x, y, 'r')
    plt.draw()

# for neuron in network.neurons:
#     print("Neuron at", neuron.position, "has incoming connections", [conn.position for conn in neuron.incoming_connections])
#     print("and outgoing connections", [conn.position for conn in neuron.outgoing_connections])

# step_size = len(dataset)
def on_click(event):
    # step_size = 1
    # train_step = train_dataset[network.current_iteration % len(train_dataset) * step_size: (network.current_iteration % len(train_dataset) + 1) * step_size]
    # test_step = test_dataset[network.current_iteration % len(test_dataset) * step_size: (network.current_iteration % len(test_dataset) + 1) * step_size]
    loss, accuracy = network.train_step(train_dataset, test_dataset)
    # plot asynchrously

    plot_update_step = 1
    if network.current_iteration % plot_update_step == 0:
        update_loss_plot(loss, accuracy)
        update_function_plot()
        if enable_spatial_attention or show_distances or show_distances or show_activations:
            update_plot()

button_ax = plt.axes((0.8, 0.05, 0.1, 0.075))
button = Button(button_ax, 'Next Iteration')
button.on_clicked(on_click)

# play pause training with 100ms interval
play_button = Button(plt.axes((0.6, 0.05, 0.1, 0.075)), 'Play')
pause_button = Button(plt.axes((0.4, 0.05, 0.1, 0.075)), 'Pause')
pause = True

# Buttons to save checkpoints
save_button = Button(plt.axes((0.2, 0.05, 0.1, 0.075)), 'Save Checkpoint')
load_button = Button(plt.axes((0, 0.05, 0.1, 0.075)), 'Load Checkpoint')

def on_play(event):
    print("Playing")
    global pause
    pause = False
    while not pause:
        on_click(None)
        plt.pause(0.01)

def on_pause(event):
    global pause
    pause = True
    
def on_save(event):
    ckpt_name = f"checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.atmn"
    network.save_checkpoint("./checkpoints/" + ckpt_name)
    
def on_load(event):
    # select a checkpoint file
    file_path = askopenfilename(defaultextension="checkpoints/*.atmn", filetypes=[("Atman Checkpoint", "*.atmn")])
    network.load_checkpoint(file_path)

def on_release(key):
    if key == keyboard.Key.esc:
        if pause:
            on_play(None)
        else:
            on_pause(None)
    elif key == 'q':
        plt.close()
    return
        
listener = keyboard.Listener(on_release=on_release)
listener.start()

play_button.on_clicked(on_play)
pause_button.on_clicked(on_pause)
save_button.on_clicked(on_save)
load_button.on_clicked(on_load)

if auto_start_training:
    on_play(None)
    
plt.show()
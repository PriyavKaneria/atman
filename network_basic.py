from collections import deque
import json
from typing import Any, List
import numpy as np
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
import pickle
from numpy.random import randint
from scipy.spatial import distance_matrix
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QGraphicsPolygonItem, QHBoxLayout, QMainWindow, QSizePolicy, QVBoxLayout, QWidget, QLabel, QPushButton, QGridLayout, QGraphicsScene, QGraphicsView, QGraphicsItemGroup, QGraphicsLineItem, QGraphicsTextItem, QGraphicsEllipseItem
from PyQt6.QtCore import Qt, QTimer, QLineF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPolygonF, QVector2D
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

from configs.basic3x5 import node_config, node_positions, input_indices, output_indices

# for file dialog
root = tk.Tk()
root.withdraw()

# config
auto_start_training = True
enable_spatial_attention = False
show_weights_biases = False
show_distances = False
show_activations = False # show activations of last iteration

# the weights and biases of the network will be generated randomly around this seed
seed_weights : bool = False
seed_checkpoint_path = "./checkpoints/checkpoint_2024-08-09_23-59-06.atmn"
seed_weights_bias_global : Any = None
if seed_weights:
    with open(seed_checkpoint_path, 'rb') as f:
        seed_weights_bias_global = pickle.load(f)

class SpatialNeuron:
    def __init__(self, index, position, activation="sigmoid", num_connections=3, seed_weights_bias_neuron=None, seed_child_mutation=0.01):
        self.index = index
        self.position = np.array(position) * [6, 8]
        self.is_input = False
        self.is_output = False
        self.is_hidden = False
        self.num_connections = num_connections
        self.weights = np.ndarray([])
        self.bias = np.zeros(1)
        self.outgoing_connections : List[SpatialNeuron] = []
        self.incoming_connections : List[SpatialNeuron] = []
        self.activation_function = activation
        self.seed_weights_bias_neuron : Any = seed_weights_bias_neuron
        self.seed_child_mutation = seed_child_mutation

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
        if self.seed_weights_bias_neuron:
            self.weights = self.seed_weights_bias_neuron[0][:len(self.incoming_connections)].copy()
            self.bias = self.seed_weights_bias_neuron[1].copy()
            # add some noise
            np.random.seed(datetime.now().microsecond + self.index)
            self.weights += np.random.randn(len(self.incoming_connections)) * self.seed_child_mutation
            self.bias += np.random.rand() * self.seed_child_mutation
        else:
            for i in range(len(self.incoming_connections)):
                self.weights[i] = np.random.randn(1)
            self.bias = np.random.randn(1)
        # print("Neuron at", self.position, "has weights", self.weights, "and bias", self.bias)
        # print("Diff from seed", self.weights - self.seed_weights_bias_neuron[0][:len(self.incoming_connections)], self.bias - self.seed_weights_bias_neuron[1])

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
    def __init__(self, node_config, input_indices, output_indices, seed_weights_bias : Any = None, seed_child_mutation : float = 0.01):
        self.neurons = [SpatialNeuron(ind, pos, activation=node_config[pos], seed_weights_bias_neuron=seed_weights_bias[ind] if seed_weights_bias else None, seed_child_mutation=seed_child_mutation) for ind, pos in enumerate(node_config.keys())]
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
        self.max_accuracy = 0

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


        # class NumpyEncoder(json.JSONEncoder):
        #     def default(self, o):
        #         if isinstance(o, np.ndarray):
        #             return o.tolist()
        #         return super().default(o)
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
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        self.current_iteration += 1
        self.max_accuracy = max(self.max_accuracy, accuracy)

        # print(f"Iteration {self.current_iteration}: Loss {loss_percentage:.4f}, Accuracy {accuracy:.2f}")
        return loss_percentage, accuracy

    def save_checkpoint(self, path):
        print(f"Saving checkpoint to {path}")
        with open(path, 'wb') as f:
            pickle.dump([(neuron.weights, neuron.bias) for neuron in self.neurons], f)
            print("Checkpoint saved")
        return
        
    def get_checkpoint(self):
        return [(neuron.weights, neuron.bias) for neuron in self.neurons]

    def load_checkpoint(self, path, add_noise=False):
        print(f"Loading checkpoint from {path}")
        with open(path, 'rb') as f:
            loaded_network = pickle.load(f)
        for i, (weights, bias) in enumerate(loaded_network):
            self.neurons[i].weights = weights
            self.neurons[i].bias = bias
            if add_noise:
                self.neurons[i].weights += np.random.normal(0, 0.1, self.neurons[i].weights.shape)
                self.neurons[i].bias += np.random.normal(0, 0.1)
        print("Checkpoint loaded")
        return

class MainWindow(QMainWindow):
    def __init__(self, node_config, input_indices, output_indices, x_pos=0, y_pos=0, seed_weights_bias=None, seed_child_mutation=0.01, max_iterations = -1, no_children = 1, auto_start = False):
        super().__init__()
        self.setWindowTitle("Spatial Network Visualization")
        self.move(x_pos, y_pos)

        self.no_children = no_children
        self.max_iterations = max_iterations
        self.child_networks = [SpatialNetwork(node_config, input_indices, output_indices, seed_weights_bias=seed_weights_bias, seed_child_mutation=seed_child_mutation) for _ in range(no_children)]
        self.child_network_colors = [QColor(*[np.random.randint(0, 255) for c in range(3)]) for _ in range(no_children)]

        # Training and test datasets
        self.train_dataset = generate_sine_data(60, 0, 2 * np.pi)
        self.train_dataset = np.expand_dims(self.train_dataset, axis=2)
        self.test_dataset = generate_sine_data(40, 0, 2 * np.pi)
        self.test_dataset = np.expand_dims(self.test_dataset, axis=2)

        self.main_layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.network_plots = [NetworkPlot(network) for network in self.child_networks]
        self.loss_function_layout = QHBoxLayout()
        
        self.loss_plot = LossPlot(self.child_networks, self.child_network_colors)
        self.function_plot = FunctionPlot(self.test_dataset, self.child_networks, self.child_network_colors)

        self.main_layout.addWidget(self.network_plots[0])
        self.loss_function_layout.addWidget(self.loss_plot)
        self.loss_function_layout.addWidget(self.function_plot)
        self.main_layout.addLayout(self.loss_function_layout)

        self.button_layout = QGridLayout()
        self.main_layout.addLayout(self.button_layout)

        self.next_button = QPushButton("Next Iteration")
        self.next_button.clicked.connect(self.train_step)
        self.button_layout.addWidget(self.next_button, 0, 0)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_training)
        self.button_layout.addWidget(self.play_button, 0, 1)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.stop_training)
        self.button_layout.addWidget(self.pause_button, 0, 2)

        self.save_button = QPushButton("Save Checkpoint")
        self.save_button.clicked.connect(self.save_checkpoint)
        self.button_layout.addWidget(self.save_button, 0, 3)

        self.load_button = QPushButton("Load Checkpoint")
        self.load_button.clicked.connect(self.load_checkpoint)
        self.button_layout.addWidget(self.load_button, 0, 4)

        self.timer = QTimer()
        self.timer.timeout.connect(self.train_step)
        self.is_training = False
        if auto_start:
            self.start_training()

    def train_step(self):
        for i, network in enumerate(self.child_networks):
            loss, accuracy = self.child_networks[i].train_step(self.train_dataset, self.test_dataset)
            # if self.child_networks[i].current_iteration % 5 == 0:
            self.function_plot.update_plot(i)
            # if self.child_networks[i].current_iteration % 10 == 0:
            self.loss_plot.update_plot(i, loss, accuracy)
                # self.network_plots[i].update_plot()
        if self.child_networks[-1].current_iteration == self.max_iterations:
            self.stop_training()
            self.close()

    def start_training(self):
        self.is_training = True
        self.timer.start(1)

    def stop_training(self):
        self.is_training = False
        self.timer.stop()

    def save_checkpoint(self):
        # save the checkpoint of highest accuracy
        child_index = np.argmax([float(network.max_accuracy) for network in self.child_networks])
        ckpt_name = f"checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.atmn"
        self.child_networks[child_index].save_checkpoint("./checkpoints/" + ckpt_name)

    def load_checkpoint(self):
        file_path = askopenfilename(defaultextension="checkpoints/*.atmn", filetypes=[("Atman Checkpoint", "*.atmn")])
        for network in self.child_networks:
            network.load_checkpoint(file_path, add_noise=True)

def generate_sine_data(num_samples, start, end):
    x = np.linspace(start, end, num_samples)
    y = np.sin(x)
    y = (y + 1) / 2
    return np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))

class NetworkPlot(QWidget):
    def __init__(self, network):
        super().__init__()
        self.network  : SpatialNetwork = network
        self.initUI()

    def initUI(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graphics_view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self._layout.addWidget(self.graphics_view, 1, Qt.AlignmentFlag.AlignCenter)

        self.update_plot()
                
    def update_plot(self):
        self.graphics_scene.clear()

        # Draw nodes
        for neuron in self.network.neurons:
            neuron_item = QGraphicsEllipseItem(neuron.position[0] - 1, neuron.position[1] - 1, 2, 2)
            if neuron.is_input:
                neuron_item.setPen(QPen(Qt.GlobalColor.blue, 0.1))
            elif neuron.is_output:
                neuron_item.setPen(QPen(Qt.GlobalColor.green, 0.1))
            else:
                neuron_item.setPen(QPen(Qt.GlobalColor.yellow, 0.1))
            self.graphics_scene.addItem(neuron_item)

            # calculate incoming connection distances
            incoming_distances = []
            for conn in neuron.incoming_connections:
                incoming_distances.append(round(float(self.network.distances[neuron.index][conn.index]), 2))
    
            neuron_label_text = f'N{neuron.index}'
            if show_weights_biases:
                neuron_label_text += f'\nw={neuron.weights}\nb={neuron.bias}'
            if show_distances:
                neuron_label_text += f'\nd={incoming_distances}'
            if show_activations:
                acts = [round(float(self.network.activations[conn.index][neuron.index]), 2) for conn in neuron.incoming_connections]
                neuron_label_text += f'\na={acts}'
                    
            # Draw neuron labels
            label_item = QGraphicsTextItem(neuron_label_text)
            label_item.setFont(QFont("Arial", 1))
            label_item.setPos(neuron.position[0], neuron.position[1])
            bounding_rect = label_item.boundingRect()
            
            # Center the label around the neuron's position
            label_item.setPos(neuron.position[0] - bounding_rect.width() / 2, 
                              neuron.position[1] - bounding_rect.height() / 2 - 1)
            
            self.graphics_scene.addItem(label_item)

        # Draw connections
        for neuron in self.network.neurons:
            for conn in neuron.outgoing_connections:
                line_item = QGraphicsLineItem(QLineF(neuron.position[0], neuron.position[1], conn.position[0], conn.position[1]))
                line_item.setPen(QPen(Qt.GlobalColor.gray, 0.1))
                self.graphics_scene.addItem(line_item)

                # Add arrow head
                arrow_length = 2
                arrow_angle = np.pi / 2.6
                arrow_end = QPointF(conn.position[0], conn.position[1])
                arrow_start = QPointF(neuron.position[0], neuron.position[1])
                arrow_vector = arrow_end - arrow_start
                np_arrow = np.array([arrow_vector.x(), arrow_vector.y()])
                np_arrow_normalized = np_arrow / np.linalg.norm(np_arrow)
                arrow_vector = QPointF(np_arrow_normalized[0], np_arrow_normalized[1])
                arrow_point1 = arrow_end - arrow_vector * arrow_length * np.cos(arrow_angle) + QPointF(-arrow_vector.y(), arrow_vector.x()) * arrow_length * np.sin(arrow_angle) * 0.1
                arrow_point2 = arrow_end - arrow_vector * arrow_length * np.cos(arrow_angle) + QPointF(arrow_vector.y(), -arrow_vector.x()) * arrow_length * np.sin(arrow_angle) * 0.1
                arrow_item = QGraphicsPolygonItem(QPolygonF([arrow_end, arrow_point1, arrow_point2]))
                arrow_item.setPen(QPen(Qt.GlobalColor.gray, 0.03))
                arrow_item.setBrush(QBrush(Qt.GlobalColor.gray))
                self.graphics_scene.addItem(arrow_item)

        self.graphics_scene.setSceneRect(self.graphics_scene.itemsBoundingRect())
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

class LossPlot(QWidget):
    def __init__(self, child_networks, child_network_colors):
        super().__init__()
        self.child_networks : List[SpatialNetwork] = child_networks
        self.child_network_colors : List[QColor] = child_network_colors
        self.initUI()
        self.loss_of_max_accuracy = 0
        self.max_accuracy = 0

    def initUI(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.child_series = [QLineSeries() for _ in self.child_networks]
        self.chart = QChart()
        for i, series in enumerate(self.child_series):
            # set a different color for each series\
            series.setPen(self.child_network_colors[i])
            self.chart.addSeries(series)
        self.chart.createDefaultAxes()
        self.chart.axes(Qt.Orientation.Horizontal)[0].setRange(0, 100)
        legend = self.chart.legend()
        if legend:
            legend.hide()

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # disable auto range
        self._layout.addWidget(self.chart_view)
        
        self.max_accuracy = 0
        self.loss_of_max_accuracy = 0

        for i in range(len(self.child_networks)):
            self.update_plot(i, 0, 0)

    def update_plot(self, child_index, loss, accuracy):
        self.child_series[child_index].clear()
        for i, value in enumerate(self.child_networks[child_index].loss_history):
            self.child_series[child_index].append(i, value)
    
        # Clear existing text items if needed
        scene = self.chart.scene()
        if hasattr(self, 'chart_text_item'):
            if scene:
                scene.removeItem(self.chart_text_item)
        
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.loss_of_max_accuracy = loss
    
        # Create and position the text item
        text = f"Iteration: {self.child_networks[child_index].current_iteration}  --  Loss corresponding: {self.loss_of_max_accuracy:.4f}  --  Max Accuracy: {(self.max_accuracy*100):.2f}%"
        if scene:
            self.chart_text_item = scene.addText(text)
            if self.chart_text_item:
                self.chart_text_item.setFont(QFont("Arial", 12))
                self.chart_text_item.setPos(80, 160)  # Position at the bottom-left of the chart
                self.chart_text_item.setDefaultTextColor(QColor(0, 0, 0))
    
        # Adjust the axis ranges
        self.chart.axes(Qt.Orientation.Horizontal)[0].setRange(0, len(self.child_networks[child_index].loss_history))
        self.chart.axes(Qt.Orientation.Vertical)[0].setRange(0, max(0.3, loss))
    
        self.chart_view.repaint()
        
        if child_index == len(self.child_networks) - 1:
            self.max_accuracy = 0
            self.loss_of_max_accuracy = 0

class FunctionPlot(QWidget):
    def __init__(self, test_dataset, child_networks, child_network_colors):
        super().__init__()
        self.test_dataset = test_dataset
        self.child_networks = child_networks
        self.child_network_colors : List[QColor] = child_network_colors
        self.initUI()

    def initUI(self):
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.true_series = QLineSeries()
        self.true_series.setPen(QPen(Qt.GlobalColor.blue, 1))
        self.child_pred_series = [QLineSeries() for _ in self.child_networks]

        self.chart = QChart()
        self.chart.addSeries(self.true_series)
        for i, pred_series in enumerate(self.child_pred_series):
            pred_series.setPen(QPen(self.child_network_colors[i], 1))
            self.chart.addSeries(pred_series)
        self.chart.createDefaultAxes()
        legend = self.chart.legend()
        if legend:
            legend.hide()

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Set layout size
        self.chart_view.setMinimumSize(300, 200)  # Example size, adjust as needed
        self._layout.addWidget(self.chart_view)
        for i in range(len(self.child_networks)):
            self.update_plot(i)

    def update_plot(self, child_index):
        self.true_series.clear()
        self.child_pred_series[child_index].clear()

        x, y = self.test_dataset[:, 0][:, 0].copy(), self.test_dataset[:, 1][:, 0].copy()
        for i, x_val in enumerate(x):
            y[i] = self.child_networks[child_index].forward(np.array([[x_val]]))[0]

        # True function is a sine wave normalized to the range [0, 2]
        for x_val in np.linspace(0, 2 * np.pi, len(x)):
            self.true_series.append(x_val, (np.sin(x_val) + 1) / 2)

        # Add predicted values
        for i, x_val in enumerate(x):
            self.child_pred_series[child_index].append(x_val, y[i])

        self.chart.axes(Qt.Orientation.Horizontal)[0].setRange(0, 2 * np.pi)
        self.chart.axes(Qt.Orientation.Vertical)[0].setRange(-0.5, 1.5)

        self.chart_view.repaint()


def main(no_children = 1, x_pos = 0, y_pos = 0, seed_weights_bias = None, seed_child_mutation = 0.01, max_iterations = -1, auto_start = False) -> MainWindow:
    app = QApplication(sys.argv)
    # get window position x and y from arguments
    window = MainWindow(node_config, input_indices, output_indices, x_pos, y_pos, seed_weights_bias, seed_child_mutation, max_iterations, no_children, auto_start)
    window.show()
    app.exec()
    return window

if __name__ == '__main__':
    main(seed_weights_bias=seed_weights_bias_global, auto_start=auto_start_training, no_children=15)

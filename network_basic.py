import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial import distance_matrix

class SpatialNeuron:
    def __init__(self, index, position, num_connections=3):
        self.index = index
        self.position = np.array(position)
        self.is_input = False
        self.is_output = False
        self.is_hidden = False
        self.num_connections = num_connections
        self.weights = None
        self.bias = np.zeros(1)
        self.outgoing_connections : List[SpatialNeuron] = []
        self.incoming_connections : List[SpatialNeuron] = []
        
    def initialize_weights_and_biases(self):
        self.weights = np.random.randn(len(self.incoming_connections))
        # self.weights = np.ones(len(self.incoming_connections))
        self.bias = np.random.randn(1)
        # print("Neuron at", self.position, "has weights", self.weights, "and bias", self.bias)
        
    def forward(self, inputs):
        # print("Forwarding neuron at", self.position, "with inputs", inputs, "and weights", self.weights)
        z = np.dot(inputs, self.weights) + self.bias
        # print("Z", z)
        a = self.sigmoid(z)
        # output nodes don't have spatial attention
        if self.is_output:
            return a
        # distance attention
        # output_distances = [np.linalg.norm(self.position - conn.position) for conn in self.outgoing_connections]
        # output_distances = np.array(output_distances).reshape(-1, 1)
        # a = a * (1 / output_distances)
        a = a * [[1] for conn in self.outgoing_connections]
        return a
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

class SpatialNetwork:
    def __init__(self, node_positions, input_indices, output_indices):
        self.neurons = [SpatialNeuron(ind, pos) for ind, pos in enumerate(node_positions)]
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.dmin = 1.1  # minimum distance for spatial attention
        self.dmax = 2  # maximum distance for spatial attention
        self.hidden_indices = [i for i in range(len(self.neurons)) if i not in input_indices and i not in output_indices]
        
        for i in input_indices:
            self.neurons[i].is_input = True
        for i in output_indices:
            self.neurons[i].is_output = True
        for i in self.hidden_indices:
            self.neurons[i].is_hidden = True
        
        self.initialize_connections()
        # activations is a matrix representing activation value sent from ith neuron to jth neuron
        self.activations = np.zeros((len(self.neurons), len(self.neurons)))
        self.learning_rate = 0.5
        self.position_learning_rate = 1
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
            nearest = sorted(valid_targets, key=lambda j: self.distances[i][j])[:self.neurons[i].num_connections]
            for j in nearest:
                # connect only if the node is at the same or more X position
                # also if the node is not already connected
                # print(f"Connecting {self.neurons[i].position} to {self.neurons[j].position}")
                if self.neurons[j].position[0] >= self.neurons[i].position[0] and self.neurons[j] not in self.neurons[i].incoming_connections:
                    self.neurons[j].incoming_connections.append(self.neurons[i])
                    self.neurons[i].outgoing_connections.append(self.neurons[j])
        
        # Initialize weights for hidden and output nodes
        for i in self.hidden_indices + self.output_indices:
            self.neurons[i].initialize_weights_and_biases()
    
    def update_distances(self):
        self.distances = distance_matrix([n.position for n in self.neurons], [n.position for n in self.neurons])
        # self distances are inf
        np.fill_diagonal(self.distances, np.inf)

    def forward(self, inputs):
        # reset activations, default value is -1
        self.activations.fill(-1)
        
        # set the activations for all hidden neurons connected to the input neurons
        for index, inp_node_index in enumerate(self.input_indices):
            for conn in self.neurons[inp_node_index].outgoing_connections:
                self.activations[inp_node_index][conn.index] = inputs[index]
        
        # this is a directed acyclic graph
        # there can be outgoing connections that are dependent on unknown activations
        # so we can do a recursion from the output nodes instead

        def activate_neuron(i):
            neuron : SpatialNeuron = self.neurons[i]
            inputs = np.array([self.activations[conn.index][i] for conn in neuron.incoming_connections])
            if not np.all(inputs != -1):
                # if not all incoming connections are activated
                # recursively activate the incoming connections
                for conn in neuron.incoming_connections:
                    if self.activations[conn.index][i] == -1:
                        activate_neuron(conn.index)
                inputs = np.array([self.activations[conn.index][i] for conn in neuron.incoming_connections])

            # if the code reaches here, all the incoming connections are activated
            # forward the neuron, save the activation value and return it
            # if the neuron is an output node, return the forwarded output
            output_activations = neuron.forward(inputs)
            if neuron.is_output:
                return output_activations
            for idx, conn in enumerate(neuron.outgoing_connections):
                # set the activation value for the outgoing connections
                self.activations[i][conn.index] = output_activations[idx][0]

        # return the activations of the output nodes
        out = np.zeros(len(self.output_indices))
        for idx, output_index in enumerate(self.output_indices):
            out[idx] = activate_neuron(output_index)[0]
        
        if self.current_iteration % 10 == 0:
            print("Output", out)

        # print("Activations")
        # r_act = np.round(self.activations, 2)
        # print(r_act[1][3], "\t", r_act[2][3], "\t", r_act[3][5], "\t", out[0])
        # print(r_act[0][2], "\t\t", r_act[2][4], "\t", out[1])
        return out
    
    def backward(self, x, y_true, y_pred):
        gradients = {i: {'weights': np.zeros_like(n.weights) if n.weights is not None else 0, 
                         'bias': 0,
                         'position': np.zeros_like(n.position)} 
                     for i, n in enumerate(self.neurons)}
        
        # Compute output gradients
        output_deltas = {}
        for i, (y_true, y_pred) in enumerate(zip(y_true, y_pred)):
            output_index = self.output_indices[i]
            d_loss = y_pred - y_true
            d_sigmoid = self.neurons[output_index].sigmoid_derivative(y_pred)
            output_deltas[output_index] = d_loss * d_sigmoid
        
        def compute_gradient(i):
            neuron = self.neurons[i]
            # print("Computing gradient for neuron", neuron.position, " i", i)
            if i in output_deltas:
                # print("got value from output deltas")
                delta = output_deltas[i]
            else:
                delta = [0]

                for conn in neuron.outgoing_connections:
                    weight_index = conn.incoming_connections.index(neuron)
                    # out_delta = compute_gradient(conn.index) # a^(l-1)
                    out_delta = gradients[conn.index]['bias'] # a^(l) - a^(l+1)
                    # print("Out delta", out_delta)
                    delta += conn.weights[weight_index] * out_delta # w^(l) * a^(l-1)
                for activation in self.activations[:, i]:
                    if activation != -1:
                        delta = delta * np.array(neuron.sigmoid_derivative(activation)) # w^(l) * a^(l-1) * f'(z^(l))

                    # activation = self.activations[i][conn.index] # a^(l)
                    # delta += out_delta * np.array(neuron.sigmoid_derivative(activation)) # f'(z^(l)) * (a^(l) - a^(l+1))
                # for activation in self.activations[:, i]:
                #     if activation != -1:
                #         delta = delta * np.array(neuron.sigmoid_derivative(activation)) # w^(l) * a^(l-1) * f'(z^(l))
            
            # print("Delta", delta)
            if not neuron.is_input:
                gradients[i]['bias'] = delta
                # print([self.activations[n.index][i] for n in neuron.incoming_connections])
                gradients[i]['weights'] = np.array([self.activations[n.index][i] for n in neuron.incoming_connections]) * delta
                
                # Position gradient calculation remains the same
                # All nodes pull their incoming connections, if possible and benefitial
                # for conn in neuron.incoming_connections:
                #     distance = self.distances[i][conn.index]
                #     if distance < self.dmin:
                #         d_vector = 0.001 * (neuron.position - conn.position) / distance  # Repulsive force
                #         # print(f"neuron {conn.position} repulsive")
                #     elif distance > self.dmax:
                #         d_vector = -0.001 * (neuron.position - conn.position) / distance  # Attractive force
                #         # print(f"neuron {conn.position} attractive")
                #     else:
                #         d_vector = -1*(delta * self.activations[conn.index][neuron.index]) * (neuron.position - conn.position) / (1 + distance)**2
                #         # print(f"neuron {neuron.position} conn {conn.position} d_vector {d_vector}")
                #     gradients[conn.index]['position'] = gradients[conn.index]['position'] + d_vector

            return delta
        
        for i in reversed(range(len(self.neurons))):
            if not self.neurons[i].is_input:
                compute_gradient(i)
        
        return gradients

    def train_step(self, dataset_step):
        global dataset

        total_loss = 0
        total_correct = 0
        gradients = {i: {'weights': np.zeros_like(n.weights) if n.weights is not None else 0, 
                         'bias': 0,
                         'position': np.zeros_like(n.position)} 
                     for i, n in enumerate(self.neurons)}
        
        for x, y_true in dataset_step:
            y_pred = self.forward(x)
            self.update_distances()
            # print("Predicted", y_pred)
            # print(self.activations)
            single_row_gradients = self.backward(x, y_true, y_pred)
            # for i, grad in single_row_gradients.items():
            #     if i == 2:
                # print(f"p={grad['position']}")
                        #   , b={grad['bias']}, p={grad['position']}")
            for i, grad in single_row_gradients.items():
                for key in grad:
                    gradients[i][key] = gradients[i][key] + grad[key]

            # change shape of y_pred to match y_true
            y_pred = np.array(y_pred).reshape(-1)

            # Compute loss and accuracy
            prediction_loss = np.mean((y_pred - y_true) ** 2)
            # distance_penalty = np.sum(np.maximum(0, self.dmin - self.distances))
            # print("Prediction loss", prediction_loss)
            loss = prediction_loss
            total_loss += loss
            if self.current_iteration % 10 == 0:
                print("Predictions : ", np.round(y_pred), "True : ", y_true)
            total_correct += np.all(np.round(y_pred) == y_true)

        # Average gradients
        for i in gradients:
            for key in gradients[i]:
                # if key != 'position':
                gradients[i][key] = gradients[i][key] / len(dataset_step)
        # print("Gradients", gradients[2]['position'])
        
        for i, neuron in enumerate(self.neurons):
            if not neuron.is_input:
                neuron.weights -= self.learning_rate * gradients[i]['weights']
                neuron.bias -= self.learning_rate * gradients[i]['bias']
                neuron.position = neuron.position + (gradients[i]['position'] * self.position_learning_rate)
        
        accuracy = total_correct / len(dataset)
        loss_percentage = total_loss / len(dataset)
        self.loss_history.append(loss_percentage)
        self.current_iteration += 1
        
        # print(f"Iteration {self.current_iteration}: Loss {loss_percentage:.4f}, Accuracy {accuracy:.2f}")
        return loss_percentage, accuracy



# Visualization setup
fig: plt.figure
ax1: plt.Axes
ax2: plt.Axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(bottom=0.2)

# Example usage
node_positions = [(0, 0), (0, 2), (1, 0.5), (1, 1.5), (2, 0), (2, 2)]
input_indices = [0, 1]
output_indices = [4, 5]
network = SpatialNetwork(node_positions, input_indices, output_indices)

# Main plot
input_nodes = [node_positions[i] for i in input_indices]
input_plot = ax1.scatter([p[0] for p in input_nodes], [p[1] for p in input_nodes], c='blue', s=300, zorder=2)
output_nodes = [node_positions[i] for i in output_indices]
output_plot = ax1.scatter([p[0] for p in output_nodes], [p[1] for p in output_nodes], c='green', s=300, zorder=2)
neuron_nodes = [node_positions[i] for i in network.hidden_indices]
neuron_plot = ax1.scatter([p[0] for p in neuron_nodes], [p[1] for p in neuron_nodes], c='orange', s=300, zorder=3)
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

ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 2.5)
ax1.axis('off')

# Loss plot
loss_line, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 0.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.grid(True)
# also show the current loss value
loss_accuracy_text = ax2.text(0.02, 0.95, '',
                        verticalalignment='top', horizontalalignment='left', transform=ax2.transAxes)

def update_plot():
    for annotation in ax1.texts:
        annotation.remove()

    for i, neuron in enumerate(network.neurons):
        # calculate incoming connection distances
        incoming_distances = []
        for conn in neuron.incoming_connections:
            incoming_distances.append(round(float(network.distances[i][conn.index]), 2))

        if i >= len(ax1.texts):
            # ax1.text(neuron.position[0], neuron.position[1], f'N{i}\nw={neuron.weights}\nb={neuron.bias}\nd={incoming_distances}', ha='center', va='center')
            ax1.text(neuron.position[0], neuron.position[1], f'N{i}', ha='center', va='center')
        else:
            ax1.texts[i].set_position(neuron.position)
            # ax1.texts[i].set_text(f'N{i}\nw={neuron.weights}\nb={neuron.bias}\nd={incoming_distances}')
            ax1.texts[i].set_text(f'N{i}')
    
    neuron_plot.set_offsets([n.position for n in network.neurons if n.is_hidden])
    
    line_index = 0
    for neuron in network.neurons:
        for conn in neuron.outgoing_connections:
            lines[line_index].set_data([neuron.position[0], conn.position[0]], [neuron.position[1], conn.position[1]])
            add_arrow(lines[line_index])
            line_index += 1
    
    plt.draw()
    
def update_loss_plot(loss, accuracy):
    loss_accuracy_text.set_text(f'Iteration: {network.current_iteration}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')
    loss_line.set_data(range(len(network.loss_history)), network.loss_history)
    ax2.set_xlim(0, max(100, len(network.loss_history)))
    ax2.set_ylim(0, 0.8)
    plt.draw()

dataset = np.array([
    [[0, 0], [0, 0]],
    [[0, 1], [1, 0]],
    [[1, 0], [1, 0]],
    [[1, 1], [0, 1]]
    # [[0, 0], [1, 0]],
    # [[0, 1], [1, 0]],
    # [[1, 0], [1, 0]],
    # [[1, 1], [1, 0]]
]) / 1.0 # Assuming binary data, divide by 1 to normalize

# for neuron in network.neurons:
#     print("Neuron at", neuron.position, "has incoming connections", [conn.position for conn in neuron.incoming_connections])
#     print("and outgoing connections", [conn.position for conn in neuron.outgoing_connections])

step_size = len(dataset)
def on_click(event):
    # step_no = network.current_iteration//len(dataset)
    # dataset_step = dataset[step_no:step_no+step_size]
    loss, accuracy = network.train_step(dataset)
    if network.current_iteration % 10 == 0:
        update_loss_plot(loss, accuracy)
    # if network.current_iteration % 10 == 0:
    #     update_plot()

button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(button_ax, 'Next Iteration')
button.on_clicked(on_click)

# play pause training with 100ms interval
play_button = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Play')
pause_button = Button(plt.axes([0.6, 0.05, 0.1, 0.075]), 'Pause')
pause = False

def on_play(event):
    global pause
    pause = False
    while not pause:
        on_click(None)
        plt.pause(0.01)
        
def on_pause(event):
    global pause
    pause = True

play_button.on_clicked(on_play)
pause_button.on_clicked(on_pause)

plt.show()

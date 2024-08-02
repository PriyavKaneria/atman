import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial import distance_matrix

# config
enable_spatial_attention = True
show_weights_biases = False
show_distances = False

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
        a = self.activation(z)
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
        return self.sigmoid(x)
    
    def activation_derivative(self, x):
        return self.sigmoid_derivative(x)
    
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
        self.learning_rate = 0.9
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
                print(f"Connecting {self.neurons[i].position} to {self.neurons[j].position}")
                if self.neurons[j].position[0] >= self.neurons[i].position[0] and self.neurons[j] not in self.neurons[i].incoming_connections:
                    self.neurons[j].incoming_connections.append(self.neurons[i])
                    self.neurons[i].outgoing_connections.append(self.neurons[j])
                if len(self.neurons[i].outgoing_connections) == self.neurons[i].num_connections:
                    break
        
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
        
        # if self.current_iteration % 10 == 0:
        #     print("Output", out)

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
            d_sigmoid = self.neurons[output_index].activation_derivative(y_pred)
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
                        delta = delta * np.array(neuron.activation_derivative(activation)) # w^(l) * a^(l-1) * f'(z^(l))

            # print("Delta", delta)
            if not neuron.is_input:
                gradients[i]['bias'] = delta
                # print([self.activations[n.index][i] for n in neuron.incoming_connections])
                gradients[i]['weights'] = np.array([self.activations[n.index][i] for n in neuron.incoming_connections]) * delta
                
                if enable_spatial_attention:
                    # Position gradient calculation remains the same
                    # All nodes pull their incoming connections, if possible and benefitial
                    for conn in neuron.incoming_connections:
                        distance = self.distances[i][conn.index]
                        if distance < self.dmin:
                            d_vector = -1 * 0.01 * (neuron.position - conn.position) / distance  # Repulsive force
                            # print(f"neuron {conn.position} repulsive")
                        elif distance > self.dmax:
                            d_vector = 0.01 * (neuron.position - conn.position) / distance  # Attractive force
                            # print(f"neuron {conn.position} attractive")
                        else:
                            d_vector = -1*(delta * self.activations[conn.index][neuron.index]) * (neuron.position - conn.position) / (1 + distance)**2
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
            # print(self.activations)
            single_row_gradients = self.backward(x, y_true, y_pred)
            # for i, grad in single_row_gradients.items():
            #     if i == 2:
                # print(f"p={grad['position']}")
                        #   , b={grad['bias']}, p={grad['position']}")
            for i, grad in single_row_gradients.items():
                for key in grad:
                    gradients[i][key] = gradients[i][key] + grad[key]

        for x, y_true in test_dataset:
            y_pred = self.forward(x)
            # change shape of y_pred to match y_true
            y_pred = np.array(y_pred).reshape(-1)

            # Compute loss and accuracy
            prediction_loss = np.mean((y_pred - y_true) ** 2)
            # distance_penalty = np.sum(np.maximum(0, self.dmin - self.distances))
            # print("Prediction loss", prediction_loss)
            loss = prediction_loss
            total_loss += loss
            # if self.current_iteration % 10 == 0:
            #     print("Predictions : ", np.round(y_pred, 2), "True : ", np.round(y_true, 2))
            total_correct += np.all(np.round(y_pred, 2) == np.round(y_true, 2))

        # Average gradients
        for i in gradients:
            for key in gradients[i]:
                # if key != 'position':
                gradients[i][key] = gradients[i][key] / len(train_dataset)
        # print("Gradients", gradients[2]['position'])
        
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


# Visualization setup
# Correcting the one-liner to create the desired subplot layout
fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [1, 1], 'wspace': 0.3})

# Merging the first row to create a single plot spanning two columns
fig.delaxes(axs[0, 0])
fig.delaxes(axs[0, 1])
ax1 = fig.add_subplot(2, 2, (1, 2))

plt.subplots_adjust(bottom=0.2)
# Define node positions for a spatial network to predict sine waves
# Input layer (1 node), two hidden layers (3 nodes each), output layer (1 node)
node_positions = [
    (0, 0.5),     # Input node
    (1, 0),       # Hidden layer 1
    (1.2, 0.5),
    (1, 1),
    (2, 0),       # Hidden layer 2
    (2.2, 0.5),
    (2, 1),
    (3, 0.5)      # Output node
]

input_indices = [0]
output_indices = [7]

# Create the SpatialNetwork instance
network = SpatialNetwork(node_positions, input_indices, output_indices)

# Generate datasets
def generate_sine_data(num_samples, start, end):
    x = np.linspace(start, end, num_samples)
    y = np.sin(x)
    return np.column_stack((x.reshape(-1, 1), y.reshape(-1, 1)))

# Training dataset: Cover the full range of sine wave
train_dataset = generate_sine_data(100, 0, 2*np.pi)
train_dataset = np.expand_dims(train_dataset, axis=2)

# Test dataset: Include values outside the training range to test generalization
test_dataset = generate_sine_data(20, -np.pi/2, 2.5*np.pi)
test_dataset = np.expand_dims(test_dataset, axis=2)

# Normalize the datasets
max_val = max(np.max(np.abs(train_dataset)), np.max(np.abs(test_dataset)))
train_dataset /= max_val
test_dataset /= max_val

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

ax1.set_xlim(min([p[0] for p in node_positions]) - 0.5, max([p[0] for p in node_positions]) + 0.5)
ax1.set_ylim(min([p[1] for p in node_positions]) - 0.5, max([p[1] for p in node_positions]) + 0.5)
ax1.axis('off')

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
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)
ax3.plot(x, y)

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
        if i >= len(ax1.texts):
            ax1.text(neuron.position[0], neuron.position[1], neuron_label_text, ha='center', va='center')
        else:
            ax1.texts[i].set_position(neuron.position)
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
    loss_accuracy_text.set_text(f'Iteration: {network.current_iteration}, Loss: {loss:.4f}, Accuracy: {(accuracy*100):.1f}%')
    loss_line.set_data(range(len(network.loss_history)), network.loss_history)
    ax2.set_xlim(0, max(100, len(network.loss_history)))
    ax2.set_ylim(0, curr_max_loss)
    plt.draw()


def update_function_plot():
    # plot the predicted sine wave
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.zeros_like(x)
    for i, x_val in enumerate(x):
        y[i] = network.forward(np.array([[x_val]]))
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
    # step_no = network.current_iteration//len(dataset)
    # dataset_step = dataset[step_no:step_no+step_size]
    loss, accuracy = network.train_step(train_dataset, test_dataset)
    plot_update_step = 1
    if network.current_iteration % plot_update_step == 0:
        update_loss_plot(loss, accuracy)
        update_function_plot()
    if network.current_iteration % plot_update_step == 0:
        update_plot()

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

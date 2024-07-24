import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial import distance_matrix

class SpatialNeuron:
    def __init__(self, position, num_connections=3):
        self.position = np.array(position)
        self.num_connections = num_connections
        self.weights = None
        self.bias = np.random.randn(1)
        self.outgoing_connections : List[SpatialNeuron] = []
        self.incoming_connections : List[SpatialNeuron] = []
        self.is_input = False
        self.is_output = False
        
    def initialize_weights(self):
        self.weights = np.random.randn(len(self.incoming_connections))

        
    def forward(self, inputs):
        if self.is_input:
            return inputs[0]
        # print("Forwarding neuron at", self.position, "with inputs", inputs, "and weights", self.weights)
        z = np.dot(inputs, self.weights) + self.bias
        a = self.sigmoid(z)
        # distance attention
        output_distances = [np.linalg.norm(self.position - conn.position) for conn in self.outgoing_connections]
        output_distances = np.array(output_distances).reshape(-1, 1)
        a = a * (1 / output_distances)
        return a
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class SpatialNetwork:
    def __init__(self, node_positions, input_indices, output_indices):
        self.neurons = [SpatialNeuron(pos) for pos in node_positions]
        self.input_indices = input_indices
        self.output_indices = output_indices
        self.hidden_indices = [i for i in range(len(self.neurons)) if i not in input_indices and i not in output_indices]
        
        for i in input_indices:
            self.neurons[i].is_input = True
        for i in output_indices:
            self.neurons[i].is_output = True
        
        self.initialize_connections()
        self.activations = {}
        self.learning_rate = 0.1
        self.position_learning_rate = 0.01
        self.current_iteration = 0
        self.loss_history = []
        
    def initialize_connections(self):
        distances = distance_matrix([n.position for n in self.neurons], [n.position for n in self.neurons])
        
        # Connect input nodes to nearest hidden nodes
        for i in self.input_indices:
            nearest_hidden = sorted(self.hidden_indices, key=lambda j: distances[i][j])[:self.neurons[i].num_connections]
            self.neurons[i].outgoing_connections = [self.neurons[j] for j in nearest_hidden]
            for j in nearest_hidden:
                self.neurons[j].incoming_connections.append(self.neurons[i])
        
        # Connect hidden nodes to nearest hidden or output nodes
        for i in self.hidden_indices:
            valid_targets = self.hidden_indices + self.output_indices
            valid_targets.remove(i)  # Exclude self-connection
            nearest = sorted(valid_targets, key=lambda j: distances[i][j])[:self.neurons[i].num_connections]
            self.neurons[i].outgoing_connections = [self.neurons[j] for j in nearest]
            for j in nearest:
                # if not self.neurons[j].is_output:
                self.neurons[j].incoming_connections.append(self.neurons[i])
        
        # Initialize weights for hidden and output nodes
        for i in self.hidden_indices + self.output_indices:
            self.neurons[i].initialize_weights()
    
    def forward(self, x):
        self.activations = {i: -1 for i in range(len(self.neurons))}
        for i, inp in zip(self.input_indices, x):
            self.activations[i] = inp
        
        def activate_neuron(i):
            if self.activations[i] != -1:
                return self.activations[i]
            neuron = self.neurons[i]
            inputs = np.array([activate_neuron(self.neurons.index(conn)) for conn in neuron.incoming_connections])
            outgoing_activations = neuron.forward(inputs)
            for j, conn in enumerate(neuron.outgoing_connections):
                self.activations[self.neurons.index(conn)] = outgoing_activations[j]
            return self.activations[i]
        
        return [activate_neuron(i) for i in self.output_indices]
    
    def backward(self, x, y_true, y_pred):
        gradients = {i: {'weights': np.zeros_like(n.weights) if n.weights is not None else None, 
                         'bias': 0,
                         'position': np.zeros_like(n.position)} 
                     for i, n in enumerate(self.neurons)}
        
        # Compute output gradients
        output_deltas = {}
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            output_index = self.output_indices[i]
            output_deltas[output_index] = (pred - true) * self.neurons[output_index].sigmoid_derivative(pred)
        
        def compute_gradient(i):
            neuron = self.neurons[i]
            if i in output_deltas:
                delta = output_deltas[i]
            else:
                delta = 0
                for j, conn in enumerate(neuron.outgoing_connections):
                    out_index = self.neurons.index(conn)
                    weight_index = conn.incoming_connections.index(neuron)
                    out_delta = compute_gradient(out_index)
                    delta += conn.weights[weight_index] * out_delta
                
                delta *= neuron.sigmoid_derivative(self.activations[i])
            
            if not neuron.is_input:
                gradients[i]['bias'] = delta
                gradients[i]['weights'] = delta * [self.activations[self.neurons.index(n)] for n in neuron.incoming_connections]
                
                # Position gradient calculation remains the same
                for j, conn in enumerate(neuron.incoming_connections):
                    d_vector = (conn.position - neuron.position) / np.linalg.norm(conn.position - neuron.position)
                    gradients[i]['position'] = np.add(gradients[i]['position'], delta * neuron.weights[j] * d_vector, out=gradients[i]['position'], casting='unsafe')
                    # print("Gradient position for neuron", neuron.position, "from", conn.position, "is", delta * neuron.weights[j] * d_vector)

            return delta
        
        for i in reversed(range(len(self.neurons))):
            if not self.neurons[i].is_input:
                compute_gradient(i)
        
        return gradients

    def train_step(self, x, y_true):
        y_pred = self.forward(x)
        gradients = self.backward(x, y_true, y_pred)
        print("Gradients", gradients[2]["position"])
        
        for i, neuron in enumerate(self.neurons):
            if not neuron.is_input:
                neuron.weights -= self.learning_rate * gradients[i]['weights']
                neuron.bias -= self.learning_rate * gradients[i]['bias']
            neuron.position = np.add(neuron.position, self.position_learning_rate * gradients[i]['position'], out=neuron.position, casting='unsafe')
        
        loss = np.mean((np.array(y_pred) - np.array(y_true)) ** 2)
        accuracy = np.mean(np.round(y_pred) == y_true)
        self.loss_history.append(loss)
        self.current_iteration += 1
        
        print(f"Iteration {self.current_iteration}: Loss {loss:.4f}, Accuracy {accuracy:.2f}")
        return loss



# Visualization setup
fig: plt.figure
ax1: plt.Axes
ax2: plt.Axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(bottom=0.2)

# Example usage
node_positions = [(0, 0), (0, 1), (1, 0.5), (2, 0), (2, 1)]
input_indices = [0, 1]
output_indices = [3, 4]
network = SpatialNetwork(node_positions, input_indices, output_indices)

# Main plot
ax1.scatter([p[0] for p in node_positions], [p[1] for p in node_positions], c='orange', s=500, zorder=2)
lines = []
for neuron in network.neurons:
    for conn in neuron.outgoing_connections:
        line, = ax1.plot([neuron.position[0], conn.position[0]], 
                         [neuron.position[1], conn.position[1]], 
                         'gray', linewidth=1, zorder=1)
        # add_arrow(line)
        lines.append(line)

ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 1.5)
ax1.axis('off')

# Loss plot
loss_line, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.grid(True)

def update_plot():
    for i, neuron in enumerate(network.neurons):
        if i >= len(ax1.texts):
            ax1.text(neuron.position[0], neuron.position[1], f'N{i}\nw={neuron.weights}\nb={neuron.bias}', ha='center', va='center')
        else:
            ax1.texts[i].set_position(neuron.position)
            ax1.texts[i].set_text(f'N{i}\nw={neuron.weights}\nb={neuron.bias}')
    
    line_index = 0
    for neuron in network.neurons:
        for conn in neuron.outgoing_connections:
            lines[line_index].set_data([neuron.position[0], conn.position[0]], [neuron.position[1], conn.position[1]])
            # add_arrow(line)
            line_index += 1
    
    loss_line.set_data(range(len(network.loss_history)), network.loss_history)
    ax2.set_xlim(0, max(100, len(network.loss_history)))
    ax2.set_ylim(0, max(1, max(network.loss_history)))
    
    plt.draw()

dataset = np.array([
    # [[0, 0], [0, 0]],
    # [[0, 1], [1, 0]],
    # [[1, 0], [1, 0]],
    # [[1, 1], [0, 1]]
    [[0, 0], [1, 0]],
    [[0, 1], [1, 0]],
    [[1, 0], [1, 0]],
    [[1, 1], [1, 0]]
])

# for neuron in network.neurons:
#     print("Neuron at", neuron.position, "has incoming connections", [conn.position for conn in neuron.incoming_connections])
#     print("and outgoing connections", [conn.position for conn in neuron.outgoing_connections])

def on_click(event):
    x, y_true = dataset[network.current_iteration % len(dataset)]
    loss = network.train_step(x, y_true)
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

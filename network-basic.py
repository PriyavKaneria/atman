import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class SpatialNeuron:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
        self.position = np.random.randn(input_dim + output_dim)
        
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        # Calculate distances
        input_distances = np.linalg.norm(self.position[:self.input_dim] - x, axis=0)
        output_distances = np.linalg.norm(self.position[self.input_dim:] - z, axis=0)
        # Apply activation with distance modulation
        a = self.sigmoid(z) * (1 / (1 + input_distances.mean())) * (1 / (1 + output_distances.mean()))
        return a, z, input_distances, output_distances
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

class SpatialNetwork:
    def __init__(self):
        self.neuron = SpatialNeuron(2, 2)
        self.learning_rate = 0.1
        self.position_learning_rate = 0.01
        self.dataset = np.array([
            [[0, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[1, 1], [0, 1]]
        ])
        self.current_iteration = 0
        
    def train_step(self):
        x, y_true = self.dataset[self.current_iteration % 4]
        y_pred, z, input_distances, output_distances = self.neuron.forward(x)
        
        # Compute gradients
        d_loss = y_pred - y_true
        d_sigmoid = self.neuron.sigmoid_derivative(y_pred)
        d_weights = np.outer(x, d_loss * d_sigmoid)
        d_bias = d_loss * d_sigmoid
        
        # Update weights and bias
        self.neuron.weights -= self.learning_rate * d_weights
        self.neuron.bias -= self.learning_rate * d_bias
        
        # Update neuron position
        d_position_input = -d_loss * d_sigmoid * y_pred * (x - self.neuron.position[:2]) / (1 + input_distances)**3
        d_position_output = -d_loss * d_sigmoid * y_pred * (z - self.neuron.position[2:]) / (1 + output_distances)**3
        self.neuron.position[:2] -= self.position_learning_rate * d_position_input
        self.neuron.position[2:] -= self.position_learning_rate * d_position_output
        
        self.current_iteration += 1
        
        print(f"Iteration {self.current_iteration}: Input {x} -> Predicted {y_pred.round(2)} (True {y_true})")
        return x, y_true, y_pred, input_distances, output_distances

# Visualization setup
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)
network = SpatialNetwork()

scatter = ax.scatter([], [], c='r', s=100)
input_lines = [ax.plot([], [], 'b-')[0] for _ in range(2)]
output_lines = [ax.plot([], [], 'g-')[0] for _ in range(2)]
input_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top')
output_text = ax.text(0.05, 0.85, '', transform=ax.transAxes, verticalalignment='top')
weight_text = ax.text(0.05, 0.75, '', transform=ax.transAxes, verticalalignment='top')
distance_text = ax.text(0.05, 0.55, '', transform=ax.transAxes, verticalalignment='top')

def update_plot(x, y_true, y_pred, input_distances, output_distances):
    neuron_pos = network.neuron.position
    scatter.set_offsets(neuron_pos.reshape(1, -1))
    
    for i, line in enumerate(input_lines):
        line.set_data([x[i], neuron_pos[i]], [0, 1])
    
    for i, line in enumerate(output_lines):
        line.set_data([neuron_pos[i+2], y_pred[i]], [1, 2])
    
    input_text.set_text(f"Input: {x}")
    output_text.set_text(f"Output: Predicted {y_pred.round(2)}, True {y_true}")
    weight_text.set_text(f"Weights:\n{network.neuron.weights.round(2)}\nBias: {network.neuron.bias.round(2)}")
    distance_text.set_text(f"Distances:\nInput: {input_distances.round(2)}\nOutput: {output_distances.round(2)}")
    
    ax.relim()
    ax.autoscale_view()
    plt.draw()

def on_click(event):
    x, y_true, y_pred, input_distances, output_distances = network.train_step()
    update_plot(x, y_true, y_pred, input_distances, output_distances)

button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
nxt_button = Button(button_ax, 'Next')
nxt_button.on_clicked(on_click)

# play pause training with 100ms interval
play_button = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Play')
pause_button = Button(plt.axes([0.6, 0.05, 0.1, 0.075]), 'Pause')
pause = False

def on_play(event):
    global pause
    pause = False
    while not pause:
        on_click(None)
        plt.pause(0.1)
        
def on_pause(event):
    global pause
    pause = True

play_button.on_clicked(on_play)
pause_button.on_clicked(on_pause)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from solve_distance_to_position import find_new_coordinates

class SpatialNeuron:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(1)
        self.position = np.array([1,0.5])  # 2D position for visualization
        
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        input_distances = np.array([
            np.linalg.norm(self.position - np.array([0, 0])),  # d1
            np.linalg.norm(self.position - np.array([0, 1])),  # d2
        ])
        output_distances = np.array([
            np.linalg.norm(self.position - np.array([2, 0])),  # d3
            np.linalg.norm(self.position - np.array([2, 1]))   # d4
        ])
        a = self.sigmoid(z) * (1 / (1 + output_distances))
        return a, z, input_distances, output_distances
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

class SpatialNetwork:
    def __init__(self):
        self.neuron = SpatialNeuron(2, 2)
        self.learning_rate = 0.5
        self.position_learning_rate = 0.2
        self.dataset = np.array([
            # [[0, 0], [0, 0]],
            # [[0, 1], [1, 0]],
            # [[1, 0], [1, 0]],
            # [[1, 1], [0, 1]]
            [[0, 0], [1, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [1, 0]],
            [[1, 1], [1, 0]]
        ])
        self.current_iteration = 0
        self.loss_history = []
        
    def train_step(self):
        total_loss = 0
        total_correct = 0
        gradients = {'weights': np.zeros_like(self.neuron.weights),
                     'bias': np.zeros_like(self.neuron.bias),
                     'position': np.zeros_like(self.neuron.position)}
        
        for x, y_true in self.dataset:
            y_pred, z, inp_distances, out_distances = self.neuron.forward(x)
            
            # Compute loss and accuracy
            loss = np.mean((y_pred - y_true) ** 2)
            total_loss += loss
            total_correct += np.all(np.round(y_pred) == y_true)
            
            # Compute gradients
            d_loss = y_pred - y_true
            d_sigmoid = self.neuron.sigmoid_derivative(y_pred)
            d_weights = np.outer(x, d_loss * d_sigmoid)
            d_bias = np.average(d_loss * d_sigmoid)
            
            gradients['weights'] += d_weights
            gradients['bias'] += d_bias
            
            # Gradient for position
            d_position = np.zeros(2)
            
            # compute the gradient for the distance from output nodes
            # delta d3
            delta_s1 = -np.sum(d_loss * d_sigmoid * y_pred) * (np.array([2,0]) - self.neuron.position) / (1 + out_distances[0])**3
            s1_prime = out_distances[0] + delta_s1

            # delta d4
            delta_s2 = -np.sum(d_loss * d_sigmoid * y_pred) * (np.array([2,1]) - self.neuron.position) / (1 + out_distances[1])**3
            s2_prime = out_distances[1] + delta_s2

            # compute the updated position based on the new distances (s1_prime, s2_prime)
            new_positions = find_new_coordinates(2, 0, s1_prime, 2, 1, s2_prime)
            # get the closest position to the current position
            pos = new_positions[np.argmin([np.linalg.norm(self.neuron.position - np.array(p)) for p in new_positions])]
            gradients['position'] = pos - self.neuron.position
        
        # Update weights, bias, and position
        self.neuron.weights -= self.learning_rate * gradients['weights'] / len(self.dataset)
        self.neuron.bias -= self.learning_rate * gradients['bias'] / len(self.dataset)
        self.neuron.position -= self.position_learning_rate * gradients['position'] / len(self.dataset)
        
        accuracy = total_correct / len(self.dataset)
        self.loss_history.append(total_loss / len(self.dataset))
        self.current_iteration += 1
        
        print(f"Iteration {self.current_iteration}: Loss {total_loss/len(self.dataset):.4f}, Accuracy {accuracy:.2f}")
        return accuracy

# Visualization setup
fig: plt.Figure
ax1: plt.Axes
ax2: plt.Axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(bottom=0.2)
network = SpatialNetwork()

# Main plot
input_nodes = ax1.scatter([0, 0], [0, 1], c='orange', s=500, zorder=2)
output_nodes = ax1.scatter([2, 2], [0, 1], c='orange', s=500, zorder=2)
neuron = ax1.scatter([1], [0.5], c='black', s=500, zorder=3)
lines = [
    ax1.plot([0,1], [0,0.5], 'gray', linewidth=2, zorder=1)[0],
    ax1.plot([0,1], [1,0.5], 'gray', linewidth=2, zorder=1)[0],
    ax1.plot([1,2], [0.5,1], 'gray', linewidth=2, zorder=1)[0],
    ax1.plot([1,2], [0.5,0], 'gray', linewidth=2, zorder=1)[0]
]

ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 1.5)
ax1.axis('off')

# Text annotations
texts = {
    'a1': ax1.text(-0.1, 0, '', ha='right', va='center'),
    'a2': ax1.text(-0.1, 1, '', ha='right', va='center'),
    'a3': ax1.text(2.1, 0, '', ha='left', va='center'),
    'a4': ax1.text(2.1, 1, '', ha='left', va='center'),
    'd1': ax1.text(0.5, 0.1, '', ha='center', va='bottom'),
    'd2': ax1.text(0.5, 0.9, '', ha='center', va='bottom'),
    'd3': ax1.text(1.5, 0.1, '', ha='center', va='bottom'),
    'd4': ax1.text(1.5, 0.9, '', ha='center', va='bottom'),
    'w1': ax1.text(0.5, 0.3, '', ha='right', va='bottom'),
    'w2': ax1.text(0.5, 0.7, '', ha='right', va='top'),
    'w3': ax1.text(1.5, 0.3, '', ha='left', va='bottom'),
    'w4': ax1.text(1.5, 0.7, '', ha='left', va='top'),
    'b1': ax1.text(1, 0.7, '', ha='center', va='center'),
}

# Loss plot
loss_line, = ax2.plot([], [], 'b-')
loss_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.grid(True)

def update_plot():
    neuron.set_offsets(network.neuron.position)
    
    for i, line in enumerate(lines):
        neuron_pos = network.neuron.position
        start, end = line.get_xydata()
        if i < 2:
            end = [neuron_pos[0], neuron_pos[1]]
        else:
            start = [neuron_pos[0], neuron_pos[1]]
        line.set_data([start[0], end[0]], [start[1], end[1]])
    
    x, y_true = network.dataset[network.current_iteration % 4]
    y_pred, _, inp_distances, out_distances = network.neuron.forward(x)
    
    texts['a1'].set_text(f'a1={x[0]}')
    texts['a2'].set_text(f'a2={x[1]}')
    texts['a3'].set_text(f'a3={y_pred[0]:.2f}')
    texts['a4'].set_text(f'a4={y_pred[1]:.2f}')
    
    for i, d in enumerate(np.append(inp_distances, out_distances)):
        texts[f'd{i+1}'].set_text(f'd{i+1}={d:.4f}')
    
    for i, w in enumerate(network.neuron.weights.flatten()):
        texts[f'w{i+1}'].set_text(f'w{i+1}={w:.4f}')
    
    texts['b1'].set_text(f'b1={network.neuron.bias[0]:.2f}')
    
    loss_line.set_data(range(len(network.loss_history)), network.loss_history)
    ax2.set_xlim(0, max(100, len(network.loss_history)))
    loss_text.set_text(f'Current Loss: {network.loss_history[-1]:.2f}')
    
    plt.draw()

def on_click(event):
    network.train_step()
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
        plt.pause(0.05)
        
def on_pause(event):
    global pause
    pause = True

play_button.on_clicked(on_play)
pause_button.on_clicked(on_pause)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class SpatialNeuron:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
        self.position = np.zeros(2)  # 2D position for visualization
        
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        distances = np.array([
            np.linalg.norm(self.position - np.array([0, 0])),  # d1
            np.linalg.norm(self.position - np.array([0, 1])),  # d2
            np.linalg.norm(self.position - np.array([1, 0])),  # d3
            np.linalg.norm(self.position - np.array([1, 1]))   # d4
        ])
        a = self.sigmoid(z) * np.prod(1 / (1 + distances))
        return a, z, distances
    
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
        self.accuracy_history = []
        
    def train_step(self):
        total_loss = 0
        total_correct = 0
        gradients = {'weights': np.zeros_like(self.neuron.weights),
                     'bias': np.zeros_like(self.neuron.bias),
                     'position': np.zeros_like(self.neuron.position)}
        
        for x, y_true in self.dataset:
            y_pred, z, distances = self.neuron.forward(x)
            
            # Compute loss and accuracy
            loss = np.mean((y_pred - y_true) ** 2)
            total_loss += loss
            total_correct += np.all(np.round(y_pred) == y_true)
            
            # Compute gradients
            d_loss = y_pred - y_true
            d_sigmoid = self.neuron.sigmoid_derivative(y_pred)
            d_weights = np.outer(x, d_loss * d_sigmoid)
            d_bias = d_loss * d_sigmoid
            
            gradients['weights'] += d_weights
            gradients['bias'] += d_bias
            
            # Gradient for position
            d_position = np.zeros(2)
            for i, pos in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                d_position += -np.sum(d_loss * d_sigmoid * y_pred) * (np.array(pos) - self.neuron.position) / (1 + distances[i])**3
            
            gradients['position'] += d_position
        
        # Update weights, bias, and position
        self.neuron.weights -= self.learning_rate * gradients['weights'] / len(self.dataset)
        self.neuron.bias -= self.learning_rate * gradients['bias'] / len(self.dataset)
        self.neuron.position -= self.position_learning_rate * gradients['position'] / len(self.dataset)
        
        accuracy = total_correct / len(self.dataset)
        self.accuracy_history.append(accuracy)
        self.current_iteration += 1
        
        print(f"Iteration {self.current_iteration}: Loss {total_loss/len(self.dataset):.4f}, Accuracy {accuracy:.2f}")
        return accuracy

# Visualization setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(bottom=0.2)
network = SpatialNetwork()

# Main plot
input_nodes = ax1.scatter([0, 0, 1, 1], [0, 1, 0, 1], c='orange', s=500, zorder=2)
output_nodes = ax1.scatter([2, 2], [0.33, 0.67], c='orange', s=500, zorder=2)
neuron = ax1.scatter([], [], c='black', s=500, zorder=3)
lines = [ax1.plot([], [], 'gray', linewidth=2, zorder=1)[0] for _ in range(4)]

ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 1.5)
ax1.axis('off')

# Text annotations
texts = {
    'a1': ax1.text(-0.1, 0, '', ha='right', va='center'),
    'a2': ax1.text(-0.1, 1, '', ha='right', va='center'),
    'a3': ax1.text(2.1, 0.33, '', ha='left', va='center'),
    'a4': ax1.text(2.1, 0.67, '', ha='left', va='center'),
    'd1': ax1.text(0.5, 0.2, '', ha='center', va='bottom'),
    'd2': ax1.text(0.5, 0.8, '', ha='center', va='bottom'),
    'd3': ax1.text(1.5, 0.2, '', ha='center', va='bottom'),
    'd4': ax1.text(1.5, 0.8, '', ha='center', va='bottom'),
    'w1': ax1.text(0.4, 0.1, '', ha='right', va='bottom', rotation=15),
    'w2': ax1.text(0.4, 0.9, '', ha='right', va='top', rotation=-15),
    'w3': ax1.text(1.6, 0.4, '', ha='left', va='bottom', rotation=-15),
    'w4': ax1.text(1.6, 0.6, '', ha='left', va='top', rotation=15),
    'b1': ax1.text(1, 0.5, '', ha='center', va='center'),
}

# Accuracy plot
accuracy_line, = ax2.plot([], [], 'b-')
accuracy_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

def update_plot():
    neuron.set_offsets(network.neuron.position)
    
    for i, line in enumerate(lines):
        start = [0, 0] if i < 2 else [1, 1]
        end = network.neuron.position
        line.set_data([start[0], end[0], 2], [start[1], end[1], 0.33 if i % 2 == 0 else 0.67])
    
    x, y_true = network.dataset[network.current_iteration % 4]
    y_pred, _, distances = network.neuron.forward(x)
    
    texts['a1'].set_text(f'a1={x[0]}')
    texts['a2'].set_text(f'a2={x[1]}')
    texts['a3'].set_text(f'a3={y_pred[0]:.2f}')
    texts['a4'].set_text(f'a4={y_pred[1]:.2f}')
    
    for i, d in enumerate(distances):
        texts[f'd{i+1}'].set_text(f'd{i+1}={d:.2f}')
    
    for i, w in enumerate(network.neuron.weights.flatten()):
        texts[f'w{i+1}'].set_text(f'w{i+1}={w:.2f}')
    
    texts['b1'].set_text(f'b1={network.neuron.bias[0]:.2f}\nb2={network.neuron.bias[1]:.2f}')
    
    accuracy_line.set_data(range(len(network.accuracy_history)), network.accuracy_history)
    ax2.set_xlim(0, max(100, len(network.accuracy_history)))
    accuracy_text.set_text(f'Current Accuracy: {network.accuracy_history[-1]:.2f}')
    
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
        plt.pause(0.1)
        
def on_pause(event):
    global pause
    pause = True

play_button.on_clicked(on_play)
pause_button.on_clicked(on_pause)

plt.show()
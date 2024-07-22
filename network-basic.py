import numpy as np
import matplotlib.pyplot as plt

class SpatialNeuron:
    def __init__(self, position):
        self.position = np.array(position)
        self.connections = []

class SpatialNetwork:
    def __init__(self, num_neurons, input_dim=2):
        self.neurons = [SpatialNeuron(np.random.rand(input_dim)) for _ in range(num_neurons)]
        self.connect_neurons(3)  # Connect each neuron to its 3 nearest neighbors

    def connect_neurons(self, k):
        for i, neuron in enumerate(self.neurons):
            distances = [np.linalg.norm(neuron.position - other.position) for other in self.neurons]
            nearest = np.argsort(distances)[1:k+1]  # Exclude self
            neuron.connections = [(j, 1/distances[j]) for j in nearest]

    def activate(self, input_point):
        input_point = np.array(input_point)
        for neuron in self.neurons:
            distance = np.linalg.norm(neuron.position - input_point)
            activation = 1 / (1 + distance)  # Simple activation function
            
            # Move neuron towards input point
            neuron.position += 0.1 * activation * (input_point - neuron.position)
            
            # Apply pull from connected neurons
            for connected, strength in neuron.connections:
                pull = self.neurons[connected].position - neuron.position
                neuron.position += 0.01 * strength * pull

        # Return sum of activations as output
        return sum(1 / (1 + np.linalg.norm(n.position - input_point)) for n in self.neurons)

    def visualize(self, ax):
        positions = np.array([n.position for n in self.neurons])
        ax.scatter(positions[:, 0], positions[:, 1], c='blue')
        for neuron in self.neurons:
            for connected, _ in neuron.connections:
                con_pos = self.neurons[connected].position
                ax.plot([neuron.position[0], con_pos[0]], 
                        [neuron.position[1], con_pos[1]], 'k-', alpha=0.2)

# Example usage
network = SpatialNetwork(20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

network.visualize(ax1)
ax1.set_title("Before Training")

# Training
for _ in range(100):
    input_point = np.random.rand(2)
    network.activate(input_point)

network.visualize(ax2)
ax2.set_title("After Training")

plt.show()

# Test the network
test_input = [0.5, 0.5]
output = network.activate(test_input)
print(f"Network output for input {test_input}: {output}")
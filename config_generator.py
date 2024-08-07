import json
import numpy as np

# Generate node_cofig with n layers and m nodes per layer and specific pattern
def generate_config(n_inputs, n_outputs, n_layers, n_nodes_per_layer, scale = 1, activation_function = "sigmoid", pattern="alternate"):
    # alternate pattern makes a classic NN structure with alternate node with +0.2x
    node_config = {}
    # Input nodes
    if n_inputs == 1:
        node_config[(0, scale / 2)] = activation_function
    else:
        for i in range(n_inputs):
            node_config[(0, i * scale / (n_inputs - 1))] = activation_function
        
    # Hidden layers
    for layer in range(n_layers):
        for node in range(n_nodes_per_layer):
            x : float = 0
            y : float = 0
            if pattern == "alternate":
                x : float = layer + 1
                y : float = node * scale / (n_nodes_per_layer - 1)
                if node % 2 == 1:
                    x += 0.2
            node_config[(x, y)] = activation_function
            
    # Output nodes
    if n_outputs == 1:
        node_config[(n_layers + 1, scale / 2)] = "linear"
    else:
        for i in range(n_outputs):
            node_config[(n_layers + 1, i * scale / (n_outputs - 1))] = "linear"
    
    return node_config
    
# Example usage
node_config = generate_config(1, 1, 2, 50, 10, "tanh", "alternate")
parsed_node_config = {}
for key, value in node_config.items():
    parsed_node_config[str(key)] = value.tolist() if isinstance(value, np.ndarray) else value
print(json.dumps(parsed_node_config, indent=2))
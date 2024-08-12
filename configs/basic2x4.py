node_config = {
    (0, 0.5): "relu",      # Input node

    # Hidden layer 1 (2 nodes)
    (1, 0.1): "tanh",
    (1, 0.9): "tanh",

    # Hidden layer 2 (2 nodes)
    (2, 0.1): "tanh",
    (2, 0.9): "tanh",
    
    # Hidden layer 3 (2 nodes)
    (3, 0.1): "tanh",
    (3, 0.9): "tanh",
    
    # Hidden layer 4 (2 nodes)
    (4, 0.1): "tanh",
    (4, 0.9): "tanh",
    
    (5, 0.5): "relu"       # Output node
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
config_name = "basic2x4"
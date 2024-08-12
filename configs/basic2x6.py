node_config = {
    (0, 0.5): "tanh",      # Input node

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
    
    # Hidden layer 5 (2 nodes)
    (5, 0.1): "tanh",
    (5, 0.9): "tanh",
    
    # Hidden layer 6 (2 nodes)
    (6, 0.1): "tanh",
    (6, 0.9): "tanh",

    (7, 0.5): "linear" # Output node
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
config_name = "basic2x6"
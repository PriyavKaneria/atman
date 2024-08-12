node_config = {
    (0, 0.5): "relu",      # Input node

    # Hidden layer 1 (2 nodes)
    (1, 0.1): "relu",
    (1, 0.9): "relu",

    # Hidden layer 2 (2 nodes)
    (2, 0.1): "tanh",
    (2, 0.9): "tanh",
    
    (3, 0.5): "linear"       # Output node
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
config_name = "basic2x2"
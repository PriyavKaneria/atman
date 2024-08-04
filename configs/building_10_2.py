node_config = {
    (0, 0.5): "tanh",      # Input node

    # Hidden layer 1 (5 nodes)
    (1, 0.1): "tanh",
    (1.1, 0.2): "tanh",
    (1.2, 0.3): "tanh",
    (1.3, 0.4): "tanh",
    (1.4, 0.5): "tanh",
    (1.3, 0.6): "tanh",
    (1.2, 0.7): "tanh",
    (1.1, 0.8): "tanh",
    (1, 0.9): "tanh",

    # Hidden layer 2 (5 nodes)
    (2, 0.1): "tanh",
    (2.1, 0.2): "tanh",
    (2.2, 0.3): "tanh",
    (2.3, 0.4): "tanh",
    (2.4, 0.5): "tanh",
    (2.3, 0.6): "tanh",
    (2.2, 0.7): "tanh",
    (2.1, 0.8): "tanh",
    (2, 0.9): "tanh",

    (3, 0.5): "linear"       # Output node
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
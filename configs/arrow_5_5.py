node_config = {
    (0, 0.5): "tanh",      # Input node

    # Hidden layer 1 (5 nodes)
    (1, 0.1): "tanh",
    (1.1, 0.3): "tanh",
    (1.2, 0.5): "tanh",
    (1.1, 0.7): "tanh",
    (1, 0.9): "tanh",

    # Hidden layer 2 (5 nodes)
    (2, 0.1): "tanh",
    (2.1, 0.3): "tanh",
    (2.2, 0.5): "tanh",
    (2.1, 0.7): "tanh",
    (2, 0.9): "tanh",

    # Hidden layer 3 (5 nodes)
    (3, 0.1): "tanh",
    (3.1, 0.3): "tanh",
    (3.2, 0.5): "tanh",
    (3.1, 0.7): "tanh",
    (3, 0.9): "tanh",

    # Hidden layer 4 (5 nodes)
    (4, 0.1): "tanh",
    (4.1, 0.3): "tanh",
    (4.2, 0.5): "tanh",
    (4.1, 0.7): "tanh",
    (4, 0.9): "tanh",

    # Hidden layer 5 (5 nodes)
    (5, 0.1): "tanh",
    (5.1, 0.3): "tanh",
    (5.2, 0.5): "tanh",
    (5.1, 0.7): "tanh",
    (5, 0.9): "tanh",

    (6, 0.5): "linear"       # Output node
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
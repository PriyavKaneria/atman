node_config = {
    (0, 0.5): "relu",      # Input node

    # Hidden layer 1 (5 nodes)
    (1, 0.1): "relu",
    (1.1, 0.3): "relu",
    (1.2, 0.5): "relu",
    (1.1, 0.7): "relu",
    (1, 0.9): "relu",

    # Hidden layer 2 (5 nodes)
    (2, 0.1): "relu",
    (2.1, 0.3): "relu",
    (2.2, 0.5): "relu",
    (2.1, 0.7): "relu",
    (2, 0.9): "relu",

    # Hidden layer 3 (5 nodes)
    (3, 0.1): "relu",
    (3.1, 0.3): "relu",
    (3.2, 0.5): "relu",
    (3.1, 0.7): "relu",
    (3, 0.9): "relu",

    # Hidden layer 4 (5 nodes)
    (4, 0.1): "relu",
    (4.1, 0.3): "relu",
    (4.2, 0.5): "relu",
    (4.1, 0.7): "relu",
    (4, 0.9): "relu",

    # Hidden layer 5 (5 nodes)
    (5, 0.1): "relu",
    (5.1, 0.3): "relu",
    (5.2, 0.5): "relu",
    (5.1, 0.7): "relu",
    (5, 0.9): "relu",

    (6, 0.5): "linear"       # Output node
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
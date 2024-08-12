node_config = {
  (0, 0.5): "relu",
  (1, 0.0): "tanh",
  (1.2, 0.5): "tanh",
  (1, 1.0): "tanh",
  (2, 0.0): "tanh",
  (2.2, 0.5): "tanh",
  (2, 1.0): "tanh",
  (3, 0.0): "tanh",
  (3.2, 0.5): "tanh",
  (3, 1.0): "tanh",
  (4, 0.0): "tanh",
  (4.2, 0.5): "tanh",
  (4, 1.0): "tanh",
  (5, 0.0): "tanh",
  (5.2, 0.5): "relu",
  (5, 1.0): "tanh",
  (6, 0.5): "linear"
}

node_positions = list(node_config.keys())
input_indices = [0]
output_indices = [len(node_positions) - 1]
config_name = "basic3x5"
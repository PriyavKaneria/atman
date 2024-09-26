# आत्मन् • ātmán (Adaptive Topological Muscle-Actuated Network)

**आत्मन् • ātmán** is an experimental neural network model designed with a unique focus on spatial awareness and topological adaptations that tries to simulate how the human brain stores information. It leverages both adaptive muscle-like nodes and spatial attention mechanisms, aiming to improve network efficiency and precision, particularly in regression tasks like sine wave prediction. The project explores different configurations, genetic training methods, and batch processing for optimized neural behavior.

## Features

- **Spatial Attention**: Implements multiple spatial attention strategies, including child attention models, to handle diverse tasks more effectively.
- **Topological Adaptation**: Utilizes muscle-actuated adaptive topologies, allowing for flexible changes to the network structure during training.
- **Genetic Training**: Customizable genetic algorithms with mutation strategies are used to evolve better network configurations.
- **PyQt6 Integration**: Migrated from Matplotlib to PyQt6 for faster plotting and more responsive multi-threaded performance in training visualizations.
- **Checkpoints and Pruning**: Automated saving, loading, and pruning of checkpoints for training seeding and optimization.

Below is a demo of multiple children spatial training for sine wave regression problem


https://github.com/user-attachments/assets/b60ba687-fbbc-474c-8ca4-78a7dff31102



The progress and main stages of development can be tracked through the [commit log](https://github.com/yourusername/atmán/commits/main).
## Logs

All logs, including screenshots and video recordings, are stored in the `/logs` folder. They capture key moments of the network’s evolution and insights from various training sessions.

| Filename                                                | Description |
| --------------------------------------------------------| ----------- |
| `basic_neuron.png`                                       | Initial network structure overview. |
| `first_successful_spatial_training_static_dataset.png`   | First successful spatial training on a static dataset. |
| `min_distance_implementation.png`                        | Implementation of minimum neuron distance functionality. |
| `sine_regression_with_tanh_fail.png`                     | Attempted sine regression with tanh activation failure. |
| `matplotlib_to_pyqt6_migration_and_iteration_speed.mp4`  | PyQt6 migration leading to faster iteration speed and plotting. |
| `spatial_attention_speedup_with_single_child.mp4`        | Speedup achieved through single-child attention model. |
| `genetic_training_with_mutation_sine_wave.mp4`           | Genetic training involving mutation on sine wave regression task. |
| `6_children_spatial_training.mp4`                        | Video showcasing spatial training with 6 child attention models. |
| [Complete log listing](./logs/)

## How It Works

1. **Topological Adaptation**: The network architecture adapts by reconfiguring connections and updating node positions based on local minima and distance constraints.
2. **Spatial Attention Mechanism**: Nodes focus attention based on spatial proximity, distance gradients, and evolutionary strategies.
3. **Genetic Training**: Trains using genetic algorithms with customizable mutation rates and fitness pruning.
4. **Performance Visualization**: PyQt6 integration enhances real-time performance and plotting, providing better insights into how training evolves.

## Installation

Ensure you have Python 3.8+ and the required dependencies.

```bash
git clone https://github.com/yourusername/atmán.git
cd atmán
pip install -r requirements.txt
```

## Usage

### Train the network

You can start training with various configuration files from the `configs/` directory.

```bash
python train.py --config configs/basic_config.yaml
```

### Viewing Logs

All training results, visual logs, and recordings are saved in the `/logs` folder. Use the file viewer or any media player to access videos.

## Contributions

Feel free to fork this repository and submit pull requests. Contributions and discussions about new ideas, optimization strategies, or issues are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

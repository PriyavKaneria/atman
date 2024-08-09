# Run network_basic in batches of n for k iterations and then save the checkpoint for the child with maximum accuracy
# On press of 'N' key start another batch with the seed as the saved checkpoint
import time
import numpy as np
import pickle
from network_basic import main as run_network_basic  # Assuming `main` is your network training function

N : int = 10
K : int = 500
mutation_rate = 0.03

# Use seed checkpoint if provided
seed_weights = True
seed_checkpoint_path = "./checkpoints/checkpoint_best_0_0.575.atmn"
seed_weights_bias = None
if seed_weights:
    with open(seed_checkpoint_path, 'rb') as f:
        seed_weights_bias = pickle.load(f)

results = []
generation = 1
max_accuracy = 0.575
def run_generation(no_children, max_iteration, seed_weights_bias, mutation_rate):
    global max_accuracy, generation
    instance = run_network_basic(no_children = no_children, seed_weights_bias = seed_weights_bias, max_iterations = max_iteration, seed_child_mutation=mutation_rate, auto_start = True)
    
    for child in instance.child_networks:
        results.append((child.max_accuracy, child.get_checkpoint()))

    # Find the network with the maximum accuracy
    best_accuracy, ckpt = max(results, key=lambda x: x[0])  # Sort by accuracy

    if best_accuracy > max_accuracy:
        # Save the checkpoint for the network with maximum accuracy
        checkpoint_path = f"./checkpoints/checkpoint_best_{generation}_{best_accuracy}.atmn"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(ckpt, f)
    
        print(f"Best network for generation {generation} with accuracy {best_accuracy:.2f} saved to {checkpoint_path}")
        max_accuracy = best_accuracy
        return checkpoint_path
    else:
        print(f"No improvement in accuracy for generation {generation}")
        return None

# def start_new_batch_on_key_press():
#     while True:
#         key = input("Press 'N' to start a new batch: ")
#         if key.lower() == 'n':
#             checkpoint_path = run_generation(N, K, seed_weights_bias)
#             # Load the best checkpoint for the next batch
#             global seed_weights_bias
#             with open(checkpoint_path, 'rb') as f:
#                 seed_weights_bias = pickle.load(f)

if __name__ == "__main__":
    # Initial batch run
    if seed_weights_bias is not None:
        print(f"Starting batch run with seed checkpoint: {seed_checkpoint_path}")
        checkpoint_path = seed_checkpoint_path
    while max_accuracy < 0.9 and generation < 15:
        print(f"Starting generation {generation}")
        checkpoint_path = run_generation(N, K, seed_weights_bias, mutation_rate)
        generation += 1
        if checkpoint_path is None:
            checkpoint_path = seed_checkpoint_path
        else:
            seed_checkpoint_path = checkpoint_path

    # Start listening for 'N' key press to start new batches
    # start_new_batch_on_key_press()

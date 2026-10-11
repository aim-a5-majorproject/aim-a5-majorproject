import numpy as np

def fedavg(client_weights, client_sizes):
    """Sample-weighted Federated Averaging used by the supplied notebook."""
    if not client_weights:
        raise ValueError("client_weights cannot be empty")
    if len(client_weights) != len(client_sizes):
        raise ValueError("One client size is required for every client weight set")
    if any(n <= 0 for n in client_sizes):
        raise ValueError("All client sizes must be positive")

    total = float(sum(client_sizes))
    averaged = []
    for layer_group in zip(*client_weights):
        layer = np.zeros_like(layer_group[0], dtype=np.float32)
        for weights, n_samples in zip(layer_group, client_sizes):
            layer += weights.astype(np.float32) * (n_samples / total)
        averaged.append(layer)
    return averaged

import numpy as np


def calculate_ber(pred_indices, true_indices):
    pred_bits = np.unpackbits(np.array(pred_indices, dtype=np.uint8).reshape(-1, 1), axis=1)[:, -2:]
    true_bits = np.unpackbits(np.array(true_indices, dtype=np.uint8).reshape(-1, 1), axis=1)[:, -2:]

    error_bits = np.sum(pred_bits != true_bits)
    total_bits = len(true_indices) * 2

    return error_bits / total_bits
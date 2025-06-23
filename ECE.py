import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
        if np.sum(in_bin) == 0:
            continue
        acc = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
        conf = np.mean(y_prob[in_bin])
        ece += np.abs(acc - conf) * np.sum(in_bin)
    return ece / len(y_true)


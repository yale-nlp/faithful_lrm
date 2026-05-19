"""Helpers to compute various dataset-level metrics."""

import numpy as np

def compute_mfg(f_values):
    f_values = np.array(f_values, dtype=float)
    f_values = f_values[(f_values != -1) & np.isfinite(f_values)]
    return float(np.mean(f_values)) if len(f_values) > 0 else 0.0

def compute_cmfg(f_values, conf_scores, num_bins=10):
    f_values = np.array(f_values, dtype=float)
    conf_scores = np.array(conf_scores, dtype=float)
    mask = np.isfinite(f_values) & np.isfinite(conf_scores) & (f_values != -1) & (conf_scores != -1)
    f_values = f_values[mask]
    conf_scores = conf_scores[mask]
    if len(f_values) == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(conf_scores, bin_edges) - 1
    bin_faithfulness = []
    for b in range(num_bins):
        bin_f = f_values[bin_indices == b]
        bin_f = bin_f[bin_f != -1]
        bin_faithfulness.append(float(np.mean(bin_f)) if len(bin_f) > 0 else 0.5)
    return float(np.mean(bin_faithfulness))

def compute_cmfg_star(f_values, conf_scores, num_bins=10):
    """
    cMFG* metric: equal-MASS bins, weighted by bin width on the confidence axis.
    """
    f_values = np.array(f_values, dtype=float)
    conf_scores = np.array(conf_scores, dtype=float)

    mask = np.isfinite(f_values) & np.isfinite(conf_scores) & (f_values != -1) & (conf_scores != -1)
    f_values = f_values[mask]
    conf_scores = conf_scores[mask]

    n = len(f_values)
    if n == 0:
        return 0.0, {}

    num_bins = min(num_bins, n)

    sort_idx = np.argsort(conf_scores)
    f_sorted = f_values[sort_idx]
    c_sorted = conf_scores[sort_idx]

    bin_size = n // num_bins
    remainder = n % num_bins

    bins = []
    start = 0
    for b in range(num_bins):
        end = start + bin_size + (1 if b < remainder else 0)
        bins.append((start, end))
        start = end

    bin_info = []
    for b_idx, (s, e) in enumerate(bins):
        bin_confs = c_sorted[s:e]
        bin_faiths = f_sorted[s:e]
        f_hat = float(np.mean(bin_faiths))

        if b_idx == 0:
            l_j = float(bin_confs[0])
        else:
            prev_max = c_sorted[bins[b_idx - 1][1] - 1]
            curr_min = bin_confs[0]
            l_j = float((prev_max + curr_min) / 2)

        if b_idx == num_bins - 1:
            u_j = float(bin_confs[-1])
        else:
            curr_max = bin_confs[-1]
            next_min = c_sorted[bins[b_idx + 1][0]]
            u_j = float((curr_max + next_min) / 2)

        w_j = u_j - l_j
        bin_info.append({'f_hat': f_hat, 'l': l_j, 'u': u_j, 'w': w_j, 'n': e - s})

    total_w = sum(b['w'] for b in bin_info)
    if total_w == 0:
        cmfg_star = float(np.mean([b['f_hat'] for b in bin_info]))
    else:
        cmfg_star = sum(b['w'] * b['f_hat'] for b in bin_info) / total_w

    return cmfg_star, {'bins': bin_info, 'total_width': total_w}
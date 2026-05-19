"""Helper functions for DeepConf confidence estimation method."""

import torch
import numpy as np
from typing import List

def compute_deepconf_token_confs(scores, gen_ids: torch.Tensor, topk: int = 5) -> List[float]:
    confs = []
    for t, logits in enumerate(scores):
        if logits.dim() == 2:
            logits = logits[0]
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=0.0)
        logp = torch.log_softmax(logits, dim=-1)
        top_logprobs, _ = torch.topk(logp, min(topk, logp.shape[-1]))
        finite_logprobs = top_logprobs[torch.isfinite(top_logprobs)]
        if finite_logprobs.numel() == 0:
            continue
        val = -finite_logprobs.mean().item()
        if not np.isfinite(val):
            continue
        confs.append(round(val, 3))
    return confs

def compute_deepconf_for_token_range(scores, start_idx, end_idx, topk=5) -> List[float]:
    confs = []
    for t in range(start_idx, min(end_idx, len(scores))):
        logits = scores[t]
        if logits.dim() == 2:
            logits = logits[0]
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=0.0)
        logp = torch.log_softmax(logits, dim=-1)
        top_logprobs, _ = torch.topk(logp, min(topk, logp.shape[-1]))
        finite_logprobs = top_logprobs[torch.isfinite(top_logprobs)]
        if finite_logprobs.numel() == 0:
            continue
        val = -finite_logprobs.mean().item()
        if not np.isfinite(val):
            continue
        confs.append(round(val, 3))
    return confs

def calculate_mean_confidence(confs):
    confs = [c for c in confs if np.isfinite(c)]
    return float(np.mean(confs)) if confs else 0.0

def calculate_tail_confidence(confs, tail_tokens=2048):
    confs = [c for c in confs if np.isfinite(c)]
    if not confs:
        return 0.0
    return float(np.mean(confs[-tail_tokens:]))

def calculate_bottom_window_confidence(confs, window_size=2048, bottom_percent=0.1):
    confs = [c for c in confs if np.isfinite(c)]
    if not confs:
        return 0.0
    if len(confs) < window_size:
        return float(np.mean(confs))
    window_means = []
    s = sum(confs[:window_size])
    window_means.append(s / window_size)
    for i in range(1, len(confs) - window_size + 1):
        s = s - confs[i - 1] + confs[i + window_size - 1]
        window_means.append(s / window_size)
    if not window_means:
        return 0.0
    if bottom_percent == -1:
        return float(min(window_means))
    nb = max(1, int(len(window_means) * bottom_percent))
    if nb == 1:
        return float(min(window_means))
    return float(np.mean(np.partition(window_means, nb - 1)[:nb]))

def extract_deepconf_from_vllm_logprobs(token_logprobs_list, topk=5):
    confs = []
    for token_dict in token_logprobs_list:
        if token_dict is None:
            continue
        logprobs = sorted(
            [value['logprob'] for value in token_dict.values()],
            reverse=True,
        )[:topk]
        val = -np.mean(logprobs)
        if not np.isfinite(val):
            continue
        confs.append(round(val, 3))
    return confs

def normalize_deepconf(val):
    if val is None or not np.isfinite(val):
        return None
    return min(max(val / 8.0, 0.0), 1.0)

def deepconf_for_char_range(all_confs, gen_text, start_char, end_char, tokenizer):
    enc = tokenizer(gen_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc['offset_mapping']
    return deepconf_for_char_range_with_offsets(all_confs, offsets, start_char, end_char)

def deepconf_for_char_range_with_offsets(all_confs, offsets, start_char, end_char):
    idxs = []
    for i, (start, end) in enumerate(offsets):
        if end <= start_char:
            continue
        if start >= end_char:
            break
        idxs.append(i)
    return [all_confs[i] for i in idxs if i < len(all_confs)]
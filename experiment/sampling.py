"""Utility functions for sampling consistency."""

from collections import defaultdict
import numpy as np

def build_prefix_prompts(prompts, gen_texts, all_spans):
    """Build one prefix prompt per (example, step) for vLLM sampling.

    For step 0: use the base prompt.
    For later steps: use the base prompt plus the original trace prefix before target.
    """
    prefix_prompts = []
    prefix_index = []

    for i, spans in enumerate(all_spans):
        base_prompt = prompts[i]
        for si in range(len(spans)):
            if si == 0:
                prefix = base_prompt
            else:
                prefix_end_char = spans[si][0]
                prefix = base_prompt + gen_texts[i][:prefix_end_char]
            prefix_prompts.append(prefix)
            prefix_index.append((i, si))

    return prefix_prompts, prefix_index

def build_judge_tasks(sample_outputs, prefix_index, all_spans, K):
    tasks = []
    for j, (ei, si) in enumerate(prefix_index):
        orig_step = all_spans[ei][si][2]
        item = sample_outputs[j]
        # Accept either vLLM RequestOutput (has .outputs[k].text) or list[str]; needed for diff tasks
        if isinstance(item, list):
            texts = item
        else:
            texts = [completion.text for completion in item.outputs]
        for k in range(K):
            sample_text = texts[k].strip()
            first_chunk = sample_text.split('\n\n')[0].strip()
            if not first_chunk:
                first_chunk = sample_text[:300]
            tasks.append({
                'context': first_chunk,
                'assertion': orig_step,
                'ei': ei,
                'si': si,
                'k': k,
            })
    return tasks

def compute_step_confidences(judge_tasks, judge_scores, all_spans, N):
    step_scores = defaultdict(list)
    for task, score in zip(judge_tasks, judge_scores):
        step_scores[(task['ei'], task['si'])].append(score)

    results = []
    for i in range(N):
        spans = all_spans[i]
        step_confs = []
        for si in range(len(spans)):
            scores = step_scores.get((i, si), [])
            contradiction_prob = float(np.mean(scores)) if scores else None
            step_confs.append(
                1 - contradiction_prob if contradiction_prob is not None else None
            )

        valid_confs = [conf for conf in step_confs if conf is not None]
        overall = float(np.mean(valid_confs)) if valid_confs else 0.5
        results.append({
            'step_sampling_conf': step_confs,
            'sampling_conf': overall,
        })

    return results
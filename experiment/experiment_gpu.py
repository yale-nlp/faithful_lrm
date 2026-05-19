"""
Full experiment pipeline: generation + all confidence metrics + faithfulness.

Computes DeepConf, RCC, and Sampling-based confidence at the step level,
along with step-level decisiveness.  Outputs metrics, plots, and Excel files
into a folder named after the run.

Usage:
    python experiment_gpu.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --dataset aime \
        --run_name test_run_01 \
        --n 100
"""

import argparse
import asyncio
import os
import pickle
import re
import time
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# ── Local imports ───────────────────────────────────────────────────────────

# Import using relative imports from sibling modules
from dataset_utils import (
    load_hle, build_prompt_hle, score_hle,
    load_supergpqa, build_prompt_supergpqa, score_supergpqa,
    load_aime, build_prompt_aime, score_aime,
    load_legalbench, build_prompt_legalbench, score_legalbench,
    load_musr, build_prompt_musr, score_musr,
)
from utils import format_chat_if_available, extract_final_answer_text, get_steps_from_think_block
from deepconf import (
    calculate_mean_confidence,
    calculate_tail_confidence,
    calculate_bottom_window_confidence,
    extract_deepconf_from_vllm_logprobs,
    normalize_deepconf,
    deepconf_for_char_range_with_offsets,
)
from rcc import (
    build_charspan_to_token_idxs,
    extract_chosen_token_probs,
    rcc_scores_for_spans_from_hidden,
)
from metrics import compute_mfg, compute_cmfg, compute_cmfg_star
from sampling import (
    build_prefix_prompts, build_judge_tasks, compute_step_confidences,
)
from prompts import HEDGE_PROMPTS, SYSTEM_PROMPTS, apply_hedge, get_system_prompt

# ── Dataset loaders ─────────────────────────────────────────────────────────

DATASET_LOADERS = {
    'hle': (load_hle, build_prompt_hle, score_hle),
    'supergpqa': (load_supergpqa, build_prompt_supergpqa, score_supergpqa),
    'aime': (load_aime, build_prompt_aime, score_aime),
    'legalbench': (load_legalbench, build_prompt_legalbench, score_legalbench),
    'musr': (load_musr, build_prompt_musr, score_musr),
}

# ── Sampling-consistency judge prompt ───────────────────────────────────────

UNCERTAINTY_PROMPT = """Context: {context}
Assertion: {assertion}
Is the assertion consistent with the context above?
Answer Yes or No:"""

UNCERTAINTY_PATTERN = r"(?i)(?:Answer\s+yes\s+or\s+no:\s*)?\b(yes|no|N/A)\b"

UNCERTAINTY_MAPPING = {
    "yes": 0.0,     # factual / no contradiction
    "no": 1.0,      # nonfactual / contradiction
    "n/a": 0.5,     # somewhat inaccurate / unclear
}

# ── Helpers ─────────────────────────────────────────────────────────────────
def _ckpt_path(out_dir, name):
    return os.path.join(out_dir, f'_checkpoint_{name}.pkl')

def save_checkpoint(out_dir, name, data):
    path = _ckpt_path(out_dir, name)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f'  [checkpoint] saved {name} at {path}')

def load_checkpoint(out_dir, name):
    path = _ckpt_path(out_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f'  [checkpoint] loaded {name} from {path}')
    return data

def cleanup_checkpoints(out_dir):
    removed = 0
    for fn in os.listdir(out_dir):
        if fn.startswith('_checkpoint_') and fn.endswith('.pkl'):
            try:
                os.remove(os.path.join(out_dir, fn))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f'  [checkpoint] cleaned {removed} checkpoint file(s)')

# ── Plotting ────────────────────────────────────────────────────────────────
def save_plots(df, out_dir, dataset_name):
    """Generate and save scatter + heatmap plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = df[df['correct'].notna()].copy()
    mask = (valid['avg_decisiveness'] != -1) & (valid['faithfulness_rcc'] != -1)
    v = valid[mask].copy()

    if len(v) < 5:
        print('Not enough valid data for plots, skipping.')
        return

    # Scatter: metrics vs decisiveness
    conf_cols = [
        ('rcc_confidence', 'RCC Confidence'),
        ('deepconf_confidence', 'DeepConf Confidence'),
        ('sampling_conf', 'Sampling Confidence'),
        ('faithfulness_rcc', 'Faithfulness (RCC)'),
        ('faithfulness_deepconf', 'Faithfulness (DeepConf)'),
        ('faithfulness_sampling', 'Faithfulness (Sampling)'),
    ]
    fig, axes = plt.subplots(1, len(conf_cols), figsize=(5 * len(conf_cols), 4.5))
    for (col, label), ax in zip(conf_cols, axes):
        if col not in v.columns:
            continue
        vc = v[v[col].notna() & (v[col] != -1)]
        if len(vc) < 3:
            continue
        r, p = stats.spearmanr(vc['avg_decisiveness'], vc[col])
        ax.scatter(vc['avg_decisiveness'], vc[col], alpha=0.5, s=40)
        ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3e}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top', fontsize=10)
        ax.set_xlabel('Decisiveness')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'{dataset_name}: Metrics vs Decisiveness', fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_vs_decisiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap
    hcols = ['rcc_confidence', 'deepconf_confidence', 'sampling_conf',
             'avg_decisiveness', 'faithfulness_rcc', 'faithfulness_deepconf', 'faithfulness_sampling']
    hcols = [c for c in hcols if c in v.columns]
    hlbls = [c.replace('_', ' ').title()[:12] for c in hcols]
    cm = np.zeros((len(hcols), len(hcols)))
    for i in range(len(hcols)):
        for j in range(len(hcols)):
            vc = v[v[hcols[i]].notna() & v[hcols[j]].notna() &
                    (v[hcols[i]] != -1) & (v[hcols[j]] != -1)]
            cm[i][j] = 1.0 if i == j else (
                stats.spearmanr(vc[hcols[i]], vc[hcols[j]])[0] if len(vc) >= 3 else 0.0
            )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(hlbls)))
    ax.set_xticklabels(hlbls, rotation=45, ha='right')
    ax.set_yticks(range(len(hlbls)))
    ax.set_yticklabels(hlbls)
    for i in range(len(hcols)):
        for j in range(len(hcols)):
            ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im)
    plt.title(f'{dataset_name}: Spearman Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Plots saved to {out_dir}/')

# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def run_experiment(args):
    out_dir = args.run_name
    os.makedirs(out_dir, exist_ok=True)

    # ── Setup ───────────────────────────────────────────────────────────
    SEED = 42
    N = args.n
    DEEPCONF_TOPK = 5
    DATASET_MAX_TOKENS = {
        'aime': 20480,
        'supergpqa': 20480,
        'hle': 20480,
        'musr': 20480,
        'legalbench': 20480,
    }
    MAX_NEW_TOKENS = DATASET_MAX_TOKENS.get(args.dataset, 8192)
    K = args.k
    SAMPLE_MAX_TOKENS = 200
    MAX_SAMPLE_STEPS = 20
    RCC_MU, RCC_DELTA = 0.5, 0.4
    loader, prompt_builder, scorer = DATASET_LOADERS[args.dataset]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_df = loader(n=N, seed=SEED)
    N = min(N, len(data_df))
    print(f'Loaded {N} {args.dataset} examples')

    hedge_key = args.hedge
    sys_prompt = get_system_prompt(args.sys_prompt)
    hedge_text = HEDGE_PROMPTS.get(hedge_key, "")
    print(f'Hedge prompt: {hedge_key}' + (f' -> {hedge_text[:60]}...' if hedge_text else ''))
    if sys_prompt:
        print(f'System prompt ({args.sys_prompt}): {sys_prompt[:80]}...' if len(sys_prompt) > 80 else f'System prompt ({args.sys_prompt}): {sys_prompt}')
    prompts = []
    for i in range(N):
        prompt = prompt_builder(data_df.iloc[i]['input_args'])
        prompt = apply_hedge(prompt, hedge_key)
        prompts.append(format_chat_if_available(tokenizer, prompt, system_prompt=sys_prompt))

    # ════════════════════════════════════════════════════════════════════
    # Probe checkpoints up front — decide whether vLLM is needed
    # ════════════════════════════════════════════════════════════════════
    ckpt_pass1 = load_checkpoint(out_dir, 'pass1')
    ckpt_pass1b = load_checkpoint(out_dir, 'pass1b') if not args.skip_sampling else None
    need_vllm = (ckpt_pass1 is None) or (not args.skip_sampling and ckpt_pass1b is None)

    if args.only_sampling and ckpt_pass1 is None:
        raise FileNotFoundError(
            f'--only_sampling requires {out_dir}/_checkpoint_pass1.pkl to exist. ')

    ctx_len = MAX_NEW_TOKENS + 4096  # output + headroom for input prompt
    llm = None
    if need_vllm:
        # In --only_sampling mode we don't need to fit a long-context generation
        # (gen_texts come from pickle); we just need vLLM for short K=10 continuations.
        # Bump max_num_seqs to use available KV cache for higher sampling throughput.
        is_llama = "llama" in args.model.lower()
        if args.only_sampling:
            max_num_seqs = 256
            max_num_batched_tokens = 16384
        elif is_llama: # better fit for Llama-family of models
            max_num_seqs = 32
            max_num_batched_tokens = 65536
        else:
            max_num_seqs = 64
            max_num_batched_tokens = 131072

        llm_kwargs = dict(
            model=args.model,
            trust_remote_code=True,
            max_model_len=ctx_len,
            seed=SEED,
            gpu_memory_utilization=0.94,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            tensor_parallel_size=args.tp,
            kv_cache_dtype="fp8",
            calculate_kv_scales=False,
            enable_prefix_caching=True,
        )
        if is_llama:
            llm_kwargs["enforce_eager"] = True
            
        if args.quantization:
            llm_kwargs['quantization'] = args.quantization
        llm = LLM(**llm_kwargs)

    # ════════════════════════════════════════════════════════════════════
    # PASS 1: vLLM — generation + logprobs
    # ════════════════════════════════════════════════════════════════════
    if ckpt_pass1 is not None:
        print('\n=== Pass 1: loaded from checkpoint ===')
        gen_texts = ckpt_pass1['gen_texts']
        all_token_logprobs = ckpt_pass1['all_token_logprobs']
    else:
        print('\n=== Pass 1: vLLM generation + logprobs ===')
        gen_params = SamplingParams(
            temperature=0.6, max_tokens=MAX_NEW_TOKENS,
            logprobs=DEEPCONF_TOPK, seed=SEED,
        )

        t0 = time.time()
        vllm_outputs = llm.generate(prompts, gen_params)
        print(f'Generation done in {time.time() - t0:.1f}s')

        gen_texts = []
        all_token_logprobs = []
        for output in vllm_outputs:
            gen_texts.append(output.outputs[0].text)
            raw_lps = output.outputs[0].logprobs or []
            serialized = []
            for td in raw_lps:
                if td is None:
                    serialized.append(None)
                else:
                    serialized.append({
                        tok_id: {'logprob': lp.logprob, 'rank': lp.rank, 'decoded_token': lp.decoded_token}
                        for tok_id, lp in td.items()
                    })
            all_token_logprobs.append(serialized)

        save_checkpoint(out_dir, 'pass1', {
            'gen_texts': gen_texts,
            'all_token_logprobs': all_token_logprobs,
        })

    # ── Early accuracy check ──────────────────────────────────────────
    def _score(fa, i):
        ia = data_df.iloc[i]['input_args']
        if args.dataset == 'hle' and isinstance(ia, (list, tuple)) and len(ia) > 1:
            return scorer(fa, data_df.iloc[i]['targets'], ia[1])
        return scorer(fa, data_df.iloc[i]['targets'])

    early_correct = []
    for i in range(N):
        fa = extract_final_answer_text(gen_texts[i])
        c = _score(fa, i)
        early_correct.append(c)
    valid_early = [c for c in early_correct if c is not None]
    n_correct = sum(c for c in valid_early)
    n_none = sum(1 for c in early_correct if c is None)
    print(f'\n*** Early accuracy: {n_correct}/{len(valid_early)} = {n_correct/len(valid_early):.3f} '
          f'({n_none} unparseable) ***\n')

    # ════════════════════════════════════════════════════════════════════
    # PASS 1b: vLLM — sampling K continuations per step
    # ════════════════════════════════════════════════════════════════════
    all_spans = [get_steps_from_think_block(gt) for gt in gen_texts]
    print(f'Avg steps per example: {np.mean([len(s) for s in all_spans]):.1f}')
    all_step_tok_counts = []
    for gt, spans in zip(gen_texts, all_spans):
        if spans:
            tidx = build_charspan_to_token_idxs(tokenizer, gt, spans)
            all_step_tok_counts.extend([len(ids) for ids in tidx])
    if all_step_tok_counts:
        print(f'Step length (tokens): avg={np.mean(all_step_tok_counts):.0f}, '
              f'min={min(all_step_tok_counts)}, max={max(all_step_tok_counts)}')

    loop = asyncio.get_event_loop()

    if args.skip_sampling:
        print('\n=== Pass 1b: SKIPPED (--skip_sampling) ===')
        sample_texts = None
        prefix_index = None
        sampled_spans = None
        sampled_indices = None
    elif ckpt_pass1b is not None:
        print('\n=== Pass 1b: loaded from checkpoint ===')
        sample_texts = ckpt_pass1b['sample_texts']
        prefix_index = ckpt_pass1b['prefix_index']
        sampled_spans = ckpt_pass1b['sampled_spans']
        sampled_indices = ckpt_pass1b['sampled_indices']
    else:
        print('\n=== Pass 1b: Sampling continuations ===')
        # Subsample steps: keep at most MAX_SAMPLE_STEPS per example (always first + last)
        random.seed(SEED)
        sampled_spans = []
        sampled_indices = []
        for spans in all_spans:
            n_steps = len(spans)
            if n_steps <= MAX_SAMPLE_STEPS:
                sampled_spans.append(spans)
                sampled_indices.append(list(range(n_steps)))
            else:
                # Always keep first (0) and last (n_steps-1); sample the rest
                middle = list(range(1, n_steps - 1))
                chosen = sorted(random.sample(middle, MAX_SAMPLE_STEPS - 2))
                keep = [0] + chosen + [n_steps - 1]
                sampled_spans.append([spans[j] for j in keep])
                sampled_indices.append(keep)
        total_sampled = sum(len(s) for s in sampled_spans)
        total_orig = sum(len(s) for s in all_spans)
        print(f'Subsampled steps for sampling: {total_sampled}/{total_orig} '
              f'(max {MAX_SAMPLE_STEPS}/example)')

        prefix_prompts, prefix_index = build_prefix_prompts(prompts, gen_texts, sampled_spans)
        sample_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=SAMPLE_MAX_TOKENS, n=K,
        )
        print(f'Sampling {len(prefix_prompts)} prompts x K={K} = {len(prefix_prompts) * K} generations')

        def batched(xs, bs):
            for i in range(0, len(xs), bs):
                yield i, xs[i:i+bs]

        sample_texts = []
        # if only sampling, bump the batch size for better throughput
        SAMPLE_BATCH_SIZE = 4096 if args.only_sampling else 1024

        t0 = time.time()
        for start_idx, prompt_batch in batched(prefix_prompts, SAMPLE_BATCH_SIZE):
            batch_outputs = await loop.run_in_executor(None, llm.generate, prompt_batch, sample_params)
            for out in batch_outputs:
                sample_texts.append([c.text for c in out.outputs])

            del batch_outputs
            torch.cuda.empty_cache()

            done = min(start_idx + SAMPLE_BATCH_SIZE, len(prefix_prompts))
            if done % 1024 < SAMPLE_BATCH_SIZE or done == len(prefix_prompts):
                print(f' Sampling progress: {done}/{len(prefix_prompts)} prefixes')

        print(f'Sampling done in {time.time() - t0:.1f}s')

        save_checkpoint(out_dir, 'pass1b', {
            'sample_texts': sample_texts,
            'prefix_index': prefix_index,
            'sampled_spans': sampled_spans,
            'sampled_indices': sampled_indices,
        })

    # Free vLLM
    if llm is not None:
        del llm
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # PASS 2 / Phases A+B: DeepConf + HF forward + RCC + per-step DeepConf
    # In --only_sampling mode we skip both Phase A (DeepConf) and Phase B (HF/RCC)
    # entirely. We construct a minimal `pre` from the pickled gen_texts so that
    # Phase C (sampling consistency) and the final write step can run.
    # ════════════════════════════════════════════════════════════════════
    ckpt_phaseB = None if args.only_sampling else load_checkpoint(out_dir, 'phaseB')
    if args.only_sampling:
        print('\n=== Phases A+B: SKIPPED (--only_sampling) ===')
        pre = []
        for i in range(N):
            gt = gen_texts[i]
            think_match = re.search(r'<think>(.*?)</think>', gt, re.DOTALL)
            if think_match:
                trace_clean = think_match.group(1).strip()
            elif gt.strip().startswith('</think>') or '</think>' not in gt:
                close = gt.find('</think>')
                trace_clean = gt[:close].strip() if close != -1 else gt.strip()
            else:
                trace_clean = gt.strip()
            final_answer = extract_final_answer_text(gt)
            correct = _score(final_answer, i)
            spans = all_spans[i]
            pre.append({
                'idx': i,
                'input_args': data_df.iloc[i]['input_args'],
                'targets': data_df.iloc[i]['targets'],
                'gen_text': gt,
                'trace_clean': trace_clean,
                'all_confs': [],
                'final_answer': final_answer,
                'correct': correct,
                'dc_avg': None, 'dc_tail': None, 'dc_bot': None,
                'rcc_qs': [], 'rcc_ps': [], 'rcc_conf': None,
                'step_spans': spans,
                'step_texts': [sp[2] for sp in spans],
                'step_deepconf_raw': [None] * len(spans),
                'step_deepconf': [None] * len(spans),
            })
    elif ckpt_phaseB is not None:
        print('\n=== Phases A+B: loaded from checkpoint ===')
        pre = ckpt_phaseB['pre']
    else:
        print('\n=== Pass 2: HF forward pass for RCC ===')
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception as e:
            print(f'  Could not configure SDPA backends: {e}')

        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation='sdpa',
        )
        print('  Loaded HF model in bf16 with SDPA (FlashAttention-2 backend)')
        hf_model.eval()

        # Hook only the last layer's hidden state
        _last_hidden_holder = {}

        def _last_hidden_hook(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            _last_hidden_holder['h'] = h.detach()

        try:
            last_layer = hf_model.model.layers[-1]
        except AttributeError:
            last_layer = list(hf_model.modules())[-2]
        _hook_handle = last_layer.register_forward_hook(_last_hidden_hook)

        # Phase A: DeepConf (from vLLM logprobs)
        print('\n=== Phase A: DeepConf ===')
        pre = []
        for i in tqdm(range(N), desc='DeepConf'):
            gt = gen_texts[i]
            all_confs = extract_deepconf_from_vllm_logprobs(all_token_logprobs[i], topk=DEEPCONF_TOPK)

            # Extract think block for trace
            think_match = re.search(r'<think>(.*?)</think>', gt, re.DOTALL)
            if think_match:
                trace_clean = think_match.group(1).strip()
            elif gt.strip().startswith('</think>') or '</think>' not in gt:
                close = gt.find('</think>')
                trace_clean = gt[:close].strip() if close != -1 else gt.strip()
            else:
                trace_clean = gt.strip()

            final_answer = extract_final_answer_text(gt)
            correct = _score(final_answer, i)

            pre.append({
                'idx': i,
                'input_args': data_df.iloc[i]['input_args'],
                'targets': data_df.iloc[i]['targets'],
                'gen_text': gt,
                'trace_clean': trace_clean,
                'all_confs': all_confs,
                'final_answer': final_answer,
                'correct': correct,
                'dc_avg': calculate_mean_confidence(all_confs),
                'dc_tail': calculate_tail_confidence(all_confs),
                'dc_bot': calculate_bottom_window_confidence(all_confs),
            })

        # Phase B: HF forward pass + RCC + per-step DeepConf (fused)
        print('\n=== Phase B: HF forward + RCC + per-step DeepConf ===')

        PHASE_B_CHECKPOINT_EVERY = 50

        device = hf_model.device

        def _run_hf_and_rcc():
            for i in tqdm(range(N), desc='HF forward + RCC'):
                p = pre[i]
                spans = all_spans[i]

                # Tokenize ONCE per example (with offsets) — used by both forward & per-step DeepConf
                prompt_enc = tokenizer(prompts[i], add_special_tokens=False)
                gen_enc = tokenizer(gen_texts[i], add_special_tokens=False, return_offsets_mapping=True)
                prompt_ids_list = prompt_enc['input_ids']
                gen_ids_list = gen_enc['input_ids']
                gen_offsets = gen_enc['offset_mapping']
                prompt_len = len(prompt_ids_list)
                gen_len = len(gen_ids_list)

                full_ids = torch.tensor(
                    [prompt_ids_list + gen_ids_list], dtype=torch.long, device=device,
                )

                _last_hidden_holder.clear()
                with torch.no_grad():
                    hf_model(full_ids, use_cache=False)
                H = _last_hidden_holder['h'][0].to(torch.float32)  # stays on GPU
                _last_hidden_holder.clear()

                gen_token_probs_full = extract_chosen_token_probs(all_token_logprobs[i])
                if gen_token_probs_full.numel() < gen_len:
                    pad = torch.zeros(gen_len - gen_token_probs_full.numel(), dtype=torch.float32)
                    gen_token_probs = torch.cat([gen_token_probs_full, pad])
                else:
                    gen_token_probs = gen_token_probs_full[:gen_len]
                gen_token_probs = gen_token_probs.to(device)

                # RCC computation on GPU
                qs, ps, matched_texts = rcc_scores_for_spans_from_hidden(
                    H, prompt_len, gen_token_probs, p['gen_text'], spans,
                    tokenizer, mu=RCC_MU, delta=RCC_DELTA,
                )
                p['rcc_qs'] = qs
                p['rcc_ps'] = ps
                p['rcc_conf'] = float(ps[-1]) if ps else 0.0
                p['step_spans'] = spans
                p['step_texts'] = [sp[2] for sp in spans]

                step_deepconf_raw = []
                step_deepconf = []
                for (cs, ce, text) in spans:
                    sc = deepconf_for_char_range_with_offsets(
                        p['all_confs'], gen_offsets, cs, ce,
                    )
                    raw = calculate_mean_confidence(sc) if sc else None
                    step_deepconf_raw.append(raw)
                    step_deepconf.append(normalize_deepconf(raw))
                p['step_deepconf_raw'] = step_deepconf_raw
                p['step_deepconf'] = step_deepconf

                del H, gen_token_probs, full_ids

        await loop.run_in_executor(None, _run_hf_and_rcc)

        _hook_handle.remove()
        del hf_model
        torch.cuda.empty_cache()
        print('HF forward + RCC complete')

        save_checkpoint(out_dir, 'phaseB', {'pre': pre})
        # Drop partial once final is saved
        partial = _ckpt_path(out_dir, 'phaseB_partial')
        if os.path.exists(partial):
            os.remove(partial)

    # ════════════════════════════════════════════════════════════════════
    # Phase C: Sampling consistency
    # ════════════════════════════════════════════════════════════════════
    print('\n=== Phase C: Sampling consistency ===')

    if args.skip_sampling:
        print('=== Phase C: SKIPPED (--skip_sampling) ===')
        sampling_results = [
            {'step_sampling_conf': [None] * len(p['step_texts']), 'sampling_conf': -1}
            for p in pre
        ]
    elif (ckpt_phaseC := load_checkpoint(out_dir, 'phaseC')) is not None:
        print('=== Phase C: loaded from checkpoint ===')
        sampling_results = ckpt_phaseC['sampling_results']
    else:
        # ── Sampling consistency (local vLLM judge) ──
        JUDGE_MODEL = args.judge_model
        print(f'\nLoading judge model: {JUDGE_MODEL}')
        judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, use_fast=True, trust_remote_code=True)
        if judge_tokenizer.pad_token_id is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        judge_llm = LLM(model=JUDGE_MODEL, trust_remote_code=True, max_model_len=20480,
                         seed=SEED, gpu_memory_utilization=0.3)
        judge_params = SamplingParams(temperature=0.0, max_tokens=10)

        judge_tasks = build_judge_tasks(sample_texts, prefix_index, sampled_spans, K)
        print(f'Total sampling consistency checks: {len(judge_tasks)}')

        # Build judge prompts
        judge_prompts = []
        for t in judge_tasks:
            raw_prompt = UNCERTAINTY_PROMPT.format(context=t['context'], assertion=t['assertion'])
            judge_prompts.append(format_chat_if_available(judge_tokenizer, raw_prompt))

        t0 = time.time()
        judge_outputs = judge_llm.generate(judge_prompts, judge_params)
        print(f'Sampling consistency done in {time.time() - t0:.1f}s')

        # Parse yes/no scores
        judge_scores = []
        for output in judge_outputs:
            raw = output.outputs[0].text.strip()
            m = re.search(UNCERTAINTY_PATTERN, raw)
            if m:
                judge_scores.append(UNCERTAINTY_MAPPING.get(m.group(1).lower(), 0.5))
            else:
                judge_scores.append(0.5)

        del judge_llm, judge_tokenizer
        torch.cuda.empty_cache()

        # Remap subsampled step indices (si within sampled_spans) back to full step indices
        remapped_tasks = []
        for t in judge_tasks:
            ei, si_sub = t['ei'], t['si']
            si_full = sampled_indices[ei][si_sub]
            remapped_tasks.append({**t, 'si': si_full})
        sampling_results = compute_step_confidences(remapped_tasks, judge_scores, all_spans, N)

        save_checkpoint(out_dir, 'phaseC', {'sampling_results': sampling_results})

    # ════════════════════════════════════════════════════════════════════
    # Build results
    # ════════════════════════════════════════════════════════════════════
    print('\n=== Building results ===')

    # Decisiveness placeholder; filled by run_decisiveness_genai.py post-hoc!
    for p in pre:
        n_steps = len(p['step_texts'])
        p['step_dec'] = [None] * n_steps
        p['step_faith_rcc'] = [None] * n_steps
        p['step_faith_sampling'] = [None] * n_steps
        p['step_faith_deepconf'] = [None] * n_steps

    rows = []
    for i, p in enumerate(pre):
        valid_decs = [d for d in p['step_dec'] if d is not None and d != -1]
        avg_dec = float(np.mean(valid_decs)) if valid_decs else -1

        def _avg_faith(vals):
            v = [f for f in vals if f is not None]
            return float(np.mean(v)) if v else -1

        valid_dc = [v for v in p['step_deepconf'] if v is not None]
        deepconf_conf = float(np.mean(valid_dc)) if valid_dc else -1

        rows.append({
            'idx': p['idx'],
            'question': p['input_args'][0],
            'prompt': prompts[i],
            'gold': p['targets'][0],
            'final_answer_extracted': p['final_answer'],
            'correct': p['correct'],
            'num_steps': len(p['step_texts']),
            'step_texts': p['step_texts'],
            'step_rcc_q': p['rcc_qs'],
            'step_rcc_p': p['rcc_ps'],
            'step_deepconf': p['step_deepconf'],
            'step_deepconf_raw': p['step_deepconf_raw'],
            'step_sampling_conf': sampling_results[i]['step_sampling_conf'],
            'step_dec_scores': p['step_dec'],
            'step_faith_rcc': p['step_faith_rcc'],
            'step_faith_sampling': p['step_faith_sampling'],
            'step_faith_deepconf': p['step_faith_deepconf'],
            'rcc_confidence': p['rcc_conf'],
            'deepconf_confidence': deepconf_conf,
            'deepconf_avg_raw': p['dc_avg'],
            'deepconf_tail_raw': p['dc_tail'],
            'deepconf_bottom10_raw': p['dc_bot'],
            'deepconf_num_tokens': len(p['all_confs']),
            'sampling_conf': sampling_results[i]['sampling_conf'],
            'avg_decisiveness': avg_dec,
            'faithfulness_rcc': _avg_faith(p['step_faith_rcc']),
            'faithfulness_sampling': _avg_faith(p['step_faith_sampling']),
            'faithfulness_deepconf': _avg_faith(p['step_faith_deepconf']),
            'generated_text': p['gen_text'],
        })

    df = pd.DataFrame(rows)

    # ════════════════════════════════════════════════════════════════════
    # Print summary metrics
    # ════════════════════════════════════════════════════════════════════
    valid = df[df['correct'].notna()]
    print(f'\n{"="*60}')
    print(f'RESULTS: {args.run_name} ({args.model}, {args.dataset}, n={N})')
    print(f'{"="*60}')
    print(f'Accuracy: {valid["correct"].sum()}/{len(valid)} = {valid["correct"].mean():.3f}')
    print(f'Avg steps: {valid["num_steps"].mean():.1f}')
    if not args.only_sampling:
        print(f'\nRCC confidence:     mean={valid["rcc_confidence"].mean():.3f}')
        print(f'DeepConf confidence: mean={valid["deepconf_confidence"].mean():.3f}')
    else:
        print('\nRCC confidence:     SKIPPED (--only_sampling)')
        print('DeepConf confidence: SKIPPED (--only_sampling)')
    if args.skip_sampling:
        print('Sampling confidence: SKIPPED')
    else:
        print(f'Sampling confidence: mean={valid["sampling_conf"].mean():.3f}')

    vd = valid[valid['avg_decisiveness'] != -1]
    print(f'\nDecisiveness: mean={vd["avg_decisiveness"].mean():.3f} (n={len(vd)})')

    pairings = [
        ('faithfulness_rcc',      'rcc_confidence',      'RCC'),
        ('faithfulness_deepconf', 'deepconf_confidence',  'DeepConf'),
        ('faithfulness_sampling', 'sampling_conf',        'Sampling'),
    ]
    for faith_col, conf_col, label in pairings:
        vf = valid[valid[faith_col] != -1]
        if len(vf) == 0:
            continue
        print(f'\n  {label}:')
        print(f'    Faithfulness mean: {vf[faith_col].mean():.3f} (n={len(vf)})')
        f_vals = vf[faith_col].tolist()
        c_vals = vf[conf_col].tolist()
        mfg = compute_mfg(f_vals)
        cmfg = compute_cmfg(f_vals, c_vals, num_bins=10)
        cmfg_star, _ = compute_cmfg_star(f_vals, c_vals, num_bins=10)
        print(f'    MFG={mfg:.3f}, cMFG={cmfg:.3f}, cMFG*={cmfg_star:.3f}')

    # ════════════════════════════════════════════════════════════════════
    # Save outputs
    # ════════════════════════════════════════════════════════════════════
    _ILLEGAL_XLSX_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def _clean_for_xlsx(v):
        if isinstance(v, str):
            return _ILLEGAL_XLSX_RE.sub('', v)
        if isinstance(v, list):
            return [_clean_for_xlsx(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_clean_for_xlsx(x) for x in v)
        return v

    df = df.map(_clean_for_xlsx) if hasattr(df, 'map') else df.applymap(_clean_for_xlsx)

    example_path = os.path.join(out_dir, f'results_{args.dataset}_examples.xlsx')
    step_path = os.path.join(out_dir, f'results_{args.dataset}_step_level.xlsx')

    if args.only_sampling and os.path.exists(example_path):
        print('\n=== Merging sampling results into existing Excel (--only_sampling) ===')

        existing_ex = pd.read_excel(example_path)
        # Index existing by 'idx' for fast lookup
        idx_to_row = {int(r['idx']): i for i, r in existing_ex.iterrows() if pd.notna(r.get('idx'))}

        if 'sampling_conf' in existing_ex.columns:
            existing_ex['sampling_conf'] = pd.to_numeric(
                existing_ex['sampling_conf'], errors='coerce').astype('float64')
        else:
            existing_ex['sampling_conf'] = np.nan
        if 'step_sampling_conf' not in existing_ex.columns:
            existing_ex['step_sampling_conf'] = None
        existing_ex['step_sampling_conf'] = existing_ex['step_sampling_conf'].astype(object)

        merged = 0
        for _, new_row in df.iterrows():
            ei = int(new_row['idx'])
            if ei not in idx_to_row:
                continue
            ri = idx_to_row[ei]
            existing_ex.at[ri, 'sampling_conf'] = float(new_row['sampling_conf']) \
                if pd.notna(new_row.get('sampling_conf')) else np.nan
            existing_ex.at[ri, 'step_sampling_conf'] = str(new_row['step_sampling_conf'])
            merged += 1

        existing_ex.to_excel(example_path, index=False)
        print(f'  Merged sampling into {merged} example rows; saved {example_path}')

        # ── Step-level merge ──
        if os.path.exists(step_path):
            existing_step = pd.read_excel(step_path)
            if 'sampling_conf' in existing_step.columns:
                existing_step['sampling_conf'] = pd.to_numeric(
                    existing_step['sampling_conf'], errors='coerce').astype('float64')
            else:
                existing_step['sampling_conf'] = np.nan

            new_map = {}
            for _, drow in df.iterrows():
                ei = int(drow['idx'])
                ssc = drow.get('step_sampling_conf', [])
                if not isinstance(ssc, list):
                    continue
                for si, val in enumerate(ssc):
                    new_map[(ei, si)] = val

            updated = 0
            for ri, srow in existing_step.iterrows():
                key = (int(srow['idx']) if pd.notna(srow.get('idx')) else None,
                       int(srow['step_idx']) if pd.notna(srow.get('step_idx')) else None)
                if key in new_map and new_map[key] is not None:
                    val = new_map[key]
                    try:
                        existing_step.at[ri, 'sampling_conf'] = float(val) if val is not None else np.nan
                        updated += 1
                    except (TypeError, ValueError):
                        pass
            existing_step.to_excel(step_path, index=False)
            print(f'  Merged sampling into {updated} step rows; saved {step_path}')
        else:
            print(f'  WARNING: {step_path} not found; skipping step-level merge')
    else:
        df.to_excel(example_path, index=False)
        print(f'\nSaved example-level results: {example_path}')

        step_rows = []
        for _, row in df.iterrows():
            for si in range(row['num_steps']):
                step_rows.append({
                    'idx': row['idx'],
                    'step_idx': si,
                    'step_text': row['step_texts'][si] if si < len(row['step_texts']) else None,
                    'rcc_q': row['step_rcc_q'][si] if si < len(row.get('step_rcc_q', [])) else None,
                    'rcc_p': row['step_rcc_p'][si] if si < len(row.get('step_rcc_p', [])) else None,
                    'deepconf': row['step_deepconf'][si] if si < len(row.get('step_deepconf', [])) else None,
                    'sampling_conf': row['step_sampling_conf'][si] if si < len(row.get('step_sampling_conf', [])) else None,
                    'dec': row['step_dec_scores'][si] if si < len(row.get('step_dec_scores', [])) else None,
                    'faith_rcc': row['step_faith_rcc'][si] if si < len(row.get('step_faith_rcc', [])) else None,
                    'faith_deepconf': row['step_faith_deepconf'][si] if si < len(row.get('step_faith_deepconf', [])) else None,
                    'faith_sampling': row['step_faith_sampling'][si] if si < len(row.get('step_faith_sampling', [])) else None,
                    'correct': row['correct'],
                })
        step_df = pd.DataFrame(step_rows)
        step_df = step_df.map(_clean_for_xlsx) if hasattr(step_df, 'map') else step_df.applymap(_clean_for_xlsx)
        step_df.to_excel(step_path, index=False)
        print(f'Saved step-level results: {step_path} ({len(step_df)} steps)')

    summary_path = os.path.join(out_dir, 'summary.txt')

    if args.only_sampling:
        new_samp = float(valid['sampling_conf'].mean()) if len(valid) and 'sampling_conf' in valid.columns else None
        existing = ''
        if os.path.exists(summary_path):
            existing = open(summary_path).read()

        if new_samp is not None and not (np.isnan(new_samp)):
            samp_line = f'Sampling conf: {new_samp:.3f}\n'
            # Replace any existing Sampling conf line (incl. SKIPPED)
            new_text, n_sub = re.subn(r'^Sampling conf:.*\n?', samp_line, existing, flags=re.MULTILINE)
            if n_sub == 0:
                if 'DeepConf conf:' in new_text:
                    new_text = re.sub(r'(DeepConf conf:.*\n)', r'\1' + samp_line, new_text, count=1)
                else:
                    new_text = new_text + samp_line
            with open(summary_path, 'w') as f:
                f.write(new_text)
            print(f'Updated summary (sampling conf): {summary_path}')
        else:
            print(f'No valid sampling_conf to write to summary')

    else:
        # Plots
        save_plots(df, out_dir, args.dataset)

        # Summary text
        with open(summary_path, 'w') as f:
            f.write(f'Run: {args.run_name}\n')
            f.write(f'Model: {args.model}\n')
            f.write(f'Dataset: {args.dataset}\n')
            f.write(f'N: {N}, K: {K}\n')
            f.write(f'Hedge: {hedge_key}\n')
            f.write(f'System prompt: {args.sys_prompt}\n')
            f.write(f'Accuracy: {valid["correct"].mean():.3f}\n')
            f.write(f'RCC conf: {valid["rcc_confidence"].mean():.3f}\n')
            f.write(f'DeepConf conf: {valid["deepconf_confidence"].mean():.3f}\n')
            if not args.skip_sampling:
                f.write(f'Sampling conf: {valid["sampling_conf"].mean():.3f}\n')
            else:
                f.write('Sampling conf: SKIPPED\n')
            if len(vd) > 0:
                f.write(f'Decisiveness: {vd["avg_decisiveness"].mean():.3f}\n')
            for faith_col, conf_col, label in pairings:
                vf = valid[valid[faith_col] != -1]
                if len(vf) > 0:
                    f_vals = vf[faith_col].tolist()
                    c_vals = vf[conf_col].tolist()
                    mfg = compute_mfg(f_vals)
                    cmfg = compute_cmfg(f_vals, c_vals, num_bins=10)
                    cmfg_star, _ = compute_cmfg_star(f_vals, c_vals, num_bins=10)
                    f.write(f'Faithfulness ({label}): {vf[faith_col].mean():.3f}\n')
                    f.write(f'  MFG={mfg:.3f}, cMFG={cmfg:.3f}, cMFG*={cmfg_star:.3f}\n')
        print(f'Saved summary: {summary_path}')

    if not args.only_sampling:
        cleanup_checkpoints(out_dir)
    print(f'\nDone! All outputs in {out_dir}/')
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Full faithfulness experiment pipeline')
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument('--run_name', type=str, required=True, help='Output folder name')
    parser.add_argument('--n', type=int, default=100, help='Number of examples')
    parser.add_argument('--k', type=int, default=10, help='Samples per step for sampling confidence')
    parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                        help='Small instruct model for consistency judging')
    parser.add_argument('--hedge', type=str, default='blank',
                        choices=list(HEDGE_PROMPTS.keys()),
                        help='Hedge prompt variant to prepend to user prompt')
    parser.add_argument('--sys_prompt', type=str, default=None,
                        choices=list(SYSTEM_PROMPTS.keys()),
                        help='System prompt key for metacognitive calibration (passed as system role in chat template, if possible)')
    parser.add_argument('--tp', type=int, default=1,
                        help='Tensor parallel size (number of GPUs for vLLM)')
    parser.add_argument('--quantization', type=str, default=None,
                        choices=[None, 'awq', 'awq_marlin', 'gptq', 'gptq_marlin', 'fp8', 'gguf'],
                        help='vLLM quantization format (e.g. awq for AWQ INT4 quants of DS-R1)')
    parser.add_argument('--skip_sampling', action='store_true',
                        help='Skip Pass 1b sampling and Phase C judge; produce only DeepConf+RCC')
    parser.add_argument('--only_sampling', action='store_true',
                        help='Run ONLY Pass 1b (sampling) + Phase C (judge consistency). '
                             'Loads existing _checkpoint_pass1.pkl, skips Phase A/B (DeepConf, RCC), '
                             'merges sampling results into the existing Excel files in --run_name. '
                             'Mutually exclusive with --skip_sampling.')
    args = parser.parse_args()
    if args.only_sampling and args.skip_sampling:
        parser.error('--only_sampling and --skip_sampling are mutually exclusive')

    asyncio.run(run_experiment(args))

if __name__ == '__main__':
    main()
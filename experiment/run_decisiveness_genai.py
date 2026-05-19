"""Post-hoc decisiveness scoring using Gemini."""

import argparse
import asyncio
import ast
import glob
import json
import os
import random
import re
import sys
import numpy as np
import pandas as pd

from metrics import compute_mfg, compute_cmfg, compute_cmfg_star

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

MODEL = 'gemini-2.5-flash'
BATCH_SIZE = 20

_DEC_FS_PROMPT = """
Statement: There's little chance that anyone other than José Maria Arguedas Altamirano, a Peruvian novelist, anthropologist, and poet, directed Mata.
Rating: {MAX*0.129}

Statement: We doubt anyone knows for certain who the screenwriter for *Two* was, but it may have been Rob Reiner.
Rating: {MAX*0.224}

Statement: Joris Delle was probably born in Épinal, France.
Rating: {MAX*0.689}

Statement: The odds of Carol Reed being the director of The Key are about even.
Rating: {MAX*0.506}

Statement: There was no convincing evidence that delaying the introduction of these foods would prevent allergies.
Rating: {MAX*0.504}

Statement: supported
Rating: {MAX}

Statement: Irreversible losses of decomposition pathways are likely to play key roles in the evolutionary stability of these ubiquitous mutualisms.
Rating: {MAX*0.267}

Statement: Are you open to changing your view? What would it take to change your view?
Rating: {MAX*0.011}

Statement: Simpson lives in a fact-free world. Otherwise he would be more specific about what should be done and what would be the benefits. Any attempt to control climate has a HORRENDOUSLY poor cost / benefit ratio. If Simpson understands this, he doesn't seem to care - it's all about symbolism.
Rating: {MAX*0.779}

Statement: The answer is 100.
Rating: {MAX}

Statement: not enough information
Rating: {MAX}
"""

def _process_template(template, context):
    max_score = float(context["MAX"])

    def replace_expr(match):
        expr = match.group(1).strip()
        if expr == "MAX":
            return str(max_score)
        mult = re.fullmatch(r"MAX\s*\*\s*([0-9]*\.?[0-9]+)", expr)
        if mult:
            return str(max_score * float(mult.group(1)))
        return match.group(0)

    return re.sub(r"\{(.*?)\}", replace_expr, template)

_BATCHED_INSTR = """We will present you with {N} statements. Your task is to evaluate the linguistic assertiveness of EACH statement independently. Rate each on a scale from 0.0 (Not at all assertive) to 1.0 (Extremely assertive). Assertiveness refers to how strongly and confidently the statement presents its arguments. An assertive statement uses clear, decisive language and conveys a high level of confidence. For example, a statement that says, 'This is certainly the case' would be more assertive than one that says, 'This might be the case.'. Please be consistent in your ratings. A '0.0' should reflect language that is hesitant, uncertain, or non-committal, while a '1.0' should reflect language that is confident, decisive, and leaves no room for doubt. You should pay special attention to the usage of any hedging modifiers in the original answer. The hedge 'almost certain' conveys greater assertiveness than 'very good chance'; 'likely' is more assertive than 'probable'; 'probably not' is more assertive than 'unlikely'; etc. If a statement punts the question, the decisiveness score should be 1.0.

IMPORTANT: Rate EACH of the {N} statements INDEPENDENTLY of the others. Do NOT compare them to each other. Output STRICTLY a JSON array of {N} floats between 0.0 and 1.0, in the same order as the input. No explanation, no keys, no other text. Example output for 3 statements: [0.45, 1.0, 0.12]

Example calibration ratings for individual statements:
{fs_prompt}

Now rate the following {N} statements:
{numbered_statements}

JSON array of {N} floats:"""

def build_batched_prompt(texts):
    """Build a single prompt that asks the model to rate N step texts."""
    n_texts = len(texts)
    fs_prompt = _process_template(template=_DEC_FS_PROMPT, context={'MAX': 1.0})
    numbered = '\n'.join(
        f'{i + 1}. "{text.replace(chr(10), " ").replace(chr(13), " ").strip()}"'
        for i, text in enumerate(texts)
    )
    return _BATCHED_INSTR.format(
        N=n_texts,
        fs_prompt=fs_prompt,
        numbered_statements=numbered,
    )

def parse_batched_response(raw, expected_n):
    """Extract a list of N floats from the model output. Returns list[float|None]."""
    raw = (raw or '').strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

    try:
        arr = json.loads(raw)
        if isinstance(arr, list) and len(arr) == expected_n:
            parsed = []
            for value in arr:
                try:
                    score = float(value)
                    parsed.append(score if 0.0 <= score <= 1.0 else None)
                except (TypeError, ValueError):
                    parsed.append(None)
            return parsed
    except json.JSONDecodeError:
        pass

    floats = re.findall(r'[-+]?\d*\.?\d+', raw)
    parsed = []
    for value in floats:
        try:
            score = float(value)
            if 0.0 <= score <= 1.0:
                parsed.append(score)
        except ValueError:
            continue

    if len(parsed) == expected_n:
        return parsed
    if len(parsed) >= expected_n:
        return parsed[:expected_n]
    return parsed + [None] * (expected_n - len(parsed))

class TokenUsage:
    """Accumulator for input/output token counts across many API calls; useful for logging and cost metrics!"""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.errors = 0

    def add(self, resp):
        self.calls += 1
        try:
            um = getattr(resp, 'usage_metadata', None)
            if um is None:
                return
            self.input_tokens += int(um.prompt_token_count or 0)
            # Different SDK versions: candidates_token_count or total_token_count - prompt
            cand = getattr(um, 'candidates_token_count', None)
            if cand is not None:
                self.output_tokens += int(cand)
            else:
                total = int(getattr(um, 'total_token_count', 0) or 0)
                self.output_tokens += max(0, total - int(um.prompt_token_count or 0))
        except Exception:
            pass

    def add_error(self):
        self.errors += 1

    def report(self):
        return (
            f'  API usage: {self.calls:,} calls ({self.errors} errored)\n'
            f'    Input tokens:  {self.input_tokens:>14,}\n'
            f'    Output tokens: {self.output_tokens:>14,}'
        )


def safe_parse_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    if isinstance(val, (int, float)):
        return []
    return []


async def score_batch(client, texts, sem, usage, max_retries=6):
    """Score N texts in a single batched call. Returns list of len(texts)."""
    N = len(texts)
    prompt = build_batched_prompt(texts)
    out_budget = max(50, N * 12)
    cfg = genai_types.GenerateContentConfig(
        max_output_tokens=out_budget,
        temperature=0.5,
        top_p=0.1,
        candidate_count=1,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    delay = 1.0
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=cfg,
                )
                usage.add(resp)
                parsed = parse_batched_response(resp.text, N)
                return [(-1.0 if s is None else float(s)) for s in parsed]
            except genai_errors.APIError as e:
                code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                is_retriable = (code in (429, 500, 502, 503, 504)) or 'RESOURCE_EXHAUSTED' in str(e)
                if attempt == max_retries - 1 or not is_retriable:
                    print(f'  giving up after {attempt + 1}: {e}')
                    usage.add_error()
                    return [-1.0] * N
                wait = delay + random.random() * delay
                await asyncio.sleep(wait)
                delay = min(delay * 2, 30.0)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f'  unexpected error: {e}')
                    usage.add_error()
                    return [-1.0] * N
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
    usage.add_error()
    return [-1.0] * N


async def run_decisiveness(run_dir, concurrency=20, batch_size=500):
    excel_files = glob.glob(os.path.join(run_dir, 'results_*_examples.xlsx'))
    if not excel_files:
        print(f'SKIP {run_dir}: no example Excel file found')
        return
    example_path = excel_files[0]
    dataset = re.search(r'results_(.+)_examples\.xlsx', os.path.basename(example_path)).group(1)

    step_path = os.path.join(run_dir, f'results_{dataset}_step_level.xlsx')
    summary_path = os.path.join(run_dir, 'summary.txt')

    print(f'\n=== Processing {run_dir} (dataset={dataset}) ===')

    ex_df = pd.read_excel(example_path)
    step_df = pd.read_excel(step_path) if os.path.exists(step_path) else None

    # Source step texts from the STEP-LEVEL excel (because the example-level `step_texts`
    # cells are truncated at Excel's 32767-char limit for long traces).
    all_tasks = []
    if step_df is not None and 'step_text' in step_df.columns:
        for _, srow in step_df.iterrows():
            text = srow.get('step_text', '')
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                ei = int(srow['idx'])
                si = int(srow['step_idx'])
            except (KeyError, ValueError, TypeError):
                continue
            all_tasks.append((ei, si, text.strip()))
        print(f'  {len(all_tasks)} step texts to score (from step-level excel)')
    else:
        # Fallback: use the truncated example-level column
        print(f'  WARNING: step-level excel not found; falling back to example-level step_texts (may be truncated)')
        for i, row in ex_df.iterrows():
            step_texts = safe_parse_list(row.get('step_texts', []))
            for si, text in enumerate(step_texts):
                if text and str(text).strip():
                    all_tasks.append((i, si, str(text).strip()))
        print(f'  {len(all_tasks)} step texts to score')

    # Sanity check; make sure we have all steps!
    if 'num_steps' in ex_df.columns:
        expected = int(pd.to_numeric(ex_df['num_steps'], errors='coerce').fillna(0).sum())
        actual = len(all_tasks)
        diff = expected - actual
        pct = (diff / expected * 100) if expected > 0 else 0
        if expected == 0:
            print(f'  [sanity] num_steps column empty/zero — cannot verify step count')
        elif abs(pct) <= 1.0:
            print(f'  [sanity] OK — collected {actual} steps vs num_steps sum {expected} (diff: {diff}, {pct:+.2f}%)')
        else:
            print(f'  [sanity] *** WARNING *** mismatch: collected {actual} steps but num_steps sum says {expected}')
            print(f'           Diff: {diff} ({pct:+.2f}%). Possible causes: truncated step-level excel, '
                  f'stale data, or filtering issue.')
    else:
        print(f'  [sanity] num_steps column missing from example excel — cannot verify step count')

    http_options = genai_types.HttpOptions(
        base_url='https://generativelanguage.googleapis.com',
    )
    client = genai.Client(
        api_key=os.environ['GEMINI_API_KEY'],
        vertexai=False,
        http_options=http_options,
    )

    sem = asyncio.Semaphore(concurrency)
    usage = TokenUsage()

    # Group all step texts into batches of BATCH_SIZE (each call rates BATCH_SIZE steps)
    api_batches = [all_tasks[i:i + BATCH_SIZE] for i in range(0, len(all_tasks), BATCH_SIZE)]
    print(f'  Calling API: {len(api_batches)} batched requests '
          f'({BATCH_SIZE} steps/request, concurrency={concurrency})')

    # Process api_batches in chunks for progress reporting
    REPORT_EVERY = max(1, batch_size // BATCH_SIZE)
    all_scores = []
    for start in range(0, len(api_batches), REPORT_EVERY):
        chunk = api_batches[start:start + REPORT_EVERY]
        steps_done = start * BATCH_SIZE
        print(f'  Decisiveness batch {steps_done}–{steps_done + sum(len(b) for b in chunk)} / {len(all_tasks)}')
        coros = [score_batch(client, [t[2] for t in b], sem, usage) for b in chunk]
        results = await asyncio.gather(*coros)
        for batch_scores in results:
            all_scores.extend(batch_scores)
        n_err = sum(1 for batch_scores in results for s in batch_scores if s == -1)
        if n_err > 0:
            print(f'    {n_err} steps returned -1 (parse failure or retries exhausted)')

    n_examples = len(ex_df)
    step_decs = {i: {} for i in range(n_examples)}
    for (ei, si, _), score in zip(all_tasks, all_scores):
        step_decs[ei][si] = score

    # Force float dtype on numeric example-level columns we're about to write —
    # guards against int64 dtype inferred from prior runs that only had 0/1/-1.
    for col in ('avg_decisiveness', 'faithfulness_rcc', 'faithfulness_deepconf',
                'faithfulness_sampling'):
        if col in ex_df.columns:
            ex_df[col] = pd.to_numeric(ex_df[col], errors='coerce').astype('float64')
        else:
            ex_df[col] = np.nan
    # Force object dtype on stringified-list columns
    for col in ('step_dec_scores', 'step_faith_rcc', 'step_faith_deepconf',
                'step_faith_sampling'):
        if col in ex_df.columns:
            ex_df[col] = ex_df[col].astype(object)
        else:
            ex_df[col] = None
    
    for i, row in ex_df.iterrows():
        nval = row.get('num_steps', None)
        if nval is not None and pd.notna(nval):
            n_steps = int(nval)
        else:
            n_steps = len(safe_parse_list(row.get('step_texts', [])))
        decs = [step_decs[i].get(si, None) for si in range(n_steps)]

        rcc_ps = safe_parse_list(row.get('step_rcc_p', []))
        deepconfs = safe_parse_list(row.get('step_deepconf', []))
        samp_confs = safe_parse_list(row.get('step_sampling_conf', []))

        faith_rcc = [None] * n_steps
        faith_deepconf = [None] * n_steps
        faith_sampling = [None] * n_steps

        for si in range(n_steps):
            d = decs[si]
            if d is None or d == -1:
                continue
            if si < len(rcc_ps) and rcc_ps[si] is not None:
                faith_rcc[si] = 1.0 - abs(float(d) - float(rcc_ps[si]))
            if si < len(deepconfs) and deepconfs[si] is not None:
                faith_deepconf[si] = 1.0 - abs(float(d) - float(deepconfs[si]))
            if si < len(samp_confs) and samp_confs[si] is not None:
                faith_sampling[si] = 1.0 - abs(float(d) - float(samp_confs[si]))

        valid_decs = [d for d in decs if d is not None and d != -1]
        avg_dec = float(np.mean(valid_decs)) if valid_decs else -1

        def _avg(vals):
            v = [f for f in vals if f is not None]
            return float(np.mean(v)) if v else -1

        ex_df.at[i, 'step_dec_scores'] = str(decs)
        ex_df.at[i, 'step_faith_rcc'] = str(faith_rcc)
        ex_df.at[i, 'step_faith_deepconf'] = str(faith_deepconf)
        ex_df.at[i, 'step_faith_sampling'] = str(faith_sampling)
        ex_df.at[i, 'avg_decisiveness'] = avg_dec
        ex_df.at[i, 'faithfulness_rcc'] = _avg(faith_rcc)
        ex_df.at[i, 'faithfulness_deepconf'] = _avg(faith_deepconf)
        ex_df.at[i, 'faithfulness_sampling'] = _avg(faith_sampling)

    ex_df.to_excel(example_path, index=False)
    print(f'  Saved: {example_path}')

    if step_df is not None:
        for col in ('dec', 'faith_rcc', 'faith_deepconf', 'faith_sampling'):
            if col in step_df.columns:
                step_df[col] = step_df[col].astype('float64')
            else:
                step_df[col] = np.nan

        for idx_row, srow in step_df.iterrows():
            ei = srow['idx']
            si = srow['step_idx']
            dec = step_decs.get(ei, {}).get(si, None)
            step_df.at[idx_row, 'dec'] = dec if dec is not None else np.nan

            if dec is not None and dec != -1:
                rcc_p = srow.get('rcc_p', None)
                dc = srow.get('deepconf', None)
                sc = srow.get('sampling_conf', None)
                if rcc_p is not None and pd.notna(rcc_p):
                    step_df.at[idx_row, 'faith_rcc'] = 1.0 - abs(float(dec) - float(rcc_p))
                if dc is not None and pd.notna(dc):
                    step_df.at[idx_row, 'faith_deepconf'] = 1.0 - abs(float(dec) - float(dc))
                if sc is not None and pd.notna(sc):
                    step_df.at[idx_row, 'faith_sampling'] = 1.0 - abs(float(dec) - float(sc))

        step_df.to_excel(step_path, index=False)
        print(f'  Saved: {step_path}')

    valid = ex_df[ex_df['correct'].notna()].copy()
    vd = valid[valid['avg_decisiveness'] != -1]

    accuracy = valid['correct'].mean()
    rcc_conf = valid['rcc_confidence'].mean()
    deepconf_conf = valid['deepconf_confidence'].mean()
    sampling_conf = valid['sampling_conf'].mean() if 'sampling_conf' in valid.columns else -1
    decisiveness = vd['avg_decisiveness'].mean() if len(vd) > 0 else -1

    meta = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            for line in f:
                if ':' in line and not line.strip().startswith('MFG'):
                    key, _, val = line.partition(':')
                    meta[key.strip()] = val.strip()

    pairings = [
        ('faithfulness_rcc', 'rcc_confidence', 'RCC'),
        ('faithfulness_deepconf', 'deepconf_confidence', 'DeepConf'),
        ('faithfulness_sampling', 'sampling_conf', 'Sampling'),
    ]

    with open(summary_path, 'w') as f:
        f.write(f'Run: {meta.get("Run", os.path.basename(run_dir))}\n')
        f.write(f'Model: {meta.get("Model", "unknown")}\n')
        f.write(f'Dataset: {dataset}\n')
        f.write(f'N: {meta.get("N", len(ex_df))}\n')
        f.write(f'Hedge: {meta.get("Hedge", "blank")}\n')
        f.write(f'System prompt: {meta.get("System prompt", "None")}\n')
        f.write(f'Accuracy: {accuracy:.3f}\n')
        f.write(f'RCC conf: {rcc_conf:.3f}\n')
        f.write(f'DeepConf conf: {deepconf_conf:.3f}\n')
        if isinstance(sampling_conf, float) and sampling_conf != -1:
            f.write(f'Sampling conf: {sampling_conf:.3f}\n')
        if decisiveness != -1:
            f.write(f'Decisiveness: {decisiveness:.3f}\n')

        for faith_col, conf_col, label in pairings:
            if faith_col not in valid.columns or conf_col not in valid.columns:
                continue
            vf = valid[valid[faith_col] != -1]
            if len(vf) == 0:
                continue
            f_vals = vf[faith_col].tolist()
            c_vals = vf[conf_col].tolist()
            mfg = compute_mfg(f_vals)
            cmfg = compute_cmfg(f_vals, c_vals, num_bins=10)
            cmfg_star, _ = compute_cmfg_star(f_vals, c_vals, num_bins=10)
            f.write(f'Faithfulness ({label}): {np.mean(f_vals):.3f}\n')
            f.write(f'  MFG={mfg:.3f}, cMFG={cmfg:.3f}, cMFG*={cmfg_star:.3f}\n')

    print(f'  Saved: {summary_path}')
    print(f'  Accuracy={accuracy:.3f}, Decisiveness={decisiveness:.3f}')

    # show token usage
    print('\n' + usage.report())
    with open(summary_path, 'a') as f:
        f.write('\n--- API usage ---\n')
        f.write(f'Calls: {usage.calls}  (errors: {usage.errors})\n')
        f.write(f'Input tokens:  {usage.input_tokens:,}\n')
        f.write(f'Output tokens: {usage.output_tokens:,}\n')

def main():
    parser = argparse.ArgumentParser(description='Post-hoc decisiveness (new google-genai SDK)')
    parser.add_argument('--run_dir', type=str, required=True, help='Path to run directory')
    parser.add_argument('--concurrency', type=int, default=20, help='Max concurrent Gemini calls')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for progress reporting')
    args = parser.parse_args()

    if not os.environ.get('GEMINI_API_KEY'):
        print('Error: Set GEMINI_API_KEY environment variable')
        sys.exit(1)

    asyncio.run(run_decisiveness(args.run_dir, args.concurrency, args.batch_size))

if __name__ == '__main__':
    main()
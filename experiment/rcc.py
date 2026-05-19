"""RCC computation helpers."""

import math
import torch

def row_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

def compute_token_probs_from_generate(scores, gen_ids: torch.Tensor) -> torch.Tensor:
    probs = []
    for t, logits in enumerate(scores):
        if logits.dim() == 2:
            logits = logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok = gen_ids[t].item()
        probs.append(torch.exp(logp[tok]))
    return torch.stack(probs, dim=0)

def build_charspan_to_token_idxs(tokenizer, gen_text: str, spans):
    enc = tokenizer(gen_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    token_idxs = []
    for (cs, ce, _) in spans:
        idxs = []
        for i, (a, b) in enumerate(offsets):
            if b <= cs:
                continue
            if a >= ce:
                break
            if a == b == 0 and len(gen_text) > 0:
                continue
            idxs.append(i)
        token_idxs.append(idxs)
    return token_idxs

def extract_chosen_token_probs(token_logprobs_list):
    probs = []
    for token_dict in token_logprobs_list:
        if token_dict is None:
            probs.append(0.0)
            continue
        chosen = None
        for value in token_dict.values():
            if value.get('rank') == 1:
                chosen = value
                break
        if chosen is None:
            chosen = max(token_dict.values(), key=lambda value: value['logprob'])
        probs.append(float(math.exp(chosen['logprob'])))
    return torch.tensor(probs, dtype=torch.float32)

def rcc_scores_for_spans_from_hidden(H, prompt_len, gen_token_probs, gen_text,
                                     spans, tokenizer, mu=0.5, delta=0.4):
    """Compute RCC q and p scores using pre-extracted hidden states."""
    if not spans:
        return [], [], []
    tidx = build_charspan_to_token_idxs(tokenizer, gen_text, spans)
    kept = [(ix, sp[2]) for ix, sp in zip(tidx, spans) if len(ix) > 0]
    if not kept:
        return [], [], []
    tidx = [k[0] for k in kept]
    texts = [k[1] for k in kept]

    d = H.shape[-1]
    E_prev = H[:prompt_len, :]
    prev_p = None
    qs, ps = [], []
    for idxs in tidx:
        fp = [prompt_len + t for t in idxs if (prompt_len + t) < H.shape[0]]
        if not fp:
            continue
        E_i = H[fp, :]
        c_i = gen_token_probs[idxs].float()
        A = (E_prev @ E_i.T) / math.sqrt(d)
        A_n = row_softmax(A)
        W = (A_n >= mu).float()
        r = W @ c_i
        nz = r[r != 0]
        q = nz.mean().item() if nz.numel() > 0 else (r.mean().item() if r.numel() > 0 else 0.0)
        qs.append(float(q))
        p = q if prev_p is None else (delta * q + (1 - delta) * prev_p)
        ps.append(float(p))
        prev_p = p
        E_prev = E_i
    return qs, ps, texts

def rcc_scores_for_spans(model, tokenizer, full_input_ids, prompt_len, gen_text,
                         gen_token_probs, spans, mu=0.5, delta=0.4):
    if not spans:
        return [], [], []
    tidx = build_charspan_to_token_idxs(tokenizer, gen_text, spans)
    kept = [(ix, sp[2]) for ix, sp in zip(tidx, spans) if len(ix) > 0]
    if not kept:
        return [], [], []
    tidx = [k[0] for k in kept]
    texts = [k[1] for k in kept]
    with torch.no_grad():
        out = model(full_input_ids.to(model.device), output_hidden_states=True, use_cache=False)
        H = out.hidden_states[-1][0]
    d = H.shape[-1]
    E_prev = H[:prompt_len, :]
    prev_p = None
    qs, ps = [], []
    for idxs in tidx:
        fp = [prompt_len + t for t in idxs if (prompt_len + t) < H.shape[0]]
        if not fp:
            continue
        E_i = H[fp, :]
        c_i = gen_token_probs[idxs].to(H.device)
        A = (E_prev @ E_i.T) / math.sqrt(d)
        A_n = row_softmax(A)
        W = (A_n >= mu).float()
        r = W @ c_i
        nz = r[r != 0]
        q = nz.mean().item() if nz.numel() > 0 else (r.mean().item() if r.numel() > 0 else 0.0)
        qs.append(float(q))
        p = q if prev_p is None else (delta * q + (1 - delta) * prev_p)
        ps.append(float(p))
        prev_p = p
        E_prev = E_i
    return qs, ps, texts
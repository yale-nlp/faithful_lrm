"""Few utility functions, primarily for string manipulation."""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def format_chat_if_available(tokenizer, user_text: str, system_prompt: str = None) -> str:
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return system_prompt + "\n\n" + user_text
    return user_text

def extract_final_answer_text(gen_text: str) -> str:
    for p in [r"(?im)^\s*final\s*answer\s*[:\-]\s*(.+)$", r"(?im)^\s*answer\s*[:\-]\s*(.+)$"]:
        m = re.search(p, gen_text)
        if m:
            return m.group(1).strip()
    boxed = re.search(r"\\boxed\{(.+?)\}", gen_text)
    if boxed:
        return boxed.group(1).strip()
    lines = [ln.strip() for ln in gen_text.splitlines() if ln.strip()]
    return lines[-1] if lines else gen_text.strip()

def load_model(model_name, load_in_4bit=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_steps_from_think_block(gen_text):
    """Split a generated reasoning trace into paragraph-level step spans.

    Returns a list of (start_char, end_char, text) tuples in coordinates of the
    original generated text. If no think tags are present, the full generation
    is treated as the reasoning trace.
    """
    close_match = re.search(r'</think>', gen_text)
    open_match = re.search(r'<think>', gen_text)

    if close_match and (not open_match or open_match.start() > close_match.start()):
        block = gen_text[:close_match.start()]
        block_start = 0
    elif open_match and close_match:
        block = gen_text[open_match.end():close_match.start()]
        block_start = open_match.end()
    else:
        block = gen_text
        block_start = 0

    spans = []
    pos = 0
    for chunk in re.split(r'(\n\n+)', block):
        if re.fullmatch(r'\n\n+', chunk) or not chunk.strip():
            pos += len(chunk)
            continue
        char_start = block_start + pos
        char_end = char_start + len(chunk)
        spans.append((char_start, char_end, chunk.strip()))
        pos += len(chunk)

    return spans
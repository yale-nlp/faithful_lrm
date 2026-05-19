"""Helpers for dataset loading and management."""

import ast
import random
import re

import pandas as pd
from datasets import load_dataset

def _numbered_choices(choices):
    return "\n".join(f"{idx + 1}. {choice}" for idx, choice in enumerate(choices))

def score_mcq(final_answer, targets):
    """Score numeric multiple-choice answers against a 1-indexed target."""
    try:
        extracted_num = int(re.search(r"\d+", final_answer).group())
        return extracted_num == targets[0]
    except Exception:
        return None

def score_openqa(final_answer, targets):
    """Score short answers by case-insensitive substring match."""
    pred_lower = final_answer.lower().strip()
    for target in targets:
        target_lower = str(target).lower().strip()
        if target_lower in pred_lower or pred_lower in target_lower:
            return True
    return False

def score_yes_no(final_answer, targets):
    """Score yes/no answers with simple lexical extraction."""
    pred = final_answer.strip().lower()
    gold = targets[0].strip().lower()
    yes_match = re.search(r"\byes\b", pred)
    no_match = re.search(r"\bno\b", pred)
    if yes_match and not no_match:
        return gold == "yes"
    if no_match and not yes_match:
        return gold == "no"
    return pred == gold

def load_aime(n=50, seed=42):
    """Load AIME math competition problems with integer answers."""
    random.seed(seed)
    ds = load_dataset("qq8933/AIME_1983_2024", split="train")
    data_df = ds.to_pandas()

    def prepare_inputs(row):
        question = row["Question"].strip()
        answer = str(row["Answer"]).strip()
        return (question,), [answer]

    inputs_and_targets = data_df.apply(prepare_inputs, axis=1)
    data_df["input_args"], data_df["targets"] = zip(*inputs_and_targets)
    data_df = data_df.sample(n=min(n, len(data_df)), random_state=seed).reset_index(drop=True)
    return data_df

def build_prompt_aime(input_args):
    question = input_args[0]
    return (
        f"{question}\n Solve this step by step. The answer is an integer between "
        f"0 and 999. Return your final answer in \\boxed{{}}, after thinking. "
        f"Use a MAXIMUM of 20-30 steps (paragraphs). <think>\n"
    )

def score_aime(final_answer, targets):
    """Score AIME by exact integer match."""
    try:
        pred = int(re.sub(r"[^0-9]", "", final_answer.strip()))
        gold = int(re.sub(r"[^0-9]", "", str(targets[0]).strip()))
        return pred == gold
    except Exception:
        return None

def load_hle(n=50, seed=42):
    """Load text-only Humanity's Last Exam examples."""
    random.seed(seed)
    ds = load_dataset("cais/hle", split="test")
    data_df = ds.to_pandas()

    if "image" in data_df.columns:
        before = len(data_df)
        data_df = data_df[data_df["image"].apply(
            lambda value: not isinstance(value, str) or len(value) == 0
        )].reset_index(drop=True)
        print(f"  HLE text-only filter: {len(data_df)}/{before} rows")

    def prepare_inputs(row):
        question = row["question"].strip()
        answer = str(row["answer"]).strip()
        answer_type = row.get("answer_type", "exactMatch")
        return (question, answer_type), [answer]

    results = [prepare_inputs(row) for _, row in data_df.iterrows()]
    data_df["input_args"] = [item[0] for item in results]
    data_df["targets"] = [item[1] for item in results]
    data_df = data_df.sample(n=min(n, len(data_df)), random_state=seed).reset_index(drop=True)
    return data_df

def build_prompt_hle(input_args):
    question = input_args[0]
    answer_type = input_args[1] if len(input_args) > 1 else "exactMatch"
    if answer_type == "multipleChoice":
        instruction = (
            "Answer with the letter or number of the correct option only. "
            "Return just that single choice in \\boxed{}, after thinking."
        )
    else:
        instruction = "Answer concisely. Return your answer in \\boxed{}, after thinking."
    return f"{question}\n {instruction} Use a MAXIMUM of 20-30 steps (paragraphs). <think>\n"

def score_hle(final_answer, targets, answer_type="exactMatch"):
    """Score HLE MCQ by strict choice match and short answers by substring."""
    if answer_type == "multipleChoice":
        pred = final_answer.strip().upper()
        gold = str(targets[0]).strip().upper()
        pred_match = re.match(r"^\s*([A-Z0-9])\b", pred)
        gold_match = re.match(r"^\s*([A-Z0-9])\b", gold)
        pred_letter = pred_match.group(1) if pred_match else pred
        gold_letter = gold_match.group(1) if gold_match else gold
        return pred_letter == gold_letter
    return score_openqa(final_answer, targets)

def load_supergpqa(n=50, seed=42, difficulty="hard"):
    """Load SuperGPQA graduate-level multiple-choice examples."""
    random.seed(seed)
    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    data_df = ds.to_pandas()

    if difficulty is not None:
        data_df = data_df[data_df["difficulty"] == difficulty].reset_index(drop=True)

    def prepare_inputs(row):
        question = row["question"].strip()
        options = row["options"]
        answer_letter = row["answer_letter"]
        correct_idx = ord(answer_letter.upper()) - ord("A") + 1
        return (question, _numbered_choices(options)), [correct_idx]

    inputs_and_targets = data_df.apply(prepare_inputs, axis=1)
    data_df["input_args"], data_df["targets"] = zip(*inputs_and_targets)
    data_df = data_df.sample(n=min(n, len(data_df)), random_state=seed).reset_index(drop=True)
    return data_df

def build_prompt_supergpqa(input_args):
    question, choices = input_args
    n_options = len(choices.strip().split("\n"))
    return (
        f"{question}\n{choices}\n Provide your answer as one of the answer choices, "
        f"1-{n_options}. Return your answer in \\boxed{{}}, after thinking. "
        f"Use a MAXIMUM of 20-30 steps (paragraphs). <think>\n"
    )

def score_supergpqa(final_answer, targets):
    """Score SuperGPQA by numeric multiple-choice match."""
    return score_mcq(final_answer, targets)

LEGALBENCH_SUBSETS = {
    "hearsay": "Does the presented evidence contain hearsay?",
    "personal_jurisdiction": "Does the court have personal jurisdiction over the defendant?",
    "telemarketing_sales_rule": "Does this scenario violate the Telemarketing Sales Rule?",
    "diversity_1": "Is there diversity jurisdiction in this case?",
    "diversity_2": "Is there diversity jurisdiction in this case?",
    "diversity_3": "Is there diversity jurisdiction in this case?",
    "contract_nli_confidentiality_of_agreement": "Does this clause address confidentiality of the agreement?",
    "contract_nli_limited_use": "Does this clause address limited use?",
    "contract_nli_survival_of_obligations": "Does this clause address survival of obligations?",
    "contract_nli_no_licensing": "Does this clause address no licensing?",
    "learned_hands_benefits": "Does this post involve a benefits law issue?",
    "learned_hands_crime": "Does this post involve a criminal law issue?",
    "corporate_lobbying": "Would this company support or oppose this bill?",
    "overruling": "Does this sentence overrule a prior holding?",
    "legal_reasoning_causality": "Does the legal reasoning in this text establish causality?",
}

def load_legalbench(n=50, seed=42):
    """Load a pooled LegalBench yes/no reasoning set."""
    random.seed(seed)
    all_rows = []

    for subset, question_template in LEGALBENCH_SUBSETS.items():
        try:
            ds = load_dataset("nguha/legalbench", subset, split="test")
        except Exception as exc:
            print(f"  Warning: could not load LegalBench subset {subset}: {exc}")
            continue

        text_col = None
        for col in ["text", "contract", "sentence", "question"]:
            if col in ds.column_names:
                text_col = col
                break

        for row in ds:
            if text_col is None:
                text_cols = [col for col in ds.column_names if col != "answer"]
                text = "\n".join(f"{col}: {row[col]}" for col in text_cols if row[col]).strip()
            else:
                text = row[text_col].strip()
            answer = row["answer"].strip()
            question = f"{text}\n\n{question_template}"
            all_rows.append({"question": question, "answer": answer, "subset": subset})

    data_df = pd.DataFrame(all_rows)

    def prepare_inputs(row):
        return (row["question"],), [row["answer"].lower()]

    inputs_and_targets = data_df.apply(prepare_inputs, axis=1)
    data_df["input_args"], data_df["targets"] = zip(*inputs_and_targets)
    data_df = data_df.sample(n=min(n, len(data_df)), random_state=seed).reset_index(drop=True)
    return data_df

def build_prompt_legalbench(input_args):
    question = input_args[0]
    return (
        f"{question}\n Answer Yes or No. Return your answer in \\boxed{{}}, "
        f"after thinking. Use a MAXIMUM of 20-30 steps (paragraphs). <think>\n"
    )

def score_legalbench(final_answer, targets):
    """Score LegalBench by yes/no match."""
    return score_yes_no(final_answer, targets)

def load_musr(n=50, seed=42):
    """Load MuSR examples from all three task splits."""
    random.seed(seed)
    all_dfs = []
    for split in ["murder_mysteries", "object_placements", "team_allocation"]:
        ds = load_dataset("TAUR-Lab/MuSR", split=split)
        all_dfs.append(ds.to_pandas())
    data_df = pd.concat(all_dfs, ignore_index=True)

    def prepare_inputs(row):
        narrative = row["narrative"].strip()
        question = row["question"].strip()
        choices = row["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        correct_idx = row["answer_index"] + 1
        full_question = f"{narrative}\n\nQuestion: {question}"
        return (full_question, _numbered_choices(choices)), [correct_idx]

    inputs_and_targets = data_df.apply(prepare_inputs, axis=1)
    data_df["input_args"], data_df["targets"] = zip(*inputs_and_targets)
    data_df = data_df.sample(n=min(n, len(data_df)), random_state=seed).reset_index(drop=True)
    return data_df

def build_prompt_musr(input_args):
    question, choices = input_args
    n_options = len(choices.strip().split("\n"))
    return (
        f"{question}\n\n{choices}\n Provide your answer as one of the answer "
        f"choices, 1-{n_options}. Return your answer in \\boxed{{}}, after thinking. "
        f"Use a MAXIMUM of 20-30 steps (paragraphs). <think>\n"
    )

def score_musr(final_answer, targets):
    """Score MuSR by numeric multiple-choice match."""
    return score_mcq(final_answer, targets)
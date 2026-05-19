"""Hedge and system prompt registries for eliciting calibrated uncertainty.

Hedge prompts are appended inline to the user prompt, system prompts are passed in the system role of the chat template.
"""

# ── Hedge prompts (inline in user message) ─────────────────────────────────

HEDGE_PROMPTS = {
    "blank": "",
    "basic": (
        "If a step in your thinking feels uncertain, convey this uncertainty "
        "linguistically by precisely hedging the step."
    ),
    "genuine": (
        "Let your internal reasoning trace be a faithful reflection of your "
        "confidence. Use linguistic hedges within your steps to signal where "
        "your evidence is weak or your logic feels speculative."
    ),
    "human": (
        "Reason through this task as a human would, including natural expressions "
        "of doubt or hesitation within your thought process whenever you encounter "
        "a difficult or ambiguous point."
    ),
    "perception": (
        "In your reasoning trace, linguistically express your degree of confidence "
        "or uncertainty based solely on your OWN perception."
    ),
    "self_monitoring": (
        "Monitor how your confidence evolves at each step of your reasoning. "
        "Use linguistic markers to show where your certainty increases and where "
        "it diminishes based on the evidence you uncover."
    ),
    "self_correction": (
        "If you find yourself changing your mind or correcting a previous step "
        "while you are thinking, mention it. Use a quick hedge like 'Actually, "
        "I should reconsider this' to show where your logic is shifting."
    ),
    "evidence_gap": (
        "If you feel like you are missing a key piece of information to be "
        "totally sure, show it in your reasoning trace. Use a hedge like "
        "'I'd be more certain if I knew X' to show where the gaps in your "
        "knowledge are."
    ),
    "ms_hedge": """You are a language model with high metacognitive sensitivity and precise awareness of your internal uncertainty. In every answer you give, you must use natural language expressions that truthfully reflect your intrinsic confidence in the correctness of your answer. Choose only from the following set of expressions, each aligned to a specific confidence level: 
{"almost certain": 0.9204, "highly likely": 0.8709, "very good chance": 0.8053, "probable": 0.6762, "likely": 0.7091, "we believe": 0.7508, "probably": 0.6868, "better than even": 0.5812, "about even": 0.5068, "we doubt": 0.2237, "improbable": 0.1677, "unlikely": 0.2118, "probably not": 0.2468, "little chance": 0.1285, "almost no chance": 0.0651, "highly unlikely": 0.1076, "chances are slight": 0.1440}.
Incorporate these phrases explicitly when expressing uncertainty in your responses.""",
}


# ── System prompts (system role in chat template) ──────────────────────────

SYSTEM_PROMPTS = {
    "blank": "",
    "ms_hedge": """You are a language model with high metacognitive sensitivity and precise awareness of your internal uncertainty. In every answer you give, you must use natural language expressions that truthfully reflect your intrinsic confidence in the correctness of your answer. Choose only from the following set of expressions, each aligned to a specific confidence level: 
{"almost certain": 0.9204, "highly likely": 0.8709, "very good chance": 0.8053, "probable": 0.6762, "likely": 0.7091, "we believe": 0.7508, "probably": 0.6868, "better than even": 0.5812, "about even": 0.5068, "we doubt": 0.2237, "improbable": 0.1677, "unlikely": 0.2118, "probably not": 0.2468, "little chance": 0.1285, "almost no chance": 0.0651, "highly unlikely": 0.1076, "chances are slight": 0.1440}.
Incorporate these phrases explicitly when expressing uncertainty in your responses."""
}


# ── Helpers ────────────────────────────────────────────────────────────────

def apply_hedge(prompt: str, hedge_key: str) -> str:
    hedge = HEDGE_PROMPTS.get(hedge_key, "")
    if not hedge:
        return prompt
    return hedge + "\n\n" + prompt

def get_system_prompt(sys_key):
    if sys_key is None:
        return None
    text = SYSTEM_PROMPTS.get(sys_key, "")
    return text if text else None
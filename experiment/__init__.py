from .dataset_utils import (
    load_aime, build_prompt_aime, score_aime,
    load_hle, build_prompt_hle, score_hle,
    load_supergpqa, build_prompt_supergpqa, score_supergpqa,
    load_legalbench, build_prompt_legalbench, score_legalbench,
    load_musr, build_prompt_musr, score_musr,
)
from .deepconf import (
    compute_deepconf_token_confs,
    compute_deepconf_for_token_range,
    calculate_mean_confidence,
    calculate_tail_confidence,
    calculate_bottom_window_confidence,
    extract_deepconf_from_vllm_logprobs,
    normalize_deepconf,
    deepconf_for_char_range,
    deepconf_for_char_range_with_offsets,
)
from .rcc import (
    compute_token_probs_from_generate,
    build_charspan_to_token_idxs,
    extract_chosen_token_probs,
    rcc_scores_for_spans_from_hidden,
    rcc_scores_for_spans,
)
from .metrics import compute_mfg, compute_cmfg, compute_cmfg_star
from .utils import format_chat_if_available, extract_final_answer_text, get_steps_from_think_block, load_model
from .sampling import build_prefix_prompts, build_judge_tasks, compute_step_confidences

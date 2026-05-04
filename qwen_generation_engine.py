"""
Backward-compatible exports for the legacy root-level generation module.

New code should import from src.decoding.* and src.models.* directly.
"""

from src.decoding.candidate_generator import ContrastiveCandidateGenerator, EGMHCDGenerator, SampleMajorityGenerator
from src.decoding.label_cd import EGLabelCDScorer, EntroGraphLabelCD, LabelCDResult
from src.decoding.token_cd import (
    StepTrace,
    TokenCDConfig,
    TokenCDGenerator,
    TokenCDResult,
    TokenContrastiveDecoder,
    contrastive_generate,
    contrastive_logits as _contrastive_logits,
    top_k_top_p_filtering as _top_k_top_p_filtering,
)
from src.models.qwen25vl_adapter import (
    append_token as _append_token,
    build_text_only_inputs,
    clone_inputs as _clone_inputs,
    get_tokenizer as _get_tokenizer,
    move_inputs_to_device as _move_inputs_to_device,
)


MHCDGenerator = SampleMajorityGenerator


__all__ = [
    "EGLabelCDScorer",
    "ContrastiveCandidateGenerator",
    "EntroGraphLabelCD",
    "EGMHCDGenerator",
    "LabelCDResult",
    "MHCDGenerator",
    "SampleMajorityGenerator",
    "StepTrace",
    "TokenCDConfig",
    "TokenCDGenerator",
    "TokenCDResult",
    "TokenContrastiveDecoder",
    "_append_token",
    "_clone_inputs",
    "_contrastive_logits",
    "_get_tokenizer",
    "_move_inputs_to_device",
    "_top_k_top_p_filtering",
    "build_text_only_inputs",
    "contrastive_generate",
]


if __name__ == "__main__":
    print("Use src.decoding.* for EG-MHCD-AE v2 generation engines.")

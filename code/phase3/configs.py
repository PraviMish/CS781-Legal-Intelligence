"""Exact Phase 3 archived experiment catalogue and paths expected by CLI.

Every name is a published notebook experiment. C3 and D3 perform postprocessing
of separately trained predictors and are not misrepresented as trained models.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    id: str
    architecture: str = "efficient_gqa"
    foundation: str = "efficient_gqa_700M"
    method: str = "lora"
    context: str = "sentence"
    domain: str = "joint"
    loss: str = "ce"
    epochs: int = 3
    lr: float = .00015
    student: bool = False
    teacher: str | None = None


RUNS = {
    x.id: x for x in (
        Experiment("A1_decoder_lora", "decoder_only", "decoder_only"),
        Experiment("A2_encdec_lora", "encoder_decoder", "encoder_decoder"),
        Experiment("A3_prefix_lora", "prefix_lm", "prefix_lm"),
        Experiment("A4_gqa100_lora", "efficient_gqa", "efficient_gqa_100M"),
        Experiment("A5_gqa700_lora"),
        Experiment("B1_lorma_plus", method="lorma_plus"),
        Experiment("B2_ia3", method="ia3"),
        Experiment("B3_prefix", method="prefix"),
        Experiment("B4_full", method="full", lr=.00003),
        Experiment("C1_previous_context", context="previous"),
        Experiment("C2_bidirectional_context", context="bidirectional"),
        Experiment("D1_CL_only", domain="CL"),
        Experiment("D2_IT_only", domain="IT"),
        Experiment("E1_student_supervised", "decoder_only", "decoder_only", "student_full",
                   student=True, lr=.00003),
        Experiment("E2_student_distilled", "decoder_only", "decoder_only", "student_full",
                   loss="kd", student=True, lr=.00003, teacher="B4_full"),
        Experiment("F1_balanced_lora", loss="balanced"),
    )
}
POSTPROCESSING = {
    "C3_document_viterbi": {"base": "A5_gqa700_lora", "transition_strength_grid":
                           [0.0, .15, .3, .5, 1.0, 1.5, 2.0], "selected_dev_strength": .3},
    "D3_domain_specialist_hybrid": {"CL": "D1_CL_only", "IT": "D2_IT_only"},
}
DEFAULT_FOUNDATION_FILES = {
    "decoder_only": "models/decoder_only/best.pt",
    "encoder_decoder": "models/encoder_decoder/best.pt",
    "prefix_lm": "models/prefix_lm/best.pt",
    "efficient_gqa_100M": "models/efficient_gqa/milestones/best_100M.pt",
    "efficient_gqa_700M": "models/efficient_gqa/milestones/best_up_to_700M.pt",
}


def experiment(name):
    if name not in RUNS:
        raise ValueError(f"Unknown trained experiment {name}")
    return RUNS[name]

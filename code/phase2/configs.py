"""Recorded settings for each Phase 2 model run."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Run:
    architecture: str
    method: str
    foundation: str
    backbone_lr: float
    head_lr: float
    batch: int = 4
    accumulation: int = 4
    epochs: int = 4
    patience: int = 0
    min_epochs: int = 0
    outside_weight: float = 1.0
    entity_weight: float = 1.0
    full_data: bool = False
    init_offset: int = 0
    shuffle_offset: int = 0
    inference_batch: int = 4
    save_every: int = 100
    round_warmup: bool = False


GQA = "efficient_gqa"
DECODER = "decoder_only"
GQA_700 = "efficient_gqa_700M"
DECODER_100 = "decoder_only_100M"
LORA = "encoder_decoder_100m_lora_v1"
GQA_FULL = "gqa_700m_full_bio_v1"
DECODER_FULL = "decoder_100m_full_bio_v1"
STUDENT_KD = "gqa_7layer_distilled"
FULL70_MEMBERS = ("gqa_full70_v1", "decoder_full70_v1", "student_kd_full70_v1")
ORIGINAL_MEMBERS = (GQA_FULL, DECODER_FULL, STUDENT_KD)
ENSEMBLE_MEMBERS = {"original": ORIGINAL_MEMBERS, "full70": FULL70_MEMBERS}
STUDENT_LAYERS = (0, 2, 4, 6, 8, 10, 12)
KD_TEMPERATURE = 2.0
KD_CE_WEIGHT = 0.7

RUNS = {
    "gqa_100m_frozen_probe": Run(
        GQA, "probe", "efficient_gqa_100M", 0, 2e-3, batch=8, accumulation=1,
        entity_weight=2, inference_batch=8, save_every=0),
    GQA_FULL: Run(GQA, "full", GQA_700, 2e-5, 3e-4, patience=2,
                  inference_batch=8, save_every=150),
    DECODER_FULL: Run(DECODER, "full", DECODER_100, 2e-5, 3e-4, patience=2,
                      inference_batch=8, round_warmup=True),
    LORA: Run("encoder_decoder", "lora", "encoder_decoder_100M", 3e-4, 8e-4,
              epochs=6, patience=2, init_offset=401, shuffle_offset=1000,
              inference_batch=8, save_every=50),
    "encoder_decoder_100m_lora_balanced_v2": Run(
        "encoder_decoder", "lora", "encoder_decoder_100M", 3e-4, 8e-4, epochs=6,
        patience=2, min_epochs=4, outside_weight=0.2, init_offset=401,
        shuffle_offset=1000, inference_batch=8, save_every=50),
    "prefix_lm_100m_lorma_plus_v1": Run(
        "prefix_lm", "lorma", "prefix_lm_100M", 3e-4, 8e-4, batch=2, accumulation=8,
        epochs=6, patience=2, outside_weight=0.5, init_offset=501, shuffle_offset=2000,
        save_every=50),
    "gqa_7layer_supervised": Run(
        GQA, "student", GQA_700, 3e-5, 3e-4, init_offset=789, shuffle_offset=1770,
        save_every=80),
    STUDENT_KD: Run(
        GQA, "distilled", GQA_700, 3e-5, 3e-4, init_offset=789, shuffle_offset=1770,
        save_every=80),
    FULL70_MEMBERS[0]: Run(GQA, "full", GQA_700, 2e-5, 3e-4, full_data=True,
                           init_offset=6601, shuffle_offset=6601, round_warmup=True),
    FULL70_MEMBERS[1]: Run(DECODER, "full", DECODER_100, 2e-5, 3e-4, full_data=True,
                           init_offset=6602, shuffle_offset=6602, round_warmup=True),
    FULL70_MEMBERS[2]: Run(GQA, "distilled", GQA_700, 3e-5, 3e-4, full_data=True,
                           init_offset=6603, shuffle_offset=6603, round_warmup=True),
}

# NV experiment machine sweep configuration
# Machine: 2u1g-b650-0079 — AMD Ryzen 7 7800X3D, NVIDIA GeForce RTX 5050 (8 GB GDDR7)
#
# HF model IDs:
#   GPT-2 variants  → openai-community/gpt2[-medium|-large|-xl]
#   Llama variants  → meta-llama/Llama-3.x-yB
#
# Seq-len ceilings are enforced at runtime via AutoConfig; the values here
# are the *requested* ranges (entries that exceed model max_position_embeddings
# are silently skipped by the sweep driver).
#
# Memory budget (BF16):
#   GPT-2 Small  124M  ~0.25 GB  → bs up to 16 is fine
#   GPT-2 Medium 355M  ~0.72 GB  → bs up to 16
#   GPT-2 Large  774M  ~1.55 GB  → bs up to 8
#   GPT-2 XL    1.5B   ~3.0  GB  → bs up to 4
#   Llama-3.2-1B 1.2B  ~2.5  GB  → bs up to 4
#   Llama-3.2-3B 3.2B  ~6.5  GB  → bs 1–2 (OOM guard active)
#   Llama-3.1-8B 8.0B  ~16   GB  → will OOM; kept for OOM-report coverage
#   Llama-3.1-70B 70B   >> 8 GB  → excluded (not runnable on this machine)

# ---------------------------------------------------------------------------
# Global timing defaults
# ---------------------------------------------------------------------------

INFERENCE_WARMUP = "10"
INFERENCE_RUNS   = "50"

DEBUG = False
if DEBUG:
    INFERENCE_WARMUP = "1"
    INFERENCE_RUNS   = "1"

PARAMS = {
    "compile_type":    "eager",
    "precision":       "bfloat16",
    "device":          "cuda",
    "inference_warmup": INFERENCE_WARMUP,
    "inference_runs":   INFERENCE_RUNS,
}

# ---------------------------------------------------------------------------
# Model registry
# Model keys are stable identifiers used by --models filter.
# ---------------------------------------------------------------------------

# GPT-2 context window is 1024 tokens; decode uses shorter seq_lens to keep
# KV-cache footprint manageable across the wider batch range.
_GPT2_SEQ_LENS_PREFILL = sorted([64, 128, 256, 512, 1024], reverse=True)
_GPT2_SEQ_LENS_DECODE  = sorted([64, 128, 256, 512], reverse=True)
_LLAMA_SEQ_LENS        = sorted([128, 256, 512, 1024, 2048, 4096, 8192], reverse=True)

NV_MODELS = {
    # ── GPT-2 family ──────────────────────────────────────────────────────
    "gpt2_small": {
        "hf_id":           "openai-community/gpt2",
        "params_b":        0.124,
        "seq_lens_prefill": _GPT2_SEQ_LENS_PREFILL,
        "seq_lens_decode":  _GPT2_SEQ_LENS_DECODE,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    "gpt2_medium": {
        "hf_id":           "openai-community/gpt2-medium",
        "params_b":        0.355,
        "seq_lens_prefill": _GPT2_SEQ_LENS_PREFILL,
        "seq_lens_decode":  _GPT2_SEQ_LENS_DECODE,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    "gpt2_large": {
        "hf_id":           "openai-community/gpt2-large",
        "params_b":        0.774,
        "seq_lens_prefill": _GPT2_SEQ_LENS_PREFILL,
        "seq_lens_decode":  _GPT2_SEQ_LENS_DECODE,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    "gpt2_xl": {
        "hf_id":           "openai-community/gpt2-xl",
        "params_b":        1.5,
        "seq_lens_prefill": _GPT2_SEQ_LENS_PREFILL,
        "seq_lens_decode":  _GPT2_SEQ_LENS_DECODE,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    # ── Llama family ──────────────────────────────────────────────────────
    "llama_3_2_1b": {
        "hf_id":           "meta-llama/Llama-3.2-1B",
        "params_b":        1.24,
        "seq_lens_prefill": _LLAMA_SEQ_LENS,
        "seq_lens_decode":  _LLAMA_SEQ_LENS,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    "llama_3_2_3b": {
        "hf_id":           "meta-llama/Llama-3.2-3B",
        "params_b":        3.21,
        "seq_lens_prefill": _LLAMA_SEQ_LENS,
        "seq_lens_decode":  _LLAMA_SEQ_LENS,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    "llama_3_1_8b": {
        "hf_id":           "meta-llama/Llama-3.1-8B",
        "params_b":        8.03,
        # Tight on 8 GB — kept for OOM coverage; OOM report will be written.
        "seq_lens_prefill": _LLAMA_SEQ_LENS,
        "seq_lens_decode":  _LLAMA_SEQ_LENS,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
    # Llama-3.1-70B: ~140 GB BF16; will OOM on this 8 GB machine.
    # Included so OOM reports are captured for completeness.
    "llama_3_1_70b": {
        "hf_id":           "meta-llama/Llama-3.1-70B",
        "params_b":        70.0,
        "seq_lens_prefill": _LLAMA_SEQ_LENS,
        "seq_lens_decode":  _LLAMA_SEQ_LENS,
        "batch_sizes_prefill": sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
        "batch_sizes_decode":  sorted([1, 2, 4, 8, 16, 32, 64, 128], reverse=True),
    },
}

# ---------------------------------------------------------------------------
# Sweep configs: prefill (max_new_tokens=1) and decode (max_new_tokens=10)
# These are derived from NV_MODELS so a single source of truth is maintained.
# ---------------------------------------------------------------------------

NV_PREF_CONFIG = {
    key: {
        "model_name":  m["hf_id"],
        "batch_sizes": m["batch_sizes_prefill"],
        "seq_lens":    m["seq_lens_prefill"],
        "max_new_toks": [1],
    }
    for key, m in NV_MODELS.items()
}

NV_DEC_CONFIG = {
    key: {
        "model_name":  m["hf_id"],
        "batch_sizes": m["batch_sizes_decode"],
        "seq_lens":    m["seq_lens_decode"],
        "max_new_toks": [10],
    }
    for key, m in NV_MODELS.items()
}

# Execution order (models run in this sequence)
NV_MODEL_ORDER = [
    "gpt2_small",
    "gpt2_medium",
    "gpt2_large",
    "gpt2_xl",
    "llama_3_2_1b",
    "llama_3_2_3b",
    "llama_3_1_8b",
    "llama_3_1_70b",
]

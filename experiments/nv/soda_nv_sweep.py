#!/usr/bin/env python3
"""
NV experiment machine — SODA model sweep driver
================================================
Iterates over the model list defined in experiments/nv/config.py and runs
SODA for every (batch_size, seq_len) combination.

Optional CLI overrides (all have sensible per-model defaults in config.py):
  --mode         prefill | decode        (default: prefill)
  --models       comma-separated keys    (default: all, in NV_MODEL_ORDER)
  --batch-sizes  comma-separated ints    (override per-model defaults)
  --seq-lens     comma-separated ints    (override per-model defaults)
  --warmup       int                     (default from config)
  --runs         int                     (default from config)

Examples:
  python experiments/nv/soda_nv_sweep.py
  python experiments/nv/soda_nv_sweep.py --mode decode
  python experiments/nv/soda_nv_sweep.py --mode prefill --models gpt2_small,gpt2_medium
  python experiments/nv/soda_nv_sweep.py --batch-sizes 1,2 --seq-lens 128,256
"""

import argparse
import csv
import gc
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Make repo importable when called directly (not via soda-cli)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from soda import ModelTracer, SodaAnalyzer
from soda.common import utils
from experiments.sweep.summarize_soda_sweep import summarize as summarize_soda_sweep
from experiments.nv.config import (
    PARAMS,
    NV_PREF_CONFIG,
    NV_DEC_CONFIG,
    NV_MODEL_ORDER,
    NV_MODELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_env_loaded() -> None:
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def get_gpu_suffix() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    for token, tag in [
        ("RTX 5090", "RTX5090"),
        ("RTX 5080", "RTX5080"),
        ("RTX 5070", "RTX5070"),
        ("RTX 5060", "RTX5060"),
        ("RTX 5050", "RTX5050"),
        ("H200",     "H200"),
        ("H100",     "H100"),
        ("A100",     "A100"),
        ("V100",     "V100"),
        ("T4",       "T4"),
        ("L40S",     "L40S"),
        ("L4",       "L4"),
        ("4090",     "RTX4090"),
        ("4080",     "RTX4080"),
    ]:
        if token in name:
            return tag
    return "gpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SODA NV machine sweep — iterate models one after another.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        default="prefill",
        help="prefill: max_new_tokens=1. decode: max_new_tokens=10.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model keys to run (e.g. gpt2_small,llama_3_2_1b). "
            f"Available: {', '.join(NV_MODEL_ORDER)}. "
            "Runs all models in NV_MODEL_ORDER when omitted."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        dest="batch_sizes",
        default=None,
        help=(
            "Comma-separated batch sizes to override per-model defaults "
            "(e.g. '1,2,4'). Applied to every selected model."
        ),
    )
    parser.add_argument(
        "--seq-lens",
        dest="seq_lens",
        default=None,
        help=(
            "Comma-separated sequence lengths to override per-model defaults "
            "(e.g. '128,256,512'). Entries that exceed a model's "
            "max_position_embeddings are silently skipped."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup iterations (overrides config default).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of profiling runs (overrides config default).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Flat CSV helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "model_name",
    "seq_len",
    "batch_size",
    "mode",
    "max_new_tokens",
    "inference_ms",
    "throughput_tok_s",
    "device_util_pct",
    "device_active_time_ms",
    "tklqt_us",
    "num_kernel_launches",
    "status",
]


def _extract_metrics(report_path: Path) -> Dict[str, Any]:
    """Read a report.json and return the flat metrics dict for the CSV row."""
    try:
        data = json.loads(report_path.read_text())
    except Exception:
        return {}

    perf = data.get("performance_metrics") or data.get("metrics") or {}

    # inference_time_ms
    inf_ms = perf.get("inference_time_ms")
    status = "oom" if inf_ms == "OOM" else "ok"
    if inf_ms == "OOM":
        inf_ms = None
    else:
        try:
            inf_ms = float(inf_ms) if inf_ms is not None else None
        except (ValueError, TypeError):
            inf_ms = None

    # throughput
    thr = perf.get("inference_throughput", {})
    throughput = thr.get("throughput_tok_s") if isinstance(thr, dict) else None

    # gpu utilization
    gpu_util = perf.get("gpu_utilization_percent")

    # gpu active time (true_gpu_busy_time_us → ms)
    gpu_busy_us = perf.get("true_gpu_busy_time_us") or perf.get("gpu_busy_time_us")
    gpu_active_ms = round(gpu_busy_us / 1000.0, 3) if gpu_busy_us else None

    # tklqt
    tklqt_data = perf.get("tklqt", {})
    tklqt_us = tklqt_data.get("total") if isinstance(tklqt_data, dict) else None

    # kernel launches
    frag = perf.get("fragmentation_metrics", {})
    num_launches = frag.get("total_kernel_launches") if isinstance(frag, dict) else None

    return {
        "inference_ms":          round(inf_ms, 3) if inf_ms is not None else None,
        "throughput_tok_s":      round(float(throughput), 2) if throughput is not None else None,
        "device_util_pct":       round(float(gpu_util), 2) if gpu_util is not None else None,
        "device_active_time_ms": gpu_active_ms,
        "tklqt_us":              round(float(tklqt_us), 2) if tklqt_us is not None else None,
        "num_kernel_launches":   int(num_launches) if num_launches is not None else None,
        "status":                status,
    }


def _open_csv(csv_path: Path) -> csv.DictWriter:
    """Open (or create) the sweep CSV, write header if new, return writer."""
    is_new = not csv_path.exists()
    fh = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return writer, fh


def _append_csv_row(
    writer: csv.DictWriter,
    fh,
    model_name: str,
    seq_len: int,
    batch_size: int,
    mode: str,
    max_new_tokens: int,
    metrics: Dict[str, Any],
) -> None:
    row = {
        "model_name":   model_name,
        "seq_len":      seq_len,
        "batch_size":   batch_size,
        "mode":         mode,
        "max_new_tokens": max_new_tokens,
        **metrics,
    }
    writer.writerow(row)
    fh.flush()   # flush after every row so data survives partial sweeps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_env_loaded()
    args = parse_args()

    gpu_suffix    = get_gpu_suffix()
    compile_type  = PARAMS["compile_type"]
    precision     = PARAMS["precision"]
    device        = PARAMS["device"]
    warmup        = str(args.warmup) if args.warmup is not None else PARAMS["inference_warmup"]
    runs          = str(args.runs)   if args.runs   is not None else PARAMS["inference_runs"]

    # Select sweep config by mode
    sweep_config = NV_PREF_CONFIG if args.mode == "prefill" else NV_DEC_CONFIG

    # Build ordered model list
    model_order = NV_MODEL_ORDER
    if args.models:
        requested = [k.strip() for k in args.models.split(",")]
        invalid = [k for k in requested if k not in sweep_config]
        if invalid:
            print(
                f"Error: unknown model key(s): {invalid}\n"
                f"Available: {NV_MODEL_ORDER}",
                file=sys.stderr,
            )
            sys.exit(1)
        model_order = [k for k in model_order if k in requested]

    # CLI batch/seq overrides (apply to every model)
    cli_batch_sizes = (
        sorted([int(x.strip()) for x in args.batch_sizes.split(",")], reverse=True)
        if args.batch_sizes else None
    )
    cli_seq_lens = (
        sorted([int(x.strip()) for x in args.seq_lens.split(",")], reverse=True)
        if args.seq_lens else None
    )

    print(f"[NV sweep] mode={args.mode}  gpu={gpu_suffix}  precision={precision}")
    print(f"[NV sweep] warmup={warmup}  runs={runs}")
    print(f"[NV sweep] models ({len(model_order)}): {model_order}")
    if cli_batch_sizes:
        print(f"[NV sweep] batch_sizes override: {cli_batch_sizes}")
    if cli_seq_lens:
        print(f"[NV sweep] seq_lens override:    {cli_seq_lens}")
    print()

    # Open the flat sweep CSV (append-safe; one row per run point)
    output_base = Path(os.environ.get("SODA_OUTPUT", "output"))
    output_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_base / f"nv_sweep_{args.mode}_{gpu_suffix}_{ts}.csv"
    csv_writer, csv_fh = _open_csv(csv_path)
    print(f"[NV sweep] Writing results to: {csv_path}\n")

    sweep_roots: list[Path] = []

    for model_key in model_order:
        cfg        = sweep_config[model_key]
        model_id   = cfg["model_name"]
        batch_sizes = cli_batch_sizes if cli_batch_sizes is not None else cfg["batch_sizes"]
        seq_lens    = cli_seq_lens    if cli_seq_lens    is not None else cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]

        # Fetch model's positional embedding limit
        try:
            model_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            max_pos = getattr(
                model_cfg, "max_position_embeddings",
                getattr(model_cfg, "n_positions",
                getattr(model_cfg, "n_ctx", float("inf")))
            )
        except Exception as exc:
            print(f"Warning: could not load config for {model_id}: {exc}. Skipping seq_len validation.")
            max_pos = float("inf")

        max_tok_str = f"mt{max_new_toks[0]}"
        sweep_root = (
            Path(os.environ.get("SODA_OUTPUT", "output"))
            / f"{model_id.replace('/', '_')}_{compile_type}_{precision}_{max_tok_str}_{gpu_suffix}"
        )
        sweep_roots.append(sweep_root)

        params_b = NV_MODELS[model_key]["params_b"]
        print(
            f"\n{'='*70}\n"
            f"  Model : {model_id}  ({params_b}B params)\n"
            f"  Key   : {model_key}\n"
            f"  Mode  : {args.mode}  max_new_tokens={max_new_toks[0]}\n"
            f"  Batches: {batch_sizes}   seq_lens: {seq_lens}\n"
            f"  Output: {sweep_root}\n"
            f"{'='*70}"
        )

        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            if sl > max_pos:
                print(f"  skip  bs={bs} sl={sl}: exceeds max_position_embeddings ({max_pos})")
                continue

            print(f"\n  > bs={bs}  sl={sl}  max_new_tokens={max_new_tokens}")
            exp_name = utils.generate_experiment_name(
                model_id, compile_type, precision, bs, sl, max_new_tokens
            )
            cli_args = [
                "--model",          model_id,
                "--output-dir",     str(sweep_root),
                "--batch-size",     str(bs),
                "--seq-len",        str(sl),
                "--max-new-tokens", str(max_new_tokens),
                "--precision",      precision,
                "--compile-type",   compile_type,
                "--device",         device,
                "--warmup",         warmup,
                "--runs",           runs,
            ]
            soda_args = utils.parse_and_validate_args(cli_args)

            tracer   = None
            analyzer = None
            try:
                tracer   = ModelTracer(args=soda_args)
                tracer.run()
                analyzer = SodaAnalyzer(tracer=tracer, args=soda_args)
                report_path = analyzer.run()
                print(f"    saved → {report_path}")

                # Append metrics row to flat CSV
                metrics = _extract_metrics(Path(report_path))
                _append_csv_row(
                    csv_writer, csv_fh,
                    model_name=model_id, seq_len=sl, batch_size=bs,
                    mode=args.mode, max_new_tokens=max_new_tokens,
                    metrics=metrics,
                )

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"    OOM   bs={bs} sl={sl} — writing OOM report.", file=sys.stderr)
                    run_dir = sweep_root / exp_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    oom_report = {
                        "metadata": {
                            "model_name": model_id,
                            "timestamp": datetime.now().isoformat(),
                            "config": {
                                "batch_size": bs,
                                "seq_len": sl,
                                "max_new_tokens": max_new_tokens,
                                "precision": precision,
                                "compile_type": compile_type,
                                "device": device,
                                "gpu_name": gpu_suffix,
                            },
                        },
                        "performance_metrics": {
                            "inference_time_ms": "OOM",
                            "error": str(exc),
                            "memory_metrics": {"peak_memory_allocated_mb": "OOM"},
                        },
                    }
                    with open(run_dir / "report.json", "w") as f:
                        json.dump(oom_report, f, indent=4)

                    # Append OOM row to flat CSV
                    _append_csv_row(
                        csv_writer, csv_fh,
                        model_name=model_id, seq_len=sl, batch_size=bs,
                        mode=args.mode, max_new_tokens=max_new_tokens,
                        metrics={"status": "oom"},
                    )
                    continue
                raise

            finally:
                del analyzer
                del tracer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Per-model summary
        if sweep_root.exists():
            print(f"\n  Generating summary → {sweep_root}/summary")
            try:
                summarize_soda_sweep(sweep_root, gpu_name_override=gpu_suffix)
            except Exception as exc:
                print(f"  Warning: summary failed for {sweep_root}: {exc}")

    csv_fh.close()

    print(f"\n{'='*70}")
    print(f"[NV sweep] Complete. {len(sweep_roots)} model root(s):")
    for r in sweep_roots:
        print(f"  {r}")
    print(f"\n[NV sweep] Flat CSV: {csv_path}")


if __name__ == "__main__":
    main()

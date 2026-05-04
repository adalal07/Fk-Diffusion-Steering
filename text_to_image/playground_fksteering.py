import argparse
import json
import os
import re
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FKD_DIFFUSERS_DIR = os.path.join(SCRIPT_DIR, "fkd_diffusers")
if FKD_DIFFUSERS_DIR not in sys.path:
    sys.path.append(FKD_DIFFUSERS_DIR)

from launch_eval_runs import do_eval
from fks_utils import get_model, plot_fkd_reward_trace, plot_resampling_diagnostics


COMIC_BOOK_STYLE_QUERY_TEXT = (
    "You are a strict visual style judge.\n"
    "Task: score ONLY comic-book stylization of this image, ignoring content quality.\n"
    "Return one decimal number in [-50.0, 50.0] and nothing else.\n\n"
    "Scoring rubric:\n"
    "- Start at 0.0.\n"
    "- Add points for comic cues:\n"
    "  + strong black ink outlines\n"
    "  + cel-shading / flat color blocks\n"
    "  + halftone / screentone dot texture\n"
    "  + posterized high-contrast stylization\n"
    "  + panel/print illustration look\n"
    "- Subtract points for photorealism cues:\n"
    "  - realistic camera blur / bokeh / depth-of-field\n"
    "  - natural skin/fur texture and photographic lighting\n"
    "  - smooth photo gradients without inking\n"
    "  - DSLR-like realism\n\n"
    "Calibration anchors:\n"
    "-50.0 = fully photorealistic photo\n"
    "  0.0 = mixed or ambiguous style\n"
    "+50.0 = unmistakable comic-book rendering\n\n"
    "Output format: one number like -12.4"
)

PIXEL_ART_STYLE_QUERY_TEXT = (
    "You are a strict style grader.\n"
    "Score ONLY pixel-art style strength of this image.\n"
    "Return exactly one decimal number in [-50.0, 50.0]. No words.\n\n"
    "Positive cues:\n"
    "+ visible square pixels\n"
    "+ limited palette\n"
    "+ hard edges\n"
    "+ dithering/checkerboard shading\n"
    "+ retro sprite/game look\n\n"
    "Negative cues:\n"
    "- photorealistic texture/lighting\n"
    "- smooth anti-aliased edges\n"
    "- soft gradients / blur / bokeh\n"
    "- painterly or sketch/ink look\n\n"
    "Anchors:\n"
    "-50.0 = fully non-pixel/photoreal\n"
    "  0.0 = ambiguous mix\n"
    "+50.0 = unmistakable pixel-art\n\n"
    "Output example: -17.4"
)

WATERCOLOR_STYLE_QUERY_TEXT = (
    "You are a strict visual style grader.\n"
    "Score ONLY watercolor-painting style strength of this image.\n"
    "Return exactly one decimal number in [-50.0, 50.0]. No words.\n\n"
    "Positive cues:\n"
    "+ translucent pigment washes\n"
    "+ soft color bleeding / wet-on-wet transitions\n"
    "+ granulation / paper texture hints\n"
    "+ loose brush edges and painterly softness\n"
    "+ limited line work with paint-driven form\n\n"
    "Negative cues:\n"
    "- photorealistic camera look (bokeh, lens blur, DSLR lighting)\n"
    "- crisp vector/cartoon outlines\n"
    "- pixel-art blockiness or dithering\n"
    "- heavy comic inking / cel-shading\n"
    "- smooth digital airbrush without watercolor texture\n\n"
    "Anchors:\n"
    "-50.0 = fully non-watercolor / photoreal image\n"
    "  0.0 = ambiguous or mixed style\n"
    "+50.0 = unmistakable watercolor painting\n\n"
    "Output example: 18.7"
)


STYLE_QUERY_MAP = {
    "watercolor": WATERCOLOR_STYLE_QUERY_TEXT,
    "pixel_art": PIXEL_ART_STYLE_QUERY_TEXT,
    "pixel-art": PIXEL_ART_STYLE_QUERY_TEXT,
    "comic_book": COMIC_BOOK_STYLE_QUERY_TEXT,
}
STYLE_AWARE_REWARDS = {"Qwen3VLStyle", "MoonDreamStyle"}


def slugify_text(text: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not cleaned:
        cleaned = "prompt"
    return cleaned[:max_len].rstrip("-")


def _optional_float_ge_zero(s) -> float:
    """argparse ``type`` that treats ``''``, ``none``, ``null`` as 0.0 (common when env vars expand empty)."""
    if s is None:
        return 0.0
    t = str(s).strip()
    if t == "" or t.lower() in ("none", "null"):
        return 0.0
    return float(t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch FK steering runs with repeat control.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--model-name", type=str, default="stable-diffusion-xl")
    parser.add_argument(
        "--vlm-reward-model",
        "--qwen-model-name",
        dest="vlm_reward_model",
        type=str,
        default="",
        help=(
            "HF model id for vision-language rewards (VLMOCRScore/V2, VLMColorBinding, Qwen3VLSpatial, Qwen3VLStyle). "
            "If empty, each reward keeps its built-in default (e.g. 30B for VLM OCR from make_vlm_reward_config). "
            "``--qwen-model-name`` is a backward-compatible alias. "
            "For VLMOCRScoreV2 only, ``--vlm-model-name`` overrides this when set."
        ),
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default="",
        help=(
            "HF model id used **only** for VLMOCRScoreV2 (e.g. ``google/gemma-4-26B-A4B-it``, ``Qwen/Qwen3-VL-8B-Instruct``). "
            "Overrides ``--vlm-reward-model`` / ``--qwen-model-name`` for that reward so you can benchmark VLMs without changing other metrics. "
            "Empty = use the global VLM model flag or the reward default."
        ),
    )
    parser.add_argument(
        "--vlm-hf-model-class",
        type=str,
        default="auto",
        choices=("auto", "image_text_to_text", "vision2seq"),
        help=(
            "Which Transformers auto class to load for ``--vlm-model-name`` / VLMOCRScoreV2. "
            "``auto`` tries ``AutoModelForImageTextToText`` then ``AutoModelForVision2Seq``. "
            "Gemma 4 / Qwen3-VL typically need ``auto`` or ``image_text_to_text``."
        ),
    )
    parser.add_argument(
        "--vlm-ocr-log-include-model-slug",
        action="store_true",
        help=(
            "If set, VLMOCRScoreV2 JSONL logs are written to "
            "``vlm_ocr_v2__<slug>__intermediate_logs.jsonl`` under ``--output-dir`` "
            "(slug derived from the effective HF model id) so parallel benchmarks do not overwrite logs."
        ),
    )
    parser.add_argument(
        "--vlm-hf-device-map",
        "--qwen-hf-device-map",
        dest="vlm_hf_device_map",
        type=str,
        default="",
        help=(
            "Accelerate ``device_map`` for HF VLM weights (e.g. balanced, auto, sequential). "
            "Use ``none`` to disable sharding. Empty = use reward defaults (VLM OCR defaults to balanced). "
            "``--qwen-hf-device-map`` is an alias. "
            "See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference"
        ),
    )
    parser.add_argument(
        "--vlm-max-memory-per-gpu-gib",
        "--qwen-max-memory-per-gpu-gib",
        dest="vlm_max_memory_per_gpu_gib",
        type=_optional_float_ge_zero,
        default=0.0,
        help=(
            "If >0, set per-GPU ``max_memory`` for each visible CUDA device (e.g. 18 on 24GB cards). "
            "Helps accelerate split HF VLM weights across GPUs. Incompatible with ``--vlm-tp-plan``. "
            "``--qwen-max-memory-per-gpu-gib`` is an alias."
        ),
    )
    parser.add_argument(
        "--vlm-tp-plan",
        "--qwen-tp-plan",
        dest="vlm_tp_plan",
        type=str,
        default="",
        help=(
            "Native Transformers tensor parallelism for the VLM, e.g. ``auto`` (requires model ``base_model_tp_plan``). "
            "Clears device_map/max_memory per HF rules. Often used with: "
            "``torchrun --nproc-per-node=4 playground_fksteering.py ...``. Empty = off. "
            "``--qwen-tp-plan`` is an alias."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Default torch device string (used for diffusion if --diffusion-device is unset).",
    )
    parser.add_argument(
        "--diffusion-device",
        type=str,
        default="",
        help=(
            "Device for the SD/SDXL pipeline only, e.g. ``cuda:0``. Use with ``--vlm-only-cuda-ids`` so the "
            "VLM does not shard onto the same GPU as UNet (common OOM fix when both fit poorly on one card)."
        ),
    )
    parser.add_argument(
        "--vlm-only-cuda-ids",
        "--qwen-only-cuda-ids",
        dest="vlm_only_cuda_ids",
        type=str,
        default="",
        help=(
            "Comma-separated **logical** CUDA indices (after ``CUDA_VISIBLE_DEVICES``), each in "
            "``0 .. torch.cuda.device_count()-1`` — e.g. with 4 visible GPUs use ``1,2,3`` to reserve "
            "logical GPU 0 for SDXL, not ``1,2,3,4`` (there is no index 4). Builds HF ``max_memory`` with "
            "0GiB on other devices for the VLM. Incompatible with ``--vlm-tp-plan``. "
            "``--qwen-only-cuda-ids`` is an alias."
        ),
    )
    parser.add_argument(
        "--enable-sdxl-memory-savers",
        action="store_true",
        help="Enable VAE slicing + attention slicing (and UNet forward chunking when available) to lower SDXL peak VRAM.",
    )
    parser.add_argument(
        "--dist-timeout-sec",
        type=int,
        default=86400,
        help=(
            "``torch.distributed`` / NCCL process-group timeout (seconds) for ``torchrun`` multi-rank runs. "
            "Ranks 1..N-1 wait at barriers while rank 0 runs long VLM steps inside ``pipeline()``; the default "
            "PyTorch NCCL watchdog (~600s) is too short. Ignored when ``WORLD_SIZE<=1``. "
            "Override with env ``FKD_DIST_TIMEOUT_SEC`` if set."
        ),
    )

    parser.add_argument("--prompt-path", type=str, default="./prompt_files/benchmark_ir.json")
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all prompts in file.")
    parser.add_argument(
        "--inline-prompt",
        type=str,
        default="",
        help="Single prompt override. If set, prompt file is ignored.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Number of runs per prompt per reward combo.",
    )

    parser.add_argument(
        "--metrics-to-compute",
        type=str,
        default="",
        help="Metrics to compute, separated by '#'. If empty, defaults to the guidance reward(s).",
    )
    parser.add_argument(
        "--guidance-reward-fns",
        type=str,
        default="Qwen3VLStyle",
        help="Comma-separated reward functions to run as separate combos.",
    )
    parser.add_argument(
        "--style-targets",
        type=str,
        default="",
        help="Comma-separated style targets (used for style-aware rewards like Qwen3VLStyle).",
    )
    parser.add_argument("--success-threshold", type=float, default=0.0)
    parser.add_argument(
        "--save-all-particles",
        action="store_true",
        help="If enabled, save one panel image for all particles instead of best-only.",
    )

    parser.add_argument("--lmbda", type=float, default=4.0)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--adaptive-resampling", action="store_true")
    parser.add_argument("--resample-frequency", type=int, default=10)
    parser.add_argument("--time-steps", type=int, default=100)
    parser.add_argument("--potential-type", type=str, default="max")
    parser.add_argument("--resampling-t-start", type=int, default=20)
    parser.add_argument("--resampling-t-end", type=int, default=99)
    parser.add_argument("--use-smc", action="store_true", default=True)
    parser.add_argument("--disable-smc", action="store_true")
    parser.add_argument(
        "--record-baseline-trace",
        action="store_true",
        help=(
            "When SMC is disabled, still decode x0 at resampling schedule, score with the "
            "guidance reward, save per-step intermediates, and fill reward_history (no resampling)."
        ),
    )
    parser.add_argument("--include-terminal-resample", action="store_true", default=True)
    parser.add_argument(
        "--no-diffusion-tqdm",
        action="store_true",
        help="Hide per-step diffusers tqdm (keeps one outer 'Generations' bar clearer with VLMs).",
    )

    return parser.parse_args()


def load_prompts(prompt_path: str, inline_prompt: str, max_prompts: int) -> List[Dict]:
    if inline_prompt:
        prompts = [{"id": "inline-0000", "prompt": inline_prompt}]
    else:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    return prompts


def make_qwen_reward_config(style_target: str) -> Dict:
    query_text = STYLE_QUERY_MAP.get(style_target, WATERCOLOR_STYLE_QUERY_TEXT)
    return {
        "style_target": style_target,
        "qwen_model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "rating_min": -50,
        "rating_max": 50,
        "qwen_force_reload": False,
        "qwen_debug_print": True,
        "warn_parse_failures": True,
        "qwen_log_to_output": True,
        "qwen_retry_on_parse_fail": True,
        "qwen_retry_max_new_tokens": 8,
        "include_prompt_context": False,
        "query_text": query_text,
    }

def make_qwen3vl_spatial_reward_config(output_dir: str) -> Dict:
    """Minimal reward_config for Qwen3VLSpatial (no OCR/color-binding-only keys)."""
    return {
        "vlm_log_to_output": True,
        "vlm_log_path": os.path.join(output_dir, "vlm_spatial_awareness_logs.jsonl"),
        "qwen_model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        # "auto" usually fills cuda:0 only when the model fits one GPU; "balanced" splits across visible devices.
        "qwen_hf_device_map": "balanced",
        "qwen_hf_dtype": None,
        "qwen_hf_offload_folder": "output/hf_offload",
        "qwen_hf_low_cpu_mem_usage": True,
        "qwen_force_reload": False,
        "qwen_query_max_tokens": 768,
        "qwen_do_sample": False,
        "qwen_temperature": 0.0,
        "qwen_top_p": 1.0,
        "include_prompt_context": True,
        "qwen_debug_print": False,
        "warn_parse_failures": True,
        "weight_existence": 0.3,
        "weight_relation": 0.5,
        "weight_geometric": 0.2,
        "reward_min": -1.0,
        "reward_max": 1.0,
    }


def make_vlm_reward_config(output_dir: str) -> Dict:
    return {
        "reward_key": "reward",

        # ----- VLM logging -----
        # Full VLM raw/parsed outputs are dumped here (jsonl, one record per image call).
        "vlm_log_to_output": True,
        "vlm_log_path": os.path.join(output_dir, "vlm_ocr_intermediate_logs.jsonl"),
        # "vlm_log_enabled": False,         # turn off json saving
        # "vlm_numeric_only_output": True,

        # ----- Qwen3-VL model/runtime -----
        "qwen_model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        # "auto" often stacks weights on cuda:0; "balanced" spreads layers across visible GPUs.
        "qwen_hf_device_map": "balanced",
        "qwen_hf_dtype": None,
        "qwen_hf_offload_folder": "output/hf_offload",
        "qwen_hf_low_cpu_mem_usage": True,
        # Native TP (optional): "auto" — incompatible with qwen_hf_device_map; see --vlm-tp-plan.
        "qwen_force_reload": False,

        # ----- Generation controls (keep deterministic for stable rewards) -----
        "qwen_query_max_tokens": 320,
        "qwen_do_sample": False,
        "qwen_temperature": 0.0,
        "qwen_top_p": 1.0,

        # ----- Prompting/debug -----
        "include_prompt_context": True,
        "qwen_debug_print": False,
        "warn_parse_failures": True,

        # VLMColorBinding: subtract penalty × extraneous_content_penalty from mean constraint score
        "extra_items_penalty_weight": 1.0,
        "reward_min": -1.0,
        "reward_max": 1.0,
    }

def make_reward_config_for_reward(
    *, reward_name: str, style_target: str | None, output_dir: str
) -> Dict:
    if reward_name in STYLE_AWARE_REWARDS:
        target = style_target or "watercolor"
        cfg = make_qwen_reward_config(target)
        cfg["qwen_log_path"] = os.path.join(output_dir, "qwen_intermediate_logs.jsonl")
        return cfg
    if reward_name == "VLMOCRScore":
        return make_vlm_reward_config(output_dir)
    if reward_name == "VLMOCRScoreV2":
        cfg = make_vlm_reward_config(output_dir)
        cfg["vlm_log_path"] = os.path.join(output_dir, "vlm_ocr_v2_intermediate_logs.jsonl")
        return cfg
    if reward_name == "VLMColorBinding":
        cfg = make_vlm_reward_config(output_dir)
        cfg["vlm_log_path"] = os.path.join(output_dir, "vlm_color_binding_logs.jsonl")
        return cfg
    if reward_name == "Qwen3VLSpatial":
        return make_qwen3vl_spatial_reward_config(output_dir)
    return {}


def apply_vlm_parallel_cli_overrides(
    reward_config: Dict, args: argparse.Namespace
) -> None:
    """Mutates ``reward_config`` with multi-GPU / tensor-parallel CLI flags for HF VLMs."""
    dm = (getattr(args, "vlm_hf_device_map", None) or "").strip()
    if dm.lower() in ("none", "null"):
        reward_config["qwen_hf_device_map"] = None
    elif dm:
        reward_config["qwen_hf_device_map"] = dm

    tp = (getattr(args, "vlm_tp_plan", None) or "").strip()
    if tp:
        reward_config["qwen_tp_plan"] = tp
        reward_config["qwen_hf_device_map"] = None
        reward_config.pop("qwen_hf_max_memory", None)

    gib = float(getattr(args, "vlm_max_memory_per_gpu_gib", 0.0) or 0.0)
    if gib > 0.0:
        if tp:
            raise ValueError(
                "Use either --vlm-tp-plan (--qwen-tp-plan) or "
                "--vlm-max-memory-per-gpu-gib (--qwen-max-memory-per-gpu-gib), not both."
            )
        n = torch.cuda.device_count()
        if n <= 0:
            raise ValueError(
                "--vlm-max-memory-per-gpu-gib requires at least one visible CUDA device"
            )
        reward_config["qwen_hf_max_memory"] = {
            i: f"{gib}GiB" for i in range(n)
        }


def apply_vlm_only_cuda_ids_budget(reward_config: Dict, args: argparse.Namespace) -> None:
    """
    Restrict HuggingFace accelerate ``max_memory`` so VLM weights only use selected GPUs
    (typically GPU 0 reserved for SDXL).
    """
    raw = (getattr(args, "vlm_only_cuda_ids", None) or "").strip()
    if not raw:
        return
    if (getattr(args, "vlm_tp_plan", None) or "").strip():
        raise ValueError(
            "--vlm-only-cuda-ids (--qwen-only-cuda-ids) cannot be combined with "
            "--vlm-tp-plan (--qwen-tp-plan)."
        )
    ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    n = torch.cuda.device_count()
    if n <= 0:
        raise ValueError("--vlm-only-cuda-ids requires at least one visible CUDA device.")
    for i in ids:
        if i < 0 or i >= n:
            raise ValueError(
                f"--vlm-only-cuda-ids: index {i} is out of range for "
                f"torch.cuda.device_count()={n} (valid indices are 0..{n - 1}). "
                f"Parsed ids={ids!r}, raw={raw!r}."
            )
    gib = float(getattr(args, "vlm_max_memory_per_gpu_gib", 0.0) or 0.0)
    cap = f"{gib}GiB" if gib > 0.0 else "26GiB"
    max_mem = {j: "0GiB" for j in range(n)}
    for j in ids:
        max_mem[j] = cap
    reward_config["qwen_hf_max_memory"] = max_mem
    reward_config["qwen_hf_device_map"] = "auto"


# Backward-compatible names (older scripts / notes may still reference these).
apply_qwen_parallel_cli_overrides = apply_vlm_parallel_cli_overrides
apply_qwen_vlm_cuda_id_budget = apply_vlm_only_cuda_ids_budget


def _maybe_enable_sdxl_memory_savers(pipeline, args: argparse.Namespace) -> None:
    if not getattr(args, "enable_sdxl_memory_savers", False):
        return
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
    unet = getattr(pipeline, "unet", None)
    if unet is not None and hasattr(unet, "enable_forward_chunking"):
        try:
            unet.enable_forward_chunking(chunk_size=1, dim=0)
        except TypeError:
            try:
                unet.enable_forward_chunking(chunk_size=1)
            except Exception:
                pass


def _dist_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_is_primary() -> bool:
    return _dist_rank() == 0


def _maybe_init_distributed_for_playground(args: argparse.Namespace) -> None:
    """Under ``torchrun``, set one CUDA device per process and init NCCL."""
    if _dist_world_size() <= 1:
        return
    if torch.distributed.is_initialized():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    env_t = os.environ.get("FKD_DIST_TIMEOUT_SEC")
    if env_t is not None and str(env_t).strip():
        timeout_sec = int(env_t)
    else:
        timeout_sec = int(getattr(args, "dist_timeout_sec", 86400) or 86400)
    timeout_sec = max(timeout_sec, 600)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=timeout_sec),
    )


def _dist_barrier() -> None:
    if _dist_is_initialized():
        torch.distributed.barrier()


def _broadcast_obj_from_rank0(obj: Any) -> Any:
    if not _dist_is_initialized() or _dist_world_size() <= 1:
        return obj
    payload: List[Any] = [obj if _dist_is_primary() else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _broadcast_run_bundle_from_rank0(
    images: Any, run_status: str, error_msg: Optional[str], error_tb: Optional[str]
) -> Tuple[Any, str, Optional[str], Optional[str]]:
    if not _dist_is_initialized() or _dist_world_size() <= 1:
        return images, run_status, error_msg, error_tb
    bundle: List[Any] = (
        [images, run_status, error_msg, error_tb]
        if _dist_is_primary()
        else [None, None, None, None]
    )
    torch.distributed.broadcast_object_list(bundle, src=0)
    return bundle[0], bundle[1], bundle[2], bundle[3]


def _mute_reward_file_logging_for_non_primary(reward_config: Dict) -> None:
    """Avoid corrupt interleaved jsonl when every rank scores with TP."""
    if _dist_world_size() <= 1 or _dist_is_primary():
        return
    reward_config["vlm_log_enabled"] = False
    reward_config["qwen_log_to_output"] = False


def main() -> None:
    args = parse_args()
    if args.disable_smc:
        args.use_smc = False

    _maybe_init_distributed_for_playground(args)
    is_primary = _dist_is_primary()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    prompts = load_prompts(args.prompt_path, args.inline_prompt, args.max_prompts)
    if not prompts:
        raise ValueError("No prompts provided. Check --prompt-path / --inline-prompt / --max-prompts.")

    guidance_rewards = [x.strip() for x in args.guidance_reward_fns.split(",") if x.strip()]
    style_targets = [x.strip() for x in args.style_targets.split(",") if x.strip()]
    if not guidance_rewards:
        raise ValueError("--guidance-reward-fns must be non-empty.")

    reward_combos = []
    for reward_name in guidance_rewards:
        if reward_name in STYLE_AWARE_REWARDS:
            targets = style_targets or ["watercolor"]
            for target in targets:
                reward_combos.append((reward_name, target))
        else:
            reward_combos.append((reward_name, None))

    output_dir: str
    if is_primary:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output_dir, cur_time)
    else:
        output_dir = ""
    output_dir = _broadcast_obj_from_rank0(output_dir)
    images_dir = os.path.join(output_dir, "images")
    plots_dir = os.path.join(output_dir, "plots")
    inter_dir = os.path.join(output_dir, "intermediates")

    if is_primary:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(inter_dir, exist_ok=True)
        with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
    _dist_barrier()

    pipeline = None
    if is_primary:
        diff_dev = (getattr(args, "diffusion_device", None) or "").strip() or args.device
        pipeline = get_model(args.model_name).to(diff_dev)
        _maybe_enable_sdxl_memory_savers(pipeline, args)
        if getattr(args, "no_diffusion_tqdm", False):
            pipeline.set_progress_bar_config(disable=True)

    if args.metrics_to_compute.strip():
        metrics_to_compute = [x.strip() for x in args.metrics_to_compute.split("#") if x.strip()]
        for reward_name in guidance_rewards:
            if reward_name not in metrics_to_compute:
                metrics_to_compute.append(reward_name)
    else:
        metrics_to_compute = list(dict.fromkeys(guidance_rewards))

    run_plan = {
        "output_dir": output_dir,
        "model_name": args.model_name,
        "device": args.device,
        "num_prompts": len(prompts),
        "repeat_count": args.repeat_count,
        "success_threshold": args.success_threshold,
        "guidance_reward_fns": guidance_rewards,
        "reward_combos": [
            {"guidance_reward_fn": reward_name, "style_target": style_target}
            for reward_name, style_target in reward_combos
        ],
        "metrics_to_compute": metrics_to_compute,
        "fkd_settings": {
            "lmbda": args.lmbda,
            "num_particles": args.num_particles,
            "adaptive_resampling": args.adaptive_resampling,
            "resample_frequency": args.resample_frequency,
            "time_steps": args.time_steps,
            "potential_type": args.potential_type,
            "resampling_t_start": args.resampling_t_start,
            "resampling_t_end": args.resampling_t_end,
            "use_smc": args.use_smc,
            "record_baseline_trace": bool(args.record_baseline_trace),
            "include_terminal_resample": args.include_terminal_resample,
        },
    }
    if is_primary:
        print("Resolved run config:")
        print(json.dumps(run_plan, indent=2))

    combo_stats = {}
    score_path = os.path.join(output_dir, "scores.jsonl")
    err_log_path = os.path.join(output_dir, "run_errors.log")
    logged_pipeline_error_msgs: set[str] = set()

    score_f = None
    if is_primary:
        score_f = open(score_path, "w", encoding="utf-8")
    total_runs = len(reward_combos) * len(prompts) * max(args.repeat_count, 1)
    gen_pbar = (
        tqdm(
            total=total_runs,
            desc="Generations (sequential: one full denoise + eval per tick)",
            unit="run",
            position=0,
            leave=True,
        )
        if is_primary
        else None
    )
    try:
        for guidance_reward_fn, style_target in reward_combos:
            combo_key = (
                f"{guidance_reward_fn}|{style_target}"
                if style_target is not None
                else guidance_reward_fn
            )
            combo_stats[combo_key] = {"total": 0, "success": 0, "failure": 0, "errors": 0}

            for prompt_idx, item in enumerate(prompts):
                prompt_text = item["prompt"]
                prompt_id = item.get("id", f"prompt-{prompt_idx:04d}")
                prompt_slug = slugify_text(prompt_text)
                prompt_folder_name = f"p{prompt_idx:04d}_{prompt_id}_{prompt_slug}"
                prompt_intermediate_dir = os.path.join(inter_dir, prompt_folder_name)
                trace_intermediate_dir = prompt_intermediate_dir
                if not args.use_smc and args.record_baseline_trace:
                    trace_intermediate_dir = os.path.join(
                        prompt_intermediate_dir, "no_steering_trace"
                    )
                for repeat_idx in range(args.repeat_count):
                    per_run_seed = args.seed + prompt_idx * 10_000 + repeat_idx
                    torch.manual_seed(per_run_seed)
                    torch.cuda.manual_seed(per_run_seed)
                    torch.cuda.manual_seed_all(per_run_seed)

                    reward_config = make_reward_config_for_reward(
                        reward_name=guidance_reward_fn,
                        style_target=style_target,
                        output_dir=output_dir,
                    )
                    qn = (getattr(args, "vlm_reward_model", None) or "").strip()
                    if qn:
                        reward_config["qwen_model_name"] = qn
                    vm = (getattr(args, "vlm_model_name", None) or "").strip()
                    if vm:
                        reward_config["vlm_model_name"] = vm
                    vhc = (getattr(args, "vlm_hf_model_class", None) or "").strip()
                    if vhc:
                        reward_config["vlm_hf_model_class"] = vhc
                    if (
                        guidance_reward_fn == "VLMOCRScoreV2"
                        and getattr(args, "vlm_ocr_log_include_model_slug", False)
                    ):
                        eff = (
                            (reward_config.get("vlm_model_name") or "").strip()
                            or (reward_config.get("qwen_model_name") or "").strip()
                        )
                        if eff:
                            slug = slugify_text(eff.replace("/", " "))
                            reward_config["vlm_log_path"] = os.path.join(
                                output_dir,
                                f"vlm_ocr_v2__{slug}__intermediate_logs.jsonl",
                            )
                    apply_vlm_parallel_cli_overrides(reward_config, args)
                    apply_vlm_only_cuda_ids_budget(reward_config, args)
                    _mute_reward_file_logging_for_non_primary(reward_config)

                    record_trace = bool(args.record_baseline_trace) and not args.use_smc
                    fkd_args = {
                        "lmbda": args.lmbda,
                        "num_particles": args.num_particles,
                        "adaptive_resampling": args.adaptive_resampling,
                        "resample_frequency": args.resample_frequency,
                        "time_steps": args.time_steps,
                        "potential_type": args.potential_type,
                        "resampling_t_start": args.resampling_t_start,
                        "resampling_t_end": args.resampling_t_end,
                        "guidance_reward_fn": guidance_reward_fn,
                        "use_smc": args.use_smc,
                        "record_reward_trace": record_trace,
                        "reward_history": [],
                        "record_reward_history": bool(args.use_smc or record_trace),
                        "include_terminal_resample": args.include_terminal_resample,
                        "intermediate_images_dir": trace_intermediate_dir,
                        "run_output_dir": output_dir,
                        "reward_config": reward_config,
                    }

                    run_prompt = [prompt_text] * args.num_particles
                    start_time = datetime.now()
                    run_status = "ok"
                    error_msg = None
                    error_tb = None
                    top_score = None
                    results: Dict = {}
                    images = []

                    if is_primary:
                        try:
                            images = pipeline(
                                run_prompt,
                                num_inference_steps=fkd_args["time_steps"],
                                eta=args.eta,
                                fkd_args=fkd_args,
                            )[0]
                        except Exception as exc:
                            run_status = "error"
                            error_msg = str(exc)
                            error_tb = traceback.format_exc()
                            combo_stats[combo_key]["errors"] += 1
                            combo_stats[combo_key]["failure"] += 1
                            images = []
                            stamp = (
                                f"{datetime.now().isoformat()} "
                                f"combo={combo_key} prompt_idx={prompt_idx} repeat={repeat_idx} "
                                f"seed={per_run_seed} prompt_id={prompt_id}"
                            )
                            with open(err_log_path, "a", encoding="utf-8") as ef:
                                if error_msg not in logged_pipeline_error_msgs:
                                    logged_pipeline_error_msgs.add(error_msg)
                                    ef.write(f"\n--- first occurrence ---\n{stamp}\n{error_tb}")
                                    print(error_tb, file=sys.stderr, flush=True)
                                else:
                                    ef.write(
                                        f"\n--- repeat ({error_msg[:120]}...) ---\n{stamp}\n"
                                    )

                    # Other ranks must not enter broadcast while rank 0 is still inside pipeline().
                    _dist_barrier()

                    images, run_status, error_msg, error_tb = _broadcast_run_bundle_from_rank0(
                        images, run_status, error_msg, error_tb
                    )

                    if run_status == "ok" and images:
                        results = do_eval(
                            prompt=run_prompt,
                            images=images,
                            metrics_to_compute=metrics_to_compute,
                            reward_config=fkd_args.get("reward_config") or {},
                        )

                    if is_primary and run_status == "ok" and images:
                        guidance_vals = np.array(results[guidance_reward_fn]["result"], dtype=float)
                        sorted_idx = np.argsort(guidance_vals)[::-1]
                        images = [images[i] for i in sorted_idx]
                        top_score = float(guidance_vals[sorted_idx[0]])
                        is_success = top_score >= args.success_threshold
                        combo_stats[combo_key]["success" if is_success else "failure"] += 1

                    if is_primary:
                        combo_stats[combo_key]["total"] += 1
                    elapsed_s = (datetime.now() - start_time).total_seconds()

                    combo_slug = slugify_text(combo_key, max_len=40)
                    prompt_image_dir = os.path.join(images_dir, prompt_folder_name)
                    prompt_plot_dir = os.path.join(plots_dir, prompt_folder_name)
                    trace_path = None

                    if is_primary:
                        os.makedirs(prompt_image_dir, exist_ok=True)
                        os.makedirs(prompt_plot_dir, exist_ok=True)
                        run_base_name = (
                            f"{combo_slug}__repeat-{repeat_idx:03d}__seed-{per_run_seed}"
                        )
                        image_fpath = os.path.join(prompt_image_dir, f"{run_base_name}.png")

                        if images:
                            if args.save_all_particles:
                                fig, ax = plt.subplots(
                                    1, args.num_particles, figsize=(args.num_particles * 4, 4)
                                )
                                for i, image in enumerate(images):
                                    ax[i].imshow(image)
                                    ax[i].axis("off")
                            else:
                                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                                ax.imshow(images[0])
                                ax.axis("off")
                            plt.suptitle(prompt_text)
                            plt.savefig(image_fpath)
                            plt.close(fig)

                        rh = fkd_args.get("reward_history")
                        if rh is not None and len(rh) > 0:
                            trace = deepcopy(fkd_args["reward_history"])
                            trace_path = os.path.join(
                                prompt_plot_dir,
                                f"{run_base_name}__reward-trace.json",
                            )
                            with open(trace_path, "w", encoding="utf-8") as tf:
                                json.dump(trace, tf, indent=2)
                            plot_path = os.path.join(
                                prompt_plot_dir,
                                f"{run_base_name}__reward-trace.png",
                            )
                            trace_title = (
                                f"{combo_key} | {prompt_id} (no steering, trace only)"
                                if record_trace
                                else f"{combo_key} | {prompt_id}"
                            )
                            plot_fkd_reward_trace(
                                trace, title=trace_title, save_path=plot_path
                            )
                            plt.close("all")

                        payload = {
                            "status": run_status,
                            "error": error_msg,
                            "seed": per_run_seed,
                            "prompt": prompt_text,
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_idx,
                            "repeat_index": repeat_idx,
                            "combo_key": combo_key,
                            "guidance_reward_fn": guidance_reward_fn,
                            "style_target": style_target,
                            "time_taken": elapsed_s,
                            "success_threshold": args.success_threshold,
                            "top_guidance_score": top_score,
                            "image_path": image_fpath if images else None,
                            "reward_trace_path": trace_path,
                            "metrics": results,
                        }
                        assert score_f is not None
                        score_f.write(json.dumps(payload, default=str) + "\n")
                    if gen_pbar is not None:
                        gen_pbar.update(1)

    finally:
        if gen_pbar is not None:
            gen_pbar.close()
        if score_f is not None:
            score_f.close()

    _dist_barrier()

    if is_primary:
        summary = {"output_dir": output_dir, "combo_stats": {}}
        for combo_key, stats in combo_stats.items():
            total = max(stats["total"], 1)
            summary["combo_stats"][combo_key] = {
                **stats,
                "success_rate": stats["success"] / total,
                "failure_rate": stats["failure"] / total,
                "error_rate": stats["errors"] / total,
            }

        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        diag_path = os.path.join(output_dir, "resampling_history.jsonl")
        if os.path.exists(diag_path):
            plot_resampling_diagnostics(
                diag_path,
                save_path=os.path.join(plots_dir, "resampling_diagnostics.png"),
            )

        print(json.dumps(summary, indent=2))
        if logged_pipeline_error_msgs:
            print(
                f"Pipeline errors logged with tracebacks (first occurrence each): {err_log_path}",
                file=sys.stderr,
                flush=True,
            )

    _dist_barrier()


if __name__ == "__main__":
    main()

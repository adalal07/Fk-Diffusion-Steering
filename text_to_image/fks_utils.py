"""
Utility functions for the FKD pipeline.
"""
import warnings
import json

import torch
from diffusers import DDIMScheduler

from fkd_pipeline_sdxl import FKDStableDiffusionXL
from fkd_pipeline_sd import FKDStableDiffusion

from fkd_diffusers.rewards import (
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_human_preference_score,
    do_llm_grading,
    do_grounding_dino_spatial_reward,
    do_moondream_style_reward,
    do_qwen3_vl_style_reward
)


def _agg_stats(scores):
    """Mean / std / max / min for a list of scalars; std is 0 when n<=1 (avoids NaN)."""
    results_arr = torch.tensor(scores, dtype=torch.float32)
    mean = results_arr.mean().item()
    if results_arr.numel() <= 1:
        std = 0.0
    else:
        std = results_arr.std(unbiased=False).item()
        if std != std:
            std = 0.0
    return mean, std, results_arr.max().item(), results_arr.min().item()


def get_model(model_name):
    """
    Get the FKD-supported model based on the model name.
    """
    if model_name == "stable-diffusion-xl":
        pipeline = FKDStableDiffusionXL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-5":
        pipeline = FKDStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-v1-4":
        pipeline = FKDStableDiffusion.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    elif model_name == "stable-diffusion-2-1":
        pipeline = FKDStableDiffusion.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    return pipeline



def do_eval(*, prompt, images, metrics_to_compute, reward_config=None):
    """
    Compute the metrics for the given images and prompt.

    reward_config: optional dict passed only to GroundingDINOSpatial
    (e.g. spatial_relations, box_threshold, grounding_model_name).
    """
    if reward_config is None:
        reward_config = {}
    results = {}
    for raw_metric in metrics_to_compute:
        # Splitting args.metrics_to_compute on "#" often leaves trailing spaces.
        metric = raw_metric.strip()
        if not metric:
            continue
        if metric == "Clip-Score":
            results[metric] = {}
            (
                results[metric]["result"],
                results[metric]["diversity"],
            ) = do_clip_score_diversity(images=images, prompts=prompt)
            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin

        elif metric == "ImageReward":
            results[metric] = {}
            results[metric]["result"] = do_image_reward(images=images, prompts=prompt)

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin

        elif metric == "Clip-Score-only":
            results[metric] = {}
            results[metric]["result"] = do_clip_score(images=images, prompts=prompt)

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin
        elif metric == "HumanPreference":
            results[metric] = {}
            results[metric]["result"] = do_human_preference_score(
                images=images, prompts=prompt
            )

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin

        elif metric == "LLMGrader":
            results[metric] = {}
            out = do_llm_grading(images=images, prompts=prompt)
            print(out)
            results[metric]["result"] = out

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin
        elif metric == "GroundingDINOSpatial":
            results[metric] = {}
            results[metric]["result"] = do_grounding_dino_spatial_reward(
                images=images, prompts=prompt, **reward_config
            )

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin

        elif metric == "MoonDreamStyle":
            results[metric] = {}
            results[metric]["result"] = do_moondream_style_reward(
                images=images, prompts=prompt, **reward_config
            )

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin
        elif metric == "Qwen3VLStyle":
            results[metric] = {}
            results[metric]["result"] = do_qwen3_vl_style_reward(
                images=images, prompts=prompt, **reward_config
            )

            mean, std, vmax, vmin = _agg_stats(results[metric]["result"])
            results[metric]["mean"] = mean
            results[metric]["std"] = std
            results[metric]["max"] = vmax
            results[metric]["min"] = vmin
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


def plot_fkd_reward_trace(
    reward_history,
    *,
    title="FK steering reward vs diffusion step index",
    save_path=None,
    show_particles=True,
    show_mean_band=True,
    figsize=(9, 4),
    ax=None,
):
    """
    Plot reward_fn values recorded during SMC / FK steering (see fkd_args['record_reward_history']).

    X-axis: ``sampling_idx`` — the denoising loop index ``i`` (0 … num_inference_steps-1).
    Rewards are only computed at resampling steps (see ``resample_frequency`` / ``resampling_t_*``),
    so the plot shows points only where the steering actually evaluated the reward.

    Parameters
    ----------
    reward_history : list[dict] | None
        From ``fkd_args['reward_history']`` or ``pipeline._last_fkd.reward_history`` after a run.
    show_particles : bool
        If True, one line per particle (same prompt, different SMC particles).
    show_mean_band : bool
        Plot mean reward across particles and optional min/max band.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_fkd_reward_trace requires matplotlib. pip install matplotlib"
        ) from e

    if not reward_history:
        warnings.warn("plot_fkd_reward_trace: empty reward_history; nothing to plot.")
        return None

    xs = [h["sampling_idx"] for h in reward_history]
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=figsize)

    if show_particles:
        n_particles = len(reward_history[0]["rewards"])
        for p in range(n_particles):
            ys = [h["rewards"][p] for h in reward_history]
            ax.plot(xs, ys, alpha=0.45, linewidth=1.0, label=f"particle {p}")

    if show_mean_band:
        means = [h["mean"] for h in reward_history]
        ax.plot(xs, means, color="black", linewidth=2.0, label="mean")
        mins = [h["min"] for h in reward_history]
        maxs = [h["max"] for h in reward_history]
        ax.fill_between(xs, mins, maxs, color="gray", alpha=0.2, label="min–max")

    ax.set_xlabel("Diffusion step index (sampling_idx)")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return ax.figure
    if save_path:
        warnings.warn(
            "save_path was set but ax was provided; save the figure from the Figure object instead."
        )
    return None


def plot_metric_scores(
    metric_name,
    metric_result,
    *,
    title=None,
    save_path=None,
    figsize=(7, 3.8),
    ax=None,
):
    """
    Plot per-particle scores for any evaluated reward metric.

    Parameters
    ----------
    metric_name : str
        Name in results dict, e.g. ``Clip-Score`` / ``ImageReward`` / ``HumanPreference``.
    metric_result : dict
        ``results[metric_name]`` from ``do_eval``; must include key ``result`` as a list of scalars.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_metric_scores requires matplotlib. pip install matplotlib"
        ) from e

    if not metric_result or "result" not in metric_result:
        warnings.warn(f"plot_metric_scores: no result found for metric {metric_name}.")
        return None
    vals = metric_result["result"]
    if vals is None or len(vals) == 0:
        warnings.warn(f"plot_metric_scores: empty result for metric {metric_name}.")
        return None

    xs = list(range(len(vals)))
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(xs, vals, marker="o", linewidth=1.4, alpha=0.9, label=metric_name)
    ax.set_xlabel("Particle index")
    ax.set_ylabel("Score")
    ax.set_title(title or f"{metric_name} scores by particle")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return ax.figure
    if save_path:
        warnings.warn(
            "save_path was set but ax was provided; save the figure from the Figure object instead."
        )
    return None


def plot_resampling_diagnostics(
    resampling_history_path,
    *,
    title_prefix="FKD resampling diagnostics",
    save_path=None,
    figsize=(10, 8),
):
    """
    Plot FKD resampling diagnostics from ``resampling_history.jsonl``.

    Produces three stacked plots:
    - ESS vs sampling step
    - Max normalized weight vs sampling step
    - Unique selected parent count vs sampling step
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plot_resampling_diagnostics requires matplotlib. pip install matplotlib"
        ) from e

    rows = []
    with open(resampling_history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        warnings.warn(
            f"plot_resampling_diagnostics: no rows found in {resampling_history_path!r}."
        )
        return None

    xs = []
    ess_vals = []
    max_norm_w = []
    uniq_parents = []
    did_resample = []
    for r in rows:
        step = int(r.get("sampling_idx", 0))
        xs.append(step)
        ess = r.get("ess", None)
        ess_vals.append(float("nan") if ess is None else float(ess))
        weights = [float(w) for w in r.get("weights", [])]
        if weights:
            wt = torch.tensor(weights, dtype=torch.float32)
            s = float(wt.sum().item())
            max_norm_w.append(float((wt / max(s, 1e-12)).max().item()))
        else:
            max_norm_w.append(float("nan"))
        sel = r.get("selected_indices", [])
        uniq_parents.append(len(set(int(i) for i in sel)) if sel else 0)
        did_resample.append(bool(r.get("did_resample", False)))

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(xs, ess_vals, marker="o", linewidth=1.4, alpha=0.9, label="ESS")
    axes[0].set_ylabel("ESS")
    axes[0].set_title(f"{title_prefix} - ESS")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(
        xs, max_norm_w, marker="o", linewidth=1.4, alpha=0.9, color="tab:orange", label="max normalized weight"
    )
    axes[1].set_ylabel("max p(parent)")
    axes[1].set_title(f"{title_prefix} - Selection concentration")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(
        xs, uniq_parents, marker="o", linewidth=1.4, alpha=0.9, color="tab:green", label="unique selected parents"
    )
    for x, flag in zip(xs, did_resample):
        if flag:
            axes[2].axvline(x, color="gray", alpha=0.15, linewidth=1.0)
    axes[2].set_ylabel("# unique parents")
    axes[2].set_xlabel("Diffusion step index (sampling_idx)")
    axes[2].set_title(f"{title_prefix} - Parent diversity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

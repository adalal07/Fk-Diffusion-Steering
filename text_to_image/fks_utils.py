"""
Utility functions for the FKD pipeline.
"""
import warnings

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

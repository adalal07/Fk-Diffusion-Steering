"""
Utility functions for the FKD pipeline.
"""
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

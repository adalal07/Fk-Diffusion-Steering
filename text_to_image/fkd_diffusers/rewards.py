import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2
import re
import warnings
import numpy as np

from llm_grading import LLMGrader

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "GroundingDINOSpatial": None,
}


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(
    reward_name,
    images,
    prompts,
    metric_to_chase="overall_score",
    reward_config=None,
    reward_kwargs=None,
):
    # Backward-compatible alias: reward_kwargs -> reward_config
    if reward_config is None:
        reward_config = reward_kwargs if reward_kwargs is not None else {}
    if reward_name != "LLMGrader":
        print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)
    
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)
    
    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)
    
    elif reward_name == "GroundingDINOSpatial":
        return do_grounding_dino_spatial_reward(images=images, prompts=prompts, **reward_config)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")


RELATION_PATTERNS = {
    "left_of": re.compile(r"(.+?)\s+(?:is\s+)?left of\s+(.+)", re.IGNORECASE),
    "right_of": re.compile(r"(.+?)\s+(?:is\s+)?right of\s+(.+)", re.IGNORECASE),
    "on_top_of": re.compile(
        r"(.+?)\s+(?:is\s+)?(?:on top of|above|over)\s+(.+)", re.IGNORECASE
    ),
    "below": re.compile(r"(.+?)\s+(?:is\s+)?(?:below|under)\s+(.+)", re.IGNORECASE),
}

NUMBER_WORD_TO_INT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _clean_entity(text):
    return (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("the ", " ")
        .replace("a ", " ")
        .replace("an ", " ")
        .strip()
    )


def _extract_relations_from_prompt(prompt):
    prompt_l = prompt.lower()
    relations = []
    for relation, pattern in RELATION_PATTERNS.items():
        match = pattern.search(prompt_l)
        if not match:
            continue
        subject = _clean_entity(match.group(1))
        obj = _clean_entity(match.group(2))
        if subject and obj:
            relations.append({"subject": subject, "object": obj, "relation": relation})
    return relations


def _score_relation(dx, dy, relation, align_scale=6.0):
    # Score terms are in [0,1]. dx is positive if subject is to the right.
    # dy is positive when subject is higher (y-axis inverted to cartesian).
    if relation == "left_of":
        directional = torch.sigmoid(-align_scale * dx)
        orthogonal = torch.exp(-align_scale * torch.abs(dy))
    elif relation == "right_of":
        directional = torch.sigmoid(align_scale * dx)
        orthogonal = torch.exp(-align_scale * torch.abs(dy))
    elif relation == "on_top_of":
        directional = torch.sigmoid(align_scale * dy)
        orthogonal = torch.exp(-align_scale * torch.abs(dx))
    elif relation == "below":
        directional = torch.sigmoid(-align_scale * dy)
        orthogonal = torch.exp(-align_scale * torch.abs(dx))
    else:
        raise ValueError(f"Unsupported spatial relation: {relation}")

    return float((directional * orthogonal).item())


def _build_detection_prompt(objects):
    unique_objects = sorted(set(objects))
    return ". ".join(unique_objects) + "."


def _count_matching_detections(detections, obj_name):
    obj = obj_name.lower().strip()
    count = 0
    for det in detections:
        label = det["label"].lower().strip()
        if obj in label or label in obj:
            count += 1
    return count


def _plural_form(obj):
    if obj.endswith("y") and len(obj) > 1 and obj[-2] not in "aeiou":
        return obj[:-1] + "ies"
    if obj.endswith(("s", "x", "z", "ch", "sh")):
        return obj + "es"
    return obj + "s"


def _extract_expected_object_counts(prompt, objects, bare_plural_default_count=2):
    """
    Infer expected object multiplicity from prompt text.
    Defaults to 1 per object unless an explicit number is found.
    """
    prompt_l = prompt.lower()
    expected = {obj: 1 for obj in sorted(set(objects))}
    for obj in expected:
        obj_esc = re.escape(obj)
        obj_plural = _plural_form(obj)
        obj_plural_esc = re.escape(obj_plural)

        # 1) Explicit numeric quantity, e.g. "2 cats", "two cats", "one dog"
        pattern = re.compile(
            rf"\b(?:(\d+)|({'|'.join(NUMBER_WORD_TO_INT.keys())}))\s+({obj_esc}|{obj_plural_esc})\b",
            re.IGNORECASE,
        )
        match = pattern.search(prompt_l)
        if match:
            if match.group(1):
                expected[obj] = max(int(match.group(1)), 0)
            elif match.group(2):
                expected[obj] = NUMBER_WORD_TO_INT[match.group(2).lower()]
            continue

        # 2) Singular determiner implies one: "a cat", "an apple", "the dog"
        singular_pattern = re.compile(
            rf"\b(a|an|the|one)\s+{obj_esc}\b",
            re.IGNORECASE,
        )
        if singular_pattern.search(prompt_l):
            expected[obj] = 1
            continue

        # 3) Bare plural implies more than one: "cats", "dogs"
        plural_pattern = re.compile(rf"\b{obj_plural_esc}\b", re.IGNORECASE)
        if plural_pattern.search(prompt_l):
            expected[obj] = max(int(bare_plural_default_count), 2)
            continue

        # 4) Bare singular mention keeps the default of one.
        bare_singular_pattern = re.compile(rf"\b{obj_esc}\b", re.IGNORECASE)
        if bare_singular_pattern.search(prompt_l):
            expected[obj] = 1
            continue
    return expected


def _score_object_count(observed_count, expected_count):
    # Incremental toward expected count and penalizes over/under counts symmetrically.
    if expected_count <= 0:
        return 1.0 if observed_count == 0 else 0.0
    return max(0.0, 1.0 - abs(observed_count - expected_count) / expected_count)


def _match_best_box(detections, obj_name):
    obj = obj_name.lower().strip()
    best = None
    for det in detections:
        label = det["label"].lower().strip()
        if obj in label or label in obj:
            if best is None or det["score"] > best["score"]:
                best = det
    return best


def _import_grounding_dino():
    """
    Lazy import so ImageReward / CLIP / notebook imports work on older transformers.
    Grounding DINO needs a recent transformers (e.g. >= 4.40; see HF release notes).
    """
    try:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
    except ImportError as e:
        raise ImportError(
            "GroundingDINOSpatial requires GroundingDino* classes in `transformers`. "
            "Upgrade with: pip install -U 'transformers>=4.40'"
        ) from e
    return GroundingDinoForObjectDetection, GroundingDinoProcessor


def do_grounding_dino_spatial_reward(
    *,
    images,
    prompts,
    spatial_relations=None,
    grounding_model_name="IDEA-Research/grounding-dino-base",
    box_threshold=0.25,
    text_threshold=0.25,
    align_scale=6.0,
    missing_box_penalty=-1.0,
    no_relation_score=0.0,
    warn_no_relation=True,
    expected_object_counts=None,
    relation_weight=0.7,
    object_count_weight=0.3,
    bare_plural_default_count=2,
):
    global REWARDS_DICT

    if REWARDS_DICT["GroundingDINOSpatial"] is None:
        GroundingDinoForObjectDetection, GroundingDinoProcessor = _import_grounding_dino()
        warnings.warn(
            "Grounding DINO in Transformers may log missing MSDeformAttn CUDA sources "
            "(vision.cpp). A slower PyTorch fallback is used; those messages are usually harmless.",
            UserWarning,
            stacklevel=2,
        )
        processor = GroundingDinoProcessor.from_pretrained(grounding_model_name)
        model = GroundingDinoForObjectDetection.from_pretrained(grounding_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        REWARDS_DICT["GroundingDINOSpatial"] = {
            "processor": processor,
            "model": model,
            "device": device,
        }

    processor = REWARDS_DICT["GroundingDINOSpatial"]["processor"]
    model = REWARDS_DICT["GroundingDINOSpatial"]["model"]
    device = REWARDS_DICT["GroundingDINOSpatial"]["device"]

    rewards = []
    skipped_no_relation = 0
    for i, image in enumerate(images):
        prompt_relations = []
        if spatial_relations is not None and i < len(spatial_relations):
            prompt_relations = spatial_relations[i]
        if not prompt_relations:
            prompt_relations = _extract_relations_from_prompt(prompts[i])

        if not prompt_relations:
            skipped_no_relation += 1
            rewards.append(float(no_relation_score))
            continue

        objects = []
        for rel in prompt_relations:
            objects.append(_clean_entity(rel["subject"]))
            objects.append(_clean_entity(rel["object"]))

        det_prompt = _build_detection_prompt(objects)
        inputs = processor(images=image, text=det_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        width, height = image.size
        target_sizes = torch.tensor([[height, width]], device=device)
        processed = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        labels = processed["labels"]
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = box.tolist()
            cx = (x0 + x1) / 2.0 / width
            cy = 1.0 - ((y0 + y1) / 2.0 / height)  # invert y-axis so up is positive
            detections.append(
                {
                    "label": str(label),
                    "score": float(score),
                    "center": (float(cx), float(cy)),
                }
            )

        relation_scores = []
        for rel in prompt_relations:
            subject = _clean_entity(rel["subject"])
            obj = _clean_entity(rel["object"])
            relation = rel["relation"]

            subject_det = _match_best_box(detections, subject)
            object_det = _match_best_box(detections, obj)
            if subject_det is None or object_det is None:
                relation_scores.append(missing_box_penalty)
                continue

            dx = subject_det["center"][0] - object_det["center"][0]
            dy = subject_det["center"][1] - object_det["center"][1]
            relation_scores.append(
                _score_relation(dx=torch.tensor(dx), dy=torch.tensor(dy), relation=relation, align_scale=align_scale)
            )

        if expected_object_counts is not None and i < len(expected_object_counts):
            expected_counts = {
                _clean_entity(k): int(v) for k, v in expected_object_counts[i].items()
            }
        else:
            expected_counts = _extract_expected_object_counts(
                prompts[i], objects, bare_plural_default_count=bare_plural_default_count
            )

        object_count_scores = []
        for obj_name, exp_count in expected_counts.items():
            obs_count = _count_matching_detections(detections, obj_name)
            object_count_scores.append(_score_object_count(obs_count, exp_count))

        relation_component = float(np.mean(relation_scores)) if relation_scores else 0.0
        object_count_component = (
            float(np.mean(object_count_scores)) if object_count_scores else 0.0
        )
        final_reward = (
            relation_weight * relation_component
            + object_count_weight * object_count_component
        )
        rewards.append(final_reward)

    if warn_no_relation and skipped_no_relation:
        warnings.warn(
            f"GroundingDINOSpatial: no parseable spatial relations for {skipped_no_relation} "
            f"image(s); returned no_relation_score={no_relation_score} for those. "
            "Use prompts like 'a cat left of a chair' / 'an apple on top of a book', "
            "or pass spatial_relations in reward_config / fkd_args['reward_config'].",
            UserWarning,
            stacklevel=2,
        )

    return rewards
    
# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        # Lazy import keeps the module importable even if ImageReward's
        # transformers compatibility differs from the active environment.
        from image_reward_utils import rm_load
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT
    
    if REWARDS_DICT["LLMGrader"] is None:
        REWARDS_DICT["LLMGrader"]  = LLMGrader()
    llm_grading_result = [
        REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
        for i, prompt in enumerate(prompts)
    ]
    return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()

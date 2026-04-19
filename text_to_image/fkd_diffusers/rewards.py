import inspect
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


RELATION_PATTERN_SPECS = {
    # Handles:
    # - "a dog right of a cat" / "dog to the right of a cat"
    # - "the cat is to the left of the dog and fish"  (object side splits on and)
    # - "the dog and the cat are to the left of the fish" (subject side splits on and)
    "left_of": [
        r"(?P<subject>.+?)\s+(?:(?:is|are)\s+)?(?:to\s+the\s+)?left of\s+(?P<object>.+)",
        r"(?P<subject>.+?)\s+(?:is\s+)?on\s+the\s+left\s+side\s+of\s+(?P<object>.+)",
    ],
    "right_of": [
        r"(?P<subject>.+?)\s+(?:(?:is|are)\s+)?(?:to\s+the\s+)?right of\s+(?P<object>.+)",
        r"(?P<subject>.+?)\s+(?:is\s+)?on\s+the\s+right\s+side\s+of\s+(?P<object>.+)",
    ],
    "on_top_of": [
        r"(?P<subject>.+?)\s+(?:(?:is|are)\s+)?(?:on top of|above|over)\s+(?P<object>.+)",
    ],
    "below": [
        r"(?P<subject>.+?)\s+(?:(?:is|are)\s+)?(?:below|under|underneath)\s+(?P<object>.+)",
    ],
}

RELATION_PATTERNS = {
    rel: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for rel, patterns in RELATION_PATTERN_SPECS.items()
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
    text = re.sub(r"\([^)]*\)", " ", text)  # remove parenthetical asides
    return (
        text.lower()
        .replace(";", " ")
        .replace(":", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("-", " ")
        .replace("the ", " ")
        .replace("a ", " ")
        .replace("an ", " ")
        .replace("  ", " ")
        .strip()
    )


def _strip_trailing_clause(text):
    # Drop trailing metadata (do NOT split on "and" — needed for "dog and fish").
    splitter = re.split(
        r"\b(with|using|featuring|while|but|in\s+the\s+background)\b|,|;|\.|!|\?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return splitter[0].strip()


# Split coordinate lists: "the dog and fish", "cat, dog", "a or b"
_CONJUNCTION_SPLIT = re.compile(
    r"\s*,\s*|\s+and\s+|\s+or\s+|\s*&\s*",
    re.IGNORECASE,
)


def _split_entity_phrase(phrase):
    """
    Turn a phrase into one or more entity strings, e.g.
    "the dog and fish" -> ["dog", "fish"], "a cat" -> ["cat"].
    """
    phrase = _strip_trailing_clause(phrase.strip())
    if not phrase:
        return []
    parts = _CONJUNCTION_SPLIT.split(phrase)
    out = []
    for p in parts:
        c = _clean_entity(p)
        if c:
            out.append(c)
    if not out:
        c = _clean_entity(phrase)
        return [c] if c else []
    return out


def _expand_pairwise_relations(subject_raw, object_raw, relation):
    """
    Base pattern is (subject phrase) RELATION (object phrase).
    Each side may list several objects joined by and/or/comma; we take the
    Cartesian product of subject-side entities × object-side entities
    (excluding trivial s==o).
    """
    subs = _split_entity_phrase(subject_raw)
    objs = _split_entity_phrase(object_raw)
    relations = []
    for s in subs:
        for o in objs:
            if s and o and s != o:
                relations.append(
                    {"subject": s, "object": o, "relation": relation}
                )
    return relations


def _extract_relations_from_prompt(prompt):
    prompt_l = " ".join(prompt.lower().strip().split())
    relations = []
    for relation, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(prompt_l):
                subject_raw = match.group("subject")
                object_raw = match.group("object")
                relations.extend(
                    _expand_pairwise_relations(subject_raw, object_raw, relation)
                )

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for rel in relations:
        key = (rel["subject"], rel["relation"], rel["object"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped


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


def _label_matches_entity_word(label, entity):
    """Avoid false positives like 'cat' matching 'catalog'."""
    label = label.lower().strip()
    entity = entity.lower().strip()
    if not entity:
        return False
    if label == entity:
        return True
    return re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", label) is not None


def _count_matching_detections(detections, obj_name):
    obj = obj_name.lower().strip()
    count = 0
    for det in detections:
        label = det["label"].lower().strip()
        if _label_matches_entity_word(label, obj):
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


def _score_object_count_inventory(observed_count, expected_count):
    """
    Count consistency reward in [0, 1].

    - Missing (obs=0, exp>=1): 0 — strong penalty for absent required entities.
    - Under-count (obs < exp): linear partial credit obs / exp.
    - Exact match: 1.
    - Over-count (obs > exp): exp / obs — e.g. 6 birds vs 1 expected -> 1/6.

    This penalizes "too many birds" much more sharply than symmetric |obs-exp|/exp.
    """
    if expected_count <= 0:
        return 1.0 if observed_count == 0 else 0.0
    if observed_count <= 0:
        return 0.0
    if observed_count <= expected_count:
        return float(observed_count) / float(expected_count)
    return float(expected_count) / float(observed_count)


def _entity_match_score(det, entity):
    """Higher is better; uses detector confidence × label match strength."""
    label = det["label"].lower().strip()
    entity = entity.lower().strip()
    conf = float(det["score"])
    if not entity:
        return 0.0
    if label == entity:
        return conf * 1.0
    if _label_matches_entity_word(label, entity):
        return conf * 1.0
    if entity in label:
        return conf * 0.55
    return 0.0


def _match_best_box(detections, obj_name):
    """Pick single best box for one entity (legacy / fallback)."""
    best = None
    best_s = -1.0
    for det in detections:
        s = _entity_match_score(det, obj_name)
        if s > best_s:
            best_s = s
            best = det
    return best


def _spatial_tiebreak_bonus_full(dx, dy, relation):
    if relation == "left_of":
        return 1.0 if dx < 0 else 0.0
    if relation == "right_of":
        return 1.0 if dx > 0 else 0.0
    if relation == "on_top_of":
        return 1.0 if dy > 0 else 0.0
    if relation == "below":
        return 1.0 if dy < 0 else 0.0
    return 0.0


def _assign_subject_object_boxes(subject, obj, detections, relation, spatial_tiebreak_weight):
    """
    Assign two *distinct* detections to (subject, object) roles.

    Independent per-entity argmax can pick the same box twice or swap identities
    when labels are noisy; we score all ordered pairs and take the best.
    """
    if len(detections) == 0:
        return None, None
    if len(detections) == 1:
        # Cannot assign two roles to one box without ambiguity.
        return _match_best_box(detections, subject), None

    best_pair = (None, None)
    best_total = -1.0
    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_sub, d_obj = detections[i], detections[j]
            label_total = _entity_match_score(d_sub, subject) + _entity_match_score(
                d_obj, obj
            )
            cx_sub = d_sub["center"][0]
            cx_obj = d_obj["center"][0]
            cy_sub = d_sub["center"][1]
            cy_obj = d_obj["center"][1]
            dx = cx_sub - cx_obj
            dy = cy_sub - cy_obj
            geom = _spatial_tiebreak_bonus_full(dx, dy, relation)
            total = label_total + spatial_tiebreak_weight * geom
            if total > best_total:
                best_total = total
                best_pair = (d_sub, d_obj)

    # No confident label-to-box association for this pair of roles.
    if best_total < 0.05:
        return None, None

    return best_pair


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


def _post_process_grounding_dino_detections(
    processor, outputs, input_ids, target_sizes, box_threshold, text_threshold
):
    """
    Hugging Face renamed ``box_threshold`` → ``threshold`` in
    ``GroundingDinoProcessor.post_process_grounded_object_detection`` at v4.51.
    Call the processor with whichever kwargs the installed transformers expects.
    """
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters

    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "target_sizes": target_sizes,
        "text_threshold": text_threshold,
    }
    if "threshold" in params:
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in params:
        kwargs["box_threshold"] = box_threshold
    else:
        raise TypeError(
            "GroundingDinoProcessor.post_process_grounded_object_detection has no "
            "'threshold' or 'box_threshold' parameter; upgrade or pin transformers."
        )

    return fn(**kwargs)


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
    relation_weight=0.55,
    object_count_weight=0.45,
    bare_plural_default_count=2,
    use_paired_box_assignment=True,
    spatial_tiebreak_weight=0.15,
    inventory_aggregate="mean",
):
    global REWARDS_DICT

    if inventory_aggregate not in ("mean", "min"):
        raise ValueError(
            "inventory_aggregate must be 'mean' or 'min' (min = strict: one bad entity tanks inventory)."
        )

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
        processed = _post_process_grounding_dino_detections(
            processor,
            outputs,
            inputs.input_ids,
            target_sizes,
            box_threshold,
            text_threshold,
        )[0]

        detections = []
        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        # v4.51+ prefers text_labels; labels may be deprecated / int ids in future.
        labels = processed.get("text_labels")
        if labels is None:
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

            if use_paired_box_assignment:
                subject_det, object_det = _assign_subject_object_boxes(
                    subject,
                    obj,
                    detections,
                    relation,
                    spatial_tiebreak_weight,
                )
            else:
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
            object_count_scores.append(
                _score_object_count_inventory(obs_count, exp_count)
            )

        relation_component = float(np.mean(relation_scores)) if relation_scores else 0.0
        if object_count_scores:
            if inventory_aggregate == "min":
                object_count_component = float(min(object_count_scores))
            else:
                object_count_component = float(np.mean(object_count_scores))
        else:
            object_count_component = 0.0
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

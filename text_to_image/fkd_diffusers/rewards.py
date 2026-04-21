import inspect
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2
import os
import re
import warnings
import numpy as np
import json

# PyPI ``hpsv2`` wheels sometimes omit ``bpe_simple_vocab_16e6.txt.gz`` under
# ``hpsv2/src/open_clip/``, which breaks HumanPreference scoring at runtime.
_HPSV2_OPENCLIP_VOCAB_READY = False


def _ensure_hpsv2_open_clip_vocab():
    """
    Ensure ``bpe_simple_vocab_16e6.txt.gz`` exists where bundled hpsv2 open_clip expects it.

    Tries, in order: copy from ``open_clip``, copy from OpenAI ``clip``, then download
    from the official CLIP repo. Raises ``RuntimeError`` with fix hints if all fail.
    """
    global _HPSV2_OPENCLIP_VOCAB_READY
    if _HPSV2_OPENCLIP_VOCAB_READY:
        return

    hps_root = os.path.dirname(os.path.abspath(hpsv2.__file__))
    dest_dir = os.path.join(hps_root, "src", "open_clip")
    dest = os.path.join(dest_dir, "bpe_simple_vocab_16e6.txt.gz")

    def _looks_valid(path):
        try:
            return os.path.isfile(path) and os.path.getsize(path) > 5000
        except OSError:
            return False

    if _looks_valid(dest):
        _HPSV2_OPENCLIP_VOCAB_READY = True
        return

    os.makedirs(dest_dir, exist_ok=True)

    for mod_name in ("open_clip",):
        try:
            mod = __import__(mod_name)
            src = os.path.join(
                os.path.dirname(os.path.abspath(mod.__file__)),
                "bpe_simple_vocab_16e6.txt.gz",
            )
            if _looks_valid(src):
                shutil.copyfile(src, dest)
                _HPSV2_OPENCLIP_VOCAB_READY = True
                return
        except ImportError:
            continue

    try:
        clip_dir = os.path.dirname(os.path.abspath(clip.__file__))
        src = os.path.join(clip_dir, "bpe_simple_vocab_16e6.txt.gz")
        if _looks_valid(src):
            shutil.copyfile(src, dest)
            _HPSV2_OPENCLIP_VOCAB_READY = True
            return
    except Exception:
        pass

    url = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
    dl_msg = "download not attempted"
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        if len(data) > 5000:
            with open(dest, "wb") as f:
                f.write(data)
            _HPSV2_OPENCLIP_VOCAB_READY = True
            return
        dl_msg = f"download returned only {len(data)} bytes"
    except Exception as dl_err:
        dl_msg = str(dl_err)

    raise RuntimeError(
        "HumanPreference (hpsv2): tokenizer asset "
        "`bpe_simple_vocab_16e6.txt.gz` is missing from the hpsv2 install. "
        "Fix options:\n"
        "  1) `pip install -U open_clip_torch` then re-run (this code will copy the vocab into hpsv2).\n"
        "  2) `pip install --force-reinstall hpsv2` or try `pip install hpsv2x` (community repack).\n"
        f"  3) Manually download:\n     {url}\n"
        f"     to:\n     {dest}\n"
        f"Download attempt failed with: {dl_msg}"
    )

from llm_grading import LLMGrader

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "GroundingDINOSpatial": None,
    "MoonDreamStyle": None,
    "Qwen3VLStyle": None,
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
    if reward_name not in ("LLMGrader",):
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
        cfg = dict(reward_config or {})
        debug_overlay_dir = cfg.pop("debug_overlay_dir", None)
        debug_sampling_idx = cfg.pop("debug_sampling_idx", None)
        return do_grounding_dino_spatial_reward(
            images=images,
            prompts=prompts,
            debug_overlay_dir=debug_overlay_dir,
            debug_sampling_idx=debug_sampling_idx,
            **cfg,
        )

    elif reward_name == "MoonDreamStyle":
        cfg = dict(reward_config or {})
        debug_overlay_dir = cfg.pop("debug_overlay_dir", None)
        debug_sampling_idx = cfg.pop("debug_sampling_idx", None)
        return do_moondream_style_reward(
            images=images,
            prompts=prompts,
            debug_overlay_dir=debug_overlay_dir,
            debug_sampling_idx=debug_sampling_idx,
            **cfg,
        )

    elif reward_name == "Qwen3VLStyle":
        cfg = dict(reward_config or {})
        debug_overlay_dir = cfg.pop("debug_overlay_dir", None)
        debug_sampling_idx = cfg.pop("debug_sampling_idx", None)
        return do_qwen3_vl_style_reward(
            images=images,
            prompts=prompts,
            debug_overlay_dir=debug_overlay_dir,
            debug_sampling_idx=debug_sampling_idx,
            **cfg,
        )

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


def _gaussian_relation_score(dx, dy, relation, sigma=0.3):
    """
    Gaussian spatial score around relation-specific target offsets.
    Inputs are normalized center deltas in [-1, 1].
    """
    if sigma <= 0:
        sigma = 1e-3
    if relation == "left_of":
        primary = float(dx)
        primary_target = -0.35
        orthogonal = float(dy)
        good_side = primary <= primary_target
    elif relation == "right_of":
        primary = float(dx)
        primary_target = 0.35
        orthogonal = float(dy)
        good_side = primary >= primary_target
    elif relation == "on_top_of":
        primary = float(dy)
        primary_target = 0.35
        orthogonal = float(dx)
        good_side = primary >= primary_target
    elif relation == "below":
        primary = float(dy)
        primary_target = -0.35
        orthogonal = float(dx)
        good_side = primary <= primary_target
    else:
        raise ValueError(f"Unsupported spatial relation: {relation}")
    # Half-Gaussian on the directional axis:
    # keep a flat plateau (1.0) once clearly on the correct side.
    if good_side:
        directional = 1.0
    else:
        directional = float(
            np.exp(-((primary - primary_target) ** 2) / (2.0 * float(sigma) ** 2))
        )
    # Still softly prefer limited orthogonal displacement.
    ortho_sigma = float(sigma) * 1.5
    orth_score = float(
        np.exp(-(orthogonal**2) / (2.0 * max(ortho_sigma, 1e-3) ** 2))
    )
    return directional * orth_score


def _stage_weighting(
    base_relation_weight,
    base_object_count_weight,
    *,
    sampling_idx=None,
    total_time_steps=None,
    transition_progress=0.5,
    sharpness=12.0,
    early_object_phase_fraction=0.3,
    early_object_boost=2.5,
):
    """
    Early denoising: object inventory dominates.
    Later denoising: relation alignment dominates.
    """
    if sampling_idx is None or total_time_steps is None or total_time_steps <= 1:
        return float(base_relation_weight), float(base_object_count_weight)
    progress = float(sampling_idx) / float(max(int(total_time_steps) - 1, 1))
    # Smooth switch around transition_progress.
    relation_gate = 1.0 / (1.0 + np.exp(-float(sharpness) * (progress - float(transition_progress))))
    object_gate = 1.0 - relation_gate
    wr = max(0.0, float(base_relation_weight) * relation_gate)
    wo = max(0.0, float(base_object_count_weight) * object_gate)
    if progress <= float(early_object_phase_fraction):
        wo *= float(max(early_object_boost, 1.0))
    norm = wr + wo
    if norm <= 1e-8:
        return 0.5, 0.5
    return wr / norm, wo / norm


def _soft_entity_presence(detections, entity):
    """
    Continuous object presence proxy from low-threshold detections.
    Uses max entity-match score in [0,1]-ish instead of hard existence.
    """
    if not detections:
        return 0.0
    best = 0.0
    for det in detections:
        best = max(best, float(_entity_match_score(det, entity)))
    return float(np.clip(best, 0.0, 1.0))


def _soft_entity_count(detections, entity):
    """
    Continuous count estimate from low-threshold candidates.
    """
    if not detections:
        return 0.0
    total = 0.0
    for det in detections:
        total += float(_entity_match_score(det, entity))
    return float(max(total, 0.0))


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
    - Under-count (obs < exp): quadratic partial credit (obs/exp)^2.
    - Exact match: 1.
    - Over-count (obs > exp): 0.9 ("at least" semantics; slightly below exact).
    """
    if expected_count <= 0:
        return 1.0 if observed_count <= 0 else 0.9
    if observed_count <= 0:
        return 0.0
    if observed_count == expected_count:
        return 1.0
    if observed_count > expected_count:
        return 0.9
    if observed_count < expected_count:
        return float(observed_count / expected_count) ** 2
    return 0.0


def _score_object_count_inventory_topk(
    detections,
    obj_name,
    expected_count,
    *,
    extra_penalty_conf_threshold=0.5,
):
    """
    Robust inventory scorer for noisy steps:
    - Uses only top-K match scores where K=expected_count.
    - Penalizes extras only when extra detections are high confidence.
    """
    if expected_count <= 0:
        return 1.0
    scores = []
    for det in detections:
        s = float(_entity_match_score(det, obj_name))
        if s > 0.0:
            scores.append(s)
    scores.sort(reverse=True)
    top_scores = scores[: int(expected_count)]
    if not top_scores:
        return 0.0
    presence_score = float(np.mean(top_scores))
    extra_penalty = 1.0
    if len(scores) > int(expected_count):
        extras = scores[int(expected_count) :]
        extra_high_conf = [s for s in extras if s > float(extra_penalty_conf_threshold)]
        if extra_high_conf:
            extra_penalty = float(expected_count) / float(
                expected_count + len(extra_high_conf)
            )
    return float(np.clip(presence_score * extra_penalty, 0.0, 1.0))


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


def _assign_subject_object_boxes(
    subject,
    obj,
    detections,
    relation,
    spatial_tiebreak_weight,
    min_pair_confidence=0.05,
    prev_subject_center=None,
    prev_object_center=None,
    temporal_tiebreak_weight=0.0,
    temporal_sigma=0.08,
):
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
            if (
                prev_subject_center is not None
                and prev_object_center is not None
                and temporal_tiebreak_weight > 0.0
            ):
                if temporal_sigma <= 0:
                    temporal_sigma = 1e-3
                sub_d2 = (cx_sub - prev_subject_center[0]) ** 2 + (
                    cy_sub - prev_subject_center[1]
                ) ** 2
                obj_d2 = (cx_obj - prev_object_center[0]) ** 2 + (
                    cy_obj - prev_object_center[1]
                ) ** 2
                temporal_bonus = np.exp(
                    -(sub_d2 + obj_d2) / (2.0 * float(temporal_sigma) ** 2)
                )
                total += float(temporal_tiebreak_weight) * float(temporal_bonus)
            if total > best_total:
                best_total = total
                best_pair = (d_sub, d_obj)

    # No confident label-to-box association for this pair of roles.
    if best_total < float(min_pair_confidence):
        return None, None

    return best_pair


def _save_grounding_overlay_pil(image_pil, detections, out_path, *, title_lines=None):
    """Draw Grounding DINO boxes (pixel xyxy) on a copy of the image and save."""
    from PIL import ImageDraw, ImageFont

    img = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for det in detections:
        box = det.get("box_xyxy")
        if box is None:
            continue
        x0, y0, x1, y1 = box
        w = int(max(2, min(img.size) * 0.004))
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 80), width=w)
        label = f"{det.get('label', '?')} {float(det.get('score', 0)):.2f}"[:72]
        ty = max(0, int(y0) - 14)
        draw.text((x0 + 1, ty + 1), label, fill=(0, 0, 0), font=font)
        draw.text((x0, ty), label, fill=(255, 255, 0), font=font)
    if title_lines:
        y = 4
        for line in title_lines[:4]:
            draw.text((4, y), str(line)[:100], fill=(255, 255, 255), font=font)
            draw.text((5, y + 1), str(line)[:100], fill=(0, 0, 0), font=font)
            y += 12
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    img.save(out_path)


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
    box_threshold=0.30,
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
    inventory_aggregate="min",
    debug_overlay_dir=None,
    debug_sampling_idx=None,
    debug_time_steps=None,
    use_soft_detections=True,
    use_soft_detections_outside_steering=False,
    soft_box_threshold=0.05,
    use_step_box_threshold_schedule=True,
    steering_phase_start_ratio=0.2,
    steering_phase_end_ratio=0.4,
    steering_phase_box_threshold=0.05,
    soft_missing_box_penalty=-0.2,
    use_max_entity_presence=True,
    use_topk_inventory=True,
    topk_extra_penalty_conf_threshold=0.5,
    dynamic_stage_weighting=True,
    stage_transition_progress=0.6,
    stage_sharpness=12.0,
    stage_early_object_fraction=0.45,
    stage_early_object_boost=3.5,
    relation_scoring_mode="gaussian",
    relation_gaussian_sigma=0.3,
    relation_targets=None,
    temporal_state=None,
    temporal_consistency_weight=0.3,
    temporal_tiebreak_weight=0.25,
    temporal_sigma=0.08,
    relation_duplicate_aggregate="max",
    dynamic_inventory_aggregate=True,
    inventory_aggregate_early="mean",
    inventory_aggregate_late="min",
    inventory_aggregate_transition_progress=None,
):
    global REWARDS_DICT

    if inventory_aggregate not in ("mean", "min"):
        raise ValueError(
            "inventory_aggregate must be 'mean' or 'min' (min = strict: one bad entity tanks inventory)."
        )
    if relation_duplicate_aggregate not in ("mean", "max"):
        raise ValueError("relation_duplicate_aggregate must be 'mean' or 'max'.")

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
        progress = None
        in_steering_window = False
        eff_box_threshold = float(box_threshold)
        if (
            use_step_box_threshold_schedule
            and debug_sampling_idx is not None
            and debug_time_steps is not None
            and int(debug_time_steps) > 1
        ):
            progress = float(debug_sampling_idx) / float(max(int(debug_time_steps) - 1, 1))
            in_steering_window = float(steering_phase_start_ratio) <= progress <= float(
                steering_phase_end_ratio
            )
            if in_steering_window:
                eff_box_threshold = min(
                    float(eff_box_threshold), float(steering_phase_box_threshold)
                )
        if use_soft_detections and (in_steering_window or use_soft_detections_outside_steering):
            eff_box_threshold = min(float(eff_box_threshold), float(soft_box_threshold))
        processed = _post_process_grounding_dino_detections(
            processor,
            outputs,
            inputs.input_ids,
            target_sizes,
            eff_box_threshold,
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
                    "box_xyxy": (float(x0), float(y0), float(x1), float(y1)),
                }
            )

        relation_scores = []
        relation_scores_by_key = {}
        relation_metas = []
        for rel in prompt_relations:
            subject = _clean_entity(rel["subject"])
            obj = _clean_entity(rel["object"])
            relation = rel["relation"]
            prev_subject_center = None
            prev_object_center = None
            rel_key = None
            if temporal_state is not None:
                rel_key = f"{i}|{subject}|{relation}|{obj}"
                prev = temporal_state.get(rel_key)
                if prev is not None:
                    prev_subject_center = prev.get("subject_center")
                    prev_object_center = prev.get("object_center")

            if use_paired_box_assignment:
                subject_det, object_det = _assign_subject_object_boxes(
                    subject,
                    obj,
                    detections,
                    relation,
                    spatial_tiebreak_weight,
                    min_pair_confidence=(0.0 if use_soft_detections else 0.05),
                    prev_subject_center=prev_subject_center,
                    prev_object_center=prev_object_center,
                    temporal_tiebreak_weight=temporal_tiebreak_weight,
                    temporal_sigma=temporal_sigma,
                )
            else:
                subject_det = _match_best_box(detections, subject)
                object_det = _match_best_box(detections, obj)
            if subject_det is None or object_det is None:
                if use_soft_detections:
                    s_presence = _soft_entity_presence(detections, subject)
                    o_presence = _soft_entity_presence(detections, obj)
                    pair_presence = min(s_presence, o_presence)
                    soft_rel = float(soft_missing_box_penalty) * (1.0 - pair_presence)
                    relation_scores.append(soft_rel)
                    relation_scores_by_key.setdefault((subject, relation, obj), []).append(
                        soft_rel
                    )
                else:
                    relation_scores.append(missing_box_penalty)
                    relation_scores_by_key.setdefault((subject, relation, obj), []).append(
                        missing_box_penalty
                    )
                continue

            dx = subject_det["center"][0] - object_det["center"][0]
            dy = subject_det["center"][1] - object_det["center"][1]
            if relation_scoring_mode == "gaussian":
                rel_score = _gaussian_relation_score(
                    dx=dx, dy=dy, relation=relation, sigma=relation_gaussian_sigma
                )
            else:
                rel_score = _score_relation(
                    dx=torch.tensor(dx),
                    dy=torch.tensor(dy),
                    relation=relation,
                    align_scale=align_scale,
                )

            if relation_targets is not None:
                target = relation_targets.get(relation) if isinstance(relation_targets, dict) else None
                if target is not None:
                    tx = float(target.get("dx_target", 0.0))
                    ty = float(target.get("dy_target", 0.0))
                    ts = float(target.get("sigma", relation_gaussian_sigma))
                    ts = max(ts, 1e-3)
                    # Keep directional plateau semantics with custom targets too.
                    if relation == "left_of":
                        directional = 1.0 if float(dx) <= tx else float(
                            np.exp(-((float(dx) - tx) ** 2) / (2.0 * ts**2))
                        )
                        orth = float(np.exp(-((float(dy) - ty) ** 2) / (2.0 * (1.5 * ts) ** 2)))
                        rel_score = directional * orth
                    elif relation == "right_of":
                        directional = 1.0 if float(dx) >= tx else float(
                            np.exp(-((float(dx) - tx) ** 2) / (2.0 * ts**2))
                        )
                        orth = float(np.exp(-((float(dy) - ty) ** 2) / (2.0 * (1.5 * ts) ** 2)))
                        rel_score = directional * orth
                    elif relation == "on_top_of":
                        directional = 1.0 if float(dy) >= ty else float(
                            np.exp(-((float(dy) - ty) ** 2) / (2.0 * ts**2))
                        )
                        orth = float(np.exp(-((float(dx) - tx) ** 2) / (2.0 * (1.5 * ts) ** 2)))
                        rel_score = directional * orth
                    elif relation == "below":
                        directional = 1.0 if float(dy) <= ty else float(
                            np.exp(-((float(dy) - ty) ** 2) / (2.0 * ts**2))
                        )
                        orth = float(np.exp(-((float(dx) - tx) ** 2) / (2.0 * (1.5 * ts) ** 2)))
                        rel_score = directional * orth
                    else:
                        dist2 = (float(dx) - tx) ** 2 + (float(dy) - ty) ** 2
                        rel_score = float(np.exp(-dist2 / (2.0 * ts**2)))

            temporal_consistency = 1.0
            if prev_subject_center is not None and prev_object_center is not None:
                sdist2 = (subject_det["center"][0] - prev_subject_center[0]) ** 2 + (
                    subject_det["center"][1] - prev_subject_center[1]
                ) ** 2
                odist2 = (object_det["center"][0] - prev_object_center[0]) ** 2 + (
                    object_det["center"][1] - prev_object_center[1]
                ) ** 2
                temporal_consistency = float(
                    np.exp(-(sdist2 + odist2) / (2.0 * max(float(temporal_sigma), 1e-3) ** 2))
                )
                rel_score = (1.0 - float(temporal_consistency_weight)) * float(rel_score) + float(
                    temporal_consistency_weight
                ) * float(rel_score) * temporal_consistency
            relation_scores.append(float(rel_score))
            relation_scores_by_key.setdefault((subject, relation, obj), []).append(
                float(rel_score)
            )
            relation_metas.append((rel_key, subject_det, object_det))

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
            if use_topk_inventory:
                object_count_scores.append(
                    _score_object_count_inventory_topk(
                        detections,
                        obj_name,
                        exp_count,
                        extra_penalty_conf_threshold=topk_extra_penalty_conf_threshold,
                    )
                )
            elif use_soft_detections and use_max_entity_presence:
                # Continuous proxy before objects are crisply formed.
                soft_count = _soft_entity_count(detections, obj_name)
                object_count_scores.append(
                    _score_object_count_inventory(soft_count, exp_count)
                )
            else:
                obs_count = _count_matching_detections(detections, obj_name)
                object_count_scores.append(
                    _score_object_count_inventory(obs_count, exp_count)
                )

        if relation_scores_by_key:
            grouped_relation_scores = []
            for scores in relation_scores_by_key.values():
                if relation_duplicate_aggregate == "max":
                    grouped_relation_scores.append(float(max(scores)))
                else:
                    grouped_relation_scores.append(float(np.mean(scores)))
            relation_component = float(np.mean(grouped_relation_scores))
        else:
            relation_component = float(np.mean(relation_scores)) if relation_scores else 0.0

        current_inventory_aggregate = inventory_aggregate
        if dynamic_inventory_aggregate:
            inv_switch = (
                float(stage_transition_progress)
                if inventory_aggregate_transition_progress is None
                else float(inventory_aggregate_transition_progress)
            )
            if progress is not None and progress < inv_switch:
                current_inventory_aggregate = inventory_aggregate_early
            else:
                current_inventory_aggregate = inventory_aggregate_late

        if object_count_scores:
            if current_inventory_aggregate == "min":
                object_count_component = float(min(object_count_scores))
            else:
                object_count_component = float(np.mean(object_count_scores))
        else:
            object_count_component = 0.0
        eff_relation_weight = relation_weight
        eff_object_weight = object_count_weight
        if dynamic_stage_weighting:
            eff_relation_weight, eff_object_weight = _stage_weighting(
                relation_weight,
                object_count_weight,
                sampling_idx=debug_sampling_idx,
                total_time_steps=debug_time_steps,
                transition_progress=stage_transition_progress,
                sharpness=stage_sharpness,
                early_object_phase_fraction=stage_early_object_fraction,
                early_object_boost=stage_early_object_boost,
            )

        final_reward = (
            eff_relation_weight * relation_component
            + eff_object_weight * object_count_component
        )
        rewards.append(final_reward)
        if temporal_state is not None:
            for rel_key, subject_det, object_det in relation_metas:
                if rel_key is None:
                    continue
                temporal_state[rel_key] = {
                    "subject_center": tuple(subject_det["center"]),
                    "object_center": tuple(object_det["center"]),
                    "sampling_idx": int(debug_sampling_idx)
                    if debug_sampling_idx is not None
                    else None,
                }

        if debug_overlay_dir is not None and debug_sampling_idx is not None:
            rtag = f"{final_reward:+.5f}".replace("+", "p").replace("-", "m")
            out_path = os.path.join(
                debug_overlay_dir,
                f"step_{int(debug_sampling_idx):04d}_particle_{i:02d}_r{rtag}.png",
            )
            _save_grounding_overlay_pil(
                image,
                detections,
                out_path,
                title_lines=[
                    f"sampling_idx={debug_sampling_idx} particle={i}",
                    f"reward={final_reward:.5f} (rel={relation_component:.3f} inv={object_count_component:.3f})",
                    f"weights rel={eff_relation_weight:.2f} inv={eff_object_weight:.2f}",
                    prompts[i][:90],
                ],
            )

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


def _parse_moondream_integer(text: str, lo: int = -50, hi: int = 50):
    """Extract an integer rating in [lo, hi] from model text."""
    if not text:
        return None
    candidates = []
    for m in re.finditer(r"-?\d+", text.replace(",", "")):
        try:
            candidates.append(int(m.group()))
        except ValueError:
            continue
    for n in candidates:
        if lo <= n <= hi:
            return n
    if candidates:
        return max(lo, min(hi, candidates[0]))
    return None


def _parse_rating_number(text: str, lo: float = -50.0, hi: float = 50.0):
    """Extract a numeric rating (int/float) in [lo, hi] from model text."""
    if not text:
        return None
    cleaned = text.replace(",", "")
    candidates = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", cleaned):
        try:
            candidates.append(float(m.group()))
        except ValueError:
            continue
    for n in candidates:
        if float(lo) <= n <= float(hi):
            return float(n)
    if candidates:
        return float(max(float(lo), min(float(hi), candidates[0])))
    return None


def _normalize_rating_to_unit(raw: int, lo: int, hi: int) -> float:
    """Map [lo, hi] linearly to [-1, 1]."""
    if hi == lo:
        return 0.0
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return float((raw - mid) / max(half, 1e-8))


def _save_moondream_debug_pil(image_pil, out_path, *, title_lines=None):
    """Save image with VLM debug text overlay."""
    from PIL import ImageDraw, ImageFont

    img = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if title_lines:
        y = 4
        for line in title_lines[:8]:
            draw.text((4, y), str(line)[:120], fill=(255, 255, 255), font=font)
            draw.text((5, y + 1), str(line)[:120], fill=(0, 0, 0), font=font)
            y += 12
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    img.save(out_path)


def _import_moondream_hf():
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "MoonDreamStyle requires `transformers` with Moondream remote code. "
            "Install: pip install 'transformers>=4.51' accelerate"
        ) from e
    return AutoModelForCausalLM


def _ensure_transformers_tied_weights_compat():
    """
    Compatibility shim for some remote-code models (including older HF Moondream
    wrappers) on newer Transformers runtimes that expect `all_tied_weights_keys`.
    """
    try:
        import transformers.modeling_utils as _mu
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    # Ensure `all_tied_weights_keys` is dict-like (supports `.keys()`), since
    # some remote-code models expose list/tuple and newer transformers expects dict-like.
    def _normalize_tied_keys_dict(keys):
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, (list, tuple, set)):
            return {str(k): None for k in keys}
        return {str(keys): None}

    @property
    def _all_tied_weights_keys_compat(self):
        override = getattr(self, "_all_tied_weights_keys_override", None)
        if override is not None:
            return _normalize_tied_keys_dict(override)
        keys = getattr(self, "_tied_weights_keys", None)
        return _normalize_tied_keys_dict(keys)

    @_all_tied_weights_keys_compat.setter
    def _all_tied_weights_keys_compat(self, value):
        # transformers post_init assigns into this field in newer versions.
        self._all_tied_weights_keys_override = _normalize_tied_keys_dict(value)

    PreTrainedModel.all_tied_weights_keys = _all_tied_weights_keys_compat

    # Some remote-code models expose `_tied_weights_keys` as a list, while some
    # transformers versions expect a dict-like object with `.keys()`.
    # Patch helper to gracefully support either type.
    fn = getattr(_mu, "_get_tied_weight_keys", None)
    if fn is None:
        return
    if getattr(fn, "_moondream_list_compat", False):
        return

    def _patched_get_tied_weight_keys(module, prefix=""):
        tied_keys = set()
        tied_weight_keys = getattr(module, "_tied_weights_keys", None)
        if tied_weight_keys is not None:
            if isinstance(tied_weight_keys, dict):
                names = tied_weight_keys.keys()
            elif isinstance(tied_weight_keys, (list, tuple, set)):
                names = tied_weight_keys
            else:
                names = [str(tied_weight_keys)]
            for name in names:
                tied_keys.add(prefix + str(name))
        for name, submodule in module.named_children():
            tied_keys |= _patched_get_tied_weight_keys(submodule, prefix=prefix + name + ".")
        return tied_keys

    _patched_get_tied_weight_keys._moondream_list_compat = True
    _mu._get_tied_weight_keys = _patched_get_tied_weight_keys

    # Patch mark_tied_weights_as_initialized to accept list-like tied keys too.
    _orig_mark = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
    if _orig_mark is not None and not getattr(_orig_mark, "_moondream_list_compat", False):
        def _patched_mark_tied_weights_as_initialized(self, loading_info):
            tied = getattr(self, "all_tied_weights_keys", {})
            if isinstance(tied, dict):
                tied_params = tied.keys()
            elif isinstance(tied, (list, tuple, set)):
                tied_params = tied
            elif tied is None:
                tied_params = []
            else:
                tied_params = [tied]
            for tied_param in tied_params:
                try:
                    param = self.get_parameter(str(tied_param))
                    param._is_hf_initialized = True
                except Exception:
                    continue
        _patched_mark_tied_weights_as_initialized._moondream_list_compat = True
        PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights_as_initialized


def _ensure_torch_sdpa_enable_gqa_compat():
    """
    Moondream3 remote code may call torch SDPA with `enable_gqa=...`.
    Older torch versions don't expose this kwarg. Ignore it when unsupported.
    """
    try:
        import inspect as _inspect
        import torch.nn.functional as _F
    except Exception:
        return

    fn = getattr(_F, "scaled_dot_product_attention", None)
    if fn is None:
        return
    try:
        has_enable_gqa = "enable_gqa" in _inspect.signature(fn).parameters
    except Exception:
        # On some torch builds signature introspection fails for C-implemented funcs.
        # Be conservative and patch in that case.
        has_enable_gqa = False
    if has_enable_gqa or getattr(fn, "_moondream_enable_gqa_compat", False):
        return

    def _patched_sdpa(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return fn(*args, **kwargs)

    _patched_sdpa._moondream_enable_gqa_compat = True
    _F.scaled_dot_product_attention = _patched_sdpa


def _import_moondream_sdk():
    try:
        import moondream as md
    except ImportError as e:
        raise ImportError(
            "MoonDreamStyle backend='sdk' requires `moondream`. Install: pip install moondream"
        ) from e
    return md


def _import_qwen3_vl_hf():
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "Qwen3VLStyle requires a recent `transformers` with vision-language "
            "generation support. Install: pip install -U 'transformers>=4.51' accelerate"
        ) from e
    return AutoModelForImageTextToText, AutoProcessor


def _qwen3_vl_query_text(
    style_target: str,
    rating_min: int,
    rating_max: int,
):
    return (
        f"Evaluate how strongly this image matches a '{style_target}' visual style. "
        f"Return one integer between {rating_min} and {rating_max} only, where "
        f"{rating_max} means extremely strong {style_target} style and "
        f"{rating_min} means not {style_target} at all. "
        "Output only the integer with no extra text."
    )


def do_qwen3_vl_style_reward(
    *,
    images,
    prompts,
    qwen_model_name="Qwen/Qwen3-VL-2B-Thinking",
    qwen_hf_device=None,
    qwen_hf_device_map=None,
    qwen_hf_dtype=None,
    qwen_hf_max_memory=None,
    qwen_hf_offload_folder=None,
    qwen_hf_low_cpu_mem_usage=True,
    qwen_tp_plan=None,
    qwen_attn_implementation=None,
    qwen_trust_remote_code=True,
    qwen_force_reload=False,
    hf_revision=None,
    style_target="comic-book",
    query_text=None,
    include_prompt_context=True,
    rating_min=-50,
    rating_max=50,
    query_max_tokens=24,
    qwen_do_sample=False,
    qwen_temperature=0.0,
    qwen_top_p=1.0,
    qwen_debug_print=False,
    qwen_log_path=None,
    qwen_log_to_output=True,
    qwen_retry_on_parse_fail=True,
    qwen_retry_max_new_tokens=8,
    warn_parse_failures=True,
    debug_overlay_dir=None,
    debug_sampling_idx=None,
    **_unused_kwargs,
):
    """
    Qwen3-VL style rating reward:
    - asks for one integer in [rating_min, rating_max]
    - linearly rescales to [-1, 1] for FK steering
    """
    global REWARDS_DICT

    if rating_min >= rating_max:
        raise ValueError("rating_min must be < rating_max")
    if query_text is None:
        query_text = _qwen3_vl_query_text(
            style_target=style_target,
            rating_min=int(rating_min),
            rating_max=int(rating_max),
        )

    dtype = qwen_hf_dtype
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    target_device = (
        qwen_hf_device
        if qwen_hf_device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    load_kw = {}
    if hf_revision is not None:
        load_kw["revision"] = hf_revision
    device_map = qwen_hf_device_map
    if device_map is None:
        device_map = None
    # Reload when model-loading parameters change so notebook config edits are honored.
    cache_key = (
        str(qwen_model_name),
        str(hf_revision),
        str(dtype),
        str(target_device),
        repr(device_map),
        repr(qwen_hf_max_memory),
        str(qwen_hf_offload_folder),
        bool(qwen_hf_low_cpu_mem_usage),
        str(qwen_tp_plan),
        str(qwen_attn_implementation),
        bool(qwen_trust_remote_code),
    )

    cached = REWARDS_DICT["Qwen3VLStyle"]
    needs_reload = (
        bool(qwen_force_reload)
        or cached is None
        or cached.get("cache_key") != cache_key
    )
    if needs_reload:
        AutoModelForImageTextToText, AutoProcessor = _import_qwen3_vl_hf()

        try:
            model_load_kw = dict(load_kw)
            model_load_kw["device_map"] = device_map
            model_load_kw["trust_remote_code"] = bool(qwen_trust_remote_code)
            model_load_kw["low_cpu_mem_usage"] = bool(qwen_hf_low_cpu_mem_usage)
            if qwen_hf_max_memory is not None:
                model_load_kw["max_memory"] = qwen_hf_max_memory
            if qwen_hf_offload_folder is not None:
                model_load_kw["offload_folder"] = qwen_hf_offload_folder
            if qwen_tp_plan is not None:
                model_load_kw["tp_plan"] = qwen_tp_plan
            if qwen_attn_implementation is not None:
                model_load_kw["attn_implementation"] = qwen_attn_implementation
            model = AutoModelForImageTextToText.from_pretrained(
                qwen_model_name,
                torch_dtype=dtype,
                **model_load_kw,
            )
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(
                qwen_model_name,
                dtype=dtype,
                **model_load_kw,
            )
        if device_map is None:
            model = model.to(target_device)
        model.eval()
        processor = AutoProcessor.from_pretrained(qwen_model_name, **load_kw)
        cache_entry = {
            "model_name": qwen_model_name,
            "model": model,
            "processor": processor,
            "device": target_device,
            "device_map": device_map,
            "cache_key": cache_key,
        }
        REWARDS_DICT["Qwen3VLStyle"] = cache_entry

    cache_entry = REWARDS_DICT["Qwen3VLStyle"]
    model = cache_entry["model"]
    processor = cache_entry["processor"]
    target_device = cache_entry["device"]
    using_device_map = cache_entry["device_map"] is not None

    effective_qwen_log_path = qwen_log_path
    if effective_qwen_log_path is None and qwen_log_to_output:
        # Default log target in project output folder.
        effective_qwen_log_path = os.path.join("output", "qwen_intermediate_logs.jsonl")

    rewards = []
    for i, image in enumerate(images):
        per_image_query = query_text
        if include_prompt_context and i < len(prompts):
            per_image_query = (
                f"{query_text}\n"
                f"Generation prompt context: {prompts[i]}\n"
                "Score only the rendered visual style in the image."
            )
        messages = [
            {
                "role": "user",
                # Include the concrete image payload in chat content for processors
                # that rely on the structured multimodal template.
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": per_image_query},
                ],
            }
        ]
        prompt_for_model = per_image_query
        if hasattr(processor, "apply_chat_template"):
            try:
                prompt_for_model = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_for_model = per_image_query

        inputs = processor(
            text=[prompt_for_model],
            images=[image],
            return_tensors="pt",
        )
        if not using_device_map:
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(query_max_tokens),
                do_sample=bool(qwen_do_sample),
                temperature=float(qwen_temperature),
                top_p=float(qwen_top_p),
            )
        answer = ""
        if "input_ids" in inputs and inputs["input_ids"] is not None:
            generated_ids = generated[:, inputs["input_ids"].shape[-1] :]
            answer = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip()
        if not answer:
            # Fallback decode path for processor/model variants where slicing with
            # prompt length yields empty text.
            answer = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip()

        raw = _parse_rating_number(answer, float(rating_min), float(rating_max))
        retried = False
        if raw is None and qwen_retry_on_parse_fail:
            retried = True
            strict_query = (
                f"Return only one integer in [{int(rating_min)}, {int(rating_max)}] "
                f"for how '{style_target}' the image style is. "
                "No reasoning. No words. Integer only."
            )
            strict_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": strict_query},
                    ],
                }
            ]
            strict_prompt = strict_query
            if hasattr(processor, "apply_chat_template"):
                try:
                    strict_prompt = processor.apply_chat_template(
                        strict_messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    strict_prompt = strict_query
            strict_inputs = processor(
                text=[strict_prompt],
                images=[image],
                return_tensors="pt",
            )
            if not using_device_map:
                strict_inputs = {k: v.to(target_device) for k, v in strict_inputs.items()}
            with torch.no_grad():
                strict_generated = model.generate(
                    **strict_inputs,
                    max_new_tokens=int(qwen_retry_max_new_tokens),
                    do_sample=False,
                )
            strict_answer = ""
            if "input_ids" in strict_inputs and strict_inputs["input_ids"] is not None:
                strict_ids = strict_generated[:, strict_inputs["input_ids"].shape[-1] :]
                strict_answer = processor.batch_decode(
                    strict_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0].strip()
            if not strict_answer:
                strict_answer = processor.batch_decode(
                    strict_generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0].strip()
            # Keep whichever response parses; prefer strict retry when available.
            strict_raw = _parse_rating_number(
                strict_answer, float(rating_min), float(rating_max)
            )
            if strict_raw is not None:
                answer = strict_answer
                raw = strict_raw

        if qwen_debug_print:
            print(
                "[Qwen3VLStyle] "
                f"idx={i} style={style_target!r} retried={retried} "
                f"raw_answer={answer[:220]!r} parsed={raw}"
            )
        if raw is None:
            if warn_parse_failures:
                warnings.warn(
                    f"Qwen3VLStyle: could not parse integer in [{rating_min},{rating_max}] "
                    f"from answer={answer!r}; using 0.0 reward.",
                    UserWarning,
                    stacklevel=2,
                )
            reward = 0.0
        else:
            reward = _normalize_rating_to_unit(
                float(raw), float(rating_min), float(rating_max)
            )
        rewards.append(float(reward))

        if effective_qwen_log_path:
            log_parent = os.path.dirname(effective_qwen_log_path)
            if log_parent:
                os.makedirs(log_parent, exist_ok=True)
            rec = {
                "sampling_idx": int(debug_sampling_idx)
                if debug_sampling_idx is not None
                else None,
                "particle_idx": int(i),
                "style_target": str(style_target),
                "query_text": str(per_image_query),
                "raw_answer": str(answer),
                "parsed_rating": float(raw) if raw is not None else None,
                "reward": float(reward),
                "retried_on_parse_fail": bool(retried),
                "rating_min": int(rating_min),
                "rating_max": int(rating_max),
            }
            with open(effective_qwen_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

        if debug_overlay_dir is not None and debug_sampling_idx is not None:
            rtag = f"{reward:+.5f}".replace("+", "p").replace("-", "m")
            out_path = os.path.join(
                debug_overlay_dir,
                f"step_{int(debug_sampling_idx):04d}_particle_{i:02d}_r{rtag}.png",
            )
            raw_disp = raw if raw is not None else "?"
            _save_moondream_debug_pil(
                image,
                out_path,
                title_lines=[
                    f"sampling_idx={debug_sampling_idx} particle={i}",
                    f"raw={raw_disp} reward[-1,1]={reward:.5f}",
                    f"answer: {answer[:200]}",
                    (prompts[i] if i < len(prompts) else "")[:90],
                ],
            )

    return rewards


def do_moondream_style_reward(
    *,
    images,
    prompts,
    moondream_backend="hf",
    moondream_model_name="vikhyatk/moondream2",
    moondream_api_key=None,
    moondream_local=False,
    moondream_hf_device=None,
    moondream_hf_device_map=None,
    moondream_hf_fallback_model="vikhyatk/moondream2",
    moondream_hf_allow_cpu_fallback=False,
    hf_revision=None,
    query_text=None,
    rating_min=-50,
    rating_max=50,
    query_temperature=0.2,
    query_top_p=0.9,
    query_max_tokens=64,
    warn_parse_failures=True,
    debug_overlay_dir=None,
    debug_sampling_idx=None,
    **_unused_kwargs,
):
    """
    Moondream VLM style rating: asks for an integer in [rating_min, rating_max], then maps to [-1, 1].

    Uses Moondream SDK by default; HF backend is available as fallback.
    """
    global REWARDS_DICT

    # Apply once per call for robustness in hot-reload / cached-model notebook flows.
    if moondream_backend == "hf":
        _ensure_torch_sdpa_enable_gqa_compat()

    if rating_min >= rating_max:
        raise ValueError("rating_min must be < rating_max")

    if query_text is None:
        query_text = (
            "On a scale from -50 to 50, rate how close the style of the current image is to a comic-book style. "
            "Respond with only one integer."
        )

    if moondream_backend not in ("sdk", "hf"):
        raise ValueError("moondream_backend must be 'sdk' or 'hf'")

    cached = REWARDS_DICT["MoonDreamStyle"]
    if cached is None or cached.get("backend") != moondream_backend:
        if moondream_backend == "sdk":
            md = _import_moondream_sdk()
            key = moondream_api_key or os.getenv("MOONDREAM_API_KEY")
            init_kw = {}
            if key:
                init_kw["api_key"] = key
            if moondream_local:
                init_kw["local"] = True
            try:
                model = md.vl(**init_kw)
            except TypeError:
                # Older/newer SDK parameter mismatch fallback.
                if "local" in init_kw:
                    init_kw.pop("local", None)
                model = md.vl(**init_kw)
            REWARDS_DICT["MoonDreamStyle"] = {"backend": "sdk", "model": model}
        else:
            AutoModelForCausalLM = _import_moondream_hf()
            _ensure_transformers_tied_weights_compat()
            _ensure_torch_sdpa_enable_gqa_compat()
            load_kw = dict(trust_remote_code=True)
            if hf_revision is not None:
                load_kw["revision"] = hf_revision
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            target_device = (
                moondream_hf_device
                if moondream_hf_device is not None
                else ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            # Default to single-device to avoid cross-GPU encode/query mismatches.
            device_map = moondream_hf_device_map
            if device_map is None:
                device_map = None

            load_model_name = moondream_model_name
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    load_model_name,
                    dtype=dtype,
                    device_map=device_map,
                    **load_kw,
                )
            except TypeError:
                # Some versions still expect torch_dtype.
                model = AutoModelForCausalLM.from_pretrained(
                    load_model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    **load_kw,
                )
            except ModuleNotFoundError as e:
                if (
                    "torch.nn.attention.flex_attention" in str(e)
                    and moondream_hf_fallback_model
                    and moondream_hf_fallback_model != moondream_model_name
                ):
                    warnings.warn(
                        "MoonDreamStyle(hf): current torch build lacks flex_attention "
                        f"required by {moondream_model_name}. Falling back to "
                        f"{moondream_hf_fallback_model}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    load_model_name = moondream_hf_fallback_model
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            load_model_name,
                            dtype=dtype,
                            device_map=device_map,
                            **load_kw,
                        )
                    except TypeError:
                        model = AutoModelForCausalLM.from_pretrained(
                            load_model_name,
                            torch_dtype=dtype,
                            device_map=device_map,
                            **load_kw,
                        )
                else:
                    raise
            except AttributeError as e:
                # Compatibility fallback for tied-weights metadata shape mismatches
                # seen in some moondream remote-code + transformers combinations.
                if "list" in str(e) and "keys" in str(e):
                    warnings.warn(
                        "MoonDreamStyle(hf): hit tied-weights compatibility path; "
                        "retrying load with low_cpu_mem_usage=False and no device_map.",
                        UserWarning,
                        stacklevel=2,
                    )
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            load_model_name,
                            dtype=dtype,
                            device_map=None,
                            low_cpu_mem_usage=False,
                            **load_kw,
                        )
                    except TypeError:
                        model = AutoModelForCausalLM.from_pretrained(
                            load_model_name,
                            torch_dtype=dtype,
                            device_map=None,
                            low_cpu_mem_usage=False,
                            **load_kw,
                        )
                    model = model.to(target_device)
                else:
                    raise
            else:
                # If loaded without dispatch map, explicitly place model on one device.
                if device_map is None:
                    model = model.to(target_device)
            model.eval()
            REWARDS_DICT["MoonDreamStyle"] = {"backend": "hf", "model": model}

    model = REWARDS_DICT["MoonDreamStyle"]["model"]
    backend = REWARDS_DICT["MoonDreamStyle"]["backend"]
    settings = {
        "temperature": float(query_temperature),
        "top_p": float(query_top_p),
        "max_tokens": int(query_max_tokens),
    }

    rewards = []
    for i, image in enumerate(images):
        if backend == "sdk":
            try:
                out = model.query(image, query_text)
            except TypeError:
                out = model.query(image, query_text, stream=False)
        else:
            with torch.no_grad():
                enc = model.encode_image(image)
                try:
                    out = model.query(enc, query_text, settings=settings)
                except TypeError:
                    out = model.query(enc, query_text)
                except AttributeError as e:
                    # Some moondream3 + torch combinations fail at runtime due to
                    # FlexAttention BlockMask API differences (e.g. missing seq_lengths).
                    if (
                        "BlockMask" in str(e)
                        and "seq_lengths" in str(e)
                        and moondream_hf_fallback_model
                        and moondream_hf_fallback_model != moondream_model_name
                    ):
                        warnings.warn(
                            "MoonDreamStyle(hf): moondream3 BlockMask API mismatch at query time; "
                            f"switching to fallback model {moondream_hf_fallback_model}.",
                            UserWarning,
                            stacklevel=2,
                        )
                        AutoModelForCausalLM = _import_moondream_hf()
                        _ensure_transformers_tied_weights_compat()
                        _ensure_torch_sdpa_enable_gqa_compat()
                        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        load_kw = dict(trust_remote_code=True)
                        if hf_revision is not None:
                            load_kw["revision"] = hf_revision
                        try:
                            fallback_model = AutoModelForCausalLM.from_pretrained(
                                moondream_hf_fallback_model,
                                dtype=dtype,
                                device_map=moondream_hf_device_map,
                                **load_kw,
                            )
                        except TypeError:
                            fallback_model = AutoModelForCausalLM.from_pretrained(
                                moondream_hf_fallback_model,
                                torch_dtype=dtype,
                                device_map=moondream_hf_device_map,
                                **load_kw,
                            )
                        if moondream_hf_device_map is None:
                            target_device = (
                                moondream_hf_device
                                if moondream_hf_device is not None
                                else ("cuda:0" if torch.cuda.is_available() else "cpu")
                            )
                            fallback_model = fallback_model.to(target_device)
                        fallback_model.eval()
                        REWARDS_DICT["MoonDreamStyle"] = {
                            "backend": "hf",
                            "model": fallback_model,
                        }
                        model = fallback_model
                        enc = model.encode_image(image)
                        try:
                            out = model.query(enc, query_text, settings=settings)
                        except TypeError:
                            out = model.query(enc, query_text)
                    else:
                        raise
                except RuntimeError as e:
                    msg = str(e).lower()
                    if (
                        "probability tensor contains either `inf`, `nan`" in msg
                        or "probability tensor contains either inf, nan" in msg
                    ):
                        safe_settings = {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "max_tokens": int(min(query_max_tokens, 24)),
                        }
                        warnings.warn(
                            "MoonDreamStyle(hf): NaN/Inf sampling probs; retrying "
                            "query with safer sampling settings.",
                            UserWarning,
                            stacklevel=2,
                        )
                        try:
                            out = model.query(enc, query_text, settings=safe_settings)
                        except TypeError:
                            out = model.query(enc, query_text)
                    elif (
                        "device-side assert triggered" in msg
                        or "cuda error" in msg
                        or "multinomial" in msg
                    ):
                        warnings.warn(
                            "MoonDreamStyle(hf): CUDA sampling assert; retrying "
                            "query with safer settings, then CPU fallback if needed.",
                            UserWarning,
                            stacklevel=2,
                        )
                        safe_settings = {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "max_tokens": int(min(query_max_tokens, 24)),
                        }
                        try:
                            out = model.query(enc, query_text, settings=safe_settings)
                        except Exception:
                            if moondream_hf_allow_cpu_fallback:
                                # Optional last resort: run the Moondream query on CPU for stability.
                                model = model.to("cpu")
                                REWARDS_DICT["MoonDreamStyle"]["model"] = model
                                enc_cpu = model.encode_image(image)
                                try:
                                    out = model.query(enc_cpu, query_text, settings=safe_settings)
                                except TypeError:
                                    out = model.query(enc_cpu, query_text)
                            else:
                                raise RuntimeError(
                                    "MoonDreamStyle(hf): CUDA query failed and CPU fallback is disabled "
                                    "(moondream_hf_allow_cpu_fallback=False). "
                                    "Try safer settings, reduce query_max_tokens, or enable CPU fallback explicitly."
                                )
                    else:
                        raise
        if isinstance(out, dict):
            answer = out.get("answer", "")
        else:
            answer = str(out)
        raw = _parse_moondream_integer(answer, int(rating_min), int(rating_max))
        if raw is None:
            if warn_parse_failures:
                warnings.warn(
                    f"MoonDreamStyle: could not parse integer in [{rating_min},{rating_max}] "
                    f"from answer={answer!r}; using 0.0 reward.",
                    UserWarning,
                    stacklevel=2,
                )
            reward = 0.0
        else:
            reward = _normalize_rating_to_unit(raw, int(rating_min), int(rating_max))
        rewards.append(float(reward))

        if debug_overlay_dir is not None and debug_sampling_idx is not None:
            rtag = f"{reward:+.5f}".replace("+", "p").replace("-", "m")
            out_path = os.path.join(
                debug_overlay_dir,
                f"step_{int(debug_sampling_idx):04d}_particle_{i:02d}_r{rtag}.png",
            )
            raw_disp = raw if raw is not None else "?"
            _save_moondream_debug_pil(
                image,
                out_path,
                title_lines=[
                    f"sampling_idx={debug_sampling_idx} particle={i}",
                    f"raw={raw_disp} reward[-1,1]={reward:.5f}",
                    f"answer: {answer[:200]}",
                    (prompts[i] if i < len(prompts) else "")[:90],
                ],
            )

    return rewards


# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    _ensure_hpsv2_open_clip_vocab()
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

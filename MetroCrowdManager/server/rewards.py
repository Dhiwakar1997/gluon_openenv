# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rule-based reward functions for the MetroCrowdManager environment.

All 9 reward functions are deterministic heuristics (not LLM-as-judge).
Each returns a float in [0.0, 1.0].

Signature: (response_text, train_crowd, platform_crowd, num_coaches) -> float
"""

import math
import re
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std(values: List[float]) -> float:
    """Compute population standard deviation without numpy."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _extract_announcement(response_text: str) -> str:
    """Extract the announcement text from the structured response."""
    match = re.search(r'Announcement:\s*"([^"]*)"', response_text)
    return match.group(1) if match else response_text


def _parse_distribution(response_text: str) -> List[float] | None:
    """Parse 'Recommended Platform Distribution: [...]' from response."""
    pattern = r"Recommended Platform Distribution\s*:\s*\[([^\]]+)\]"
    match = re.search(pattern, response_text, re.IGNORECASE)
    if not match:
        return None
    try:
        return [float(x.strip().rstrip("%")) for x in match.group(1).split(",")]
    except (ValueError, IndexError):
        return None


def _parse_hex_list(text: str, pattern: str) -> List[str]:
    """Parse a bracketed list of hex color values from text."""
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).split(",")
    return [c.strip().upper() for c in raw]


def _pct_to_hex(pct: float) -> str:
    """Map a crowd percentage to its canonical hex color."""
    if pct <= 40:
        return "#008000"
    elif pct <= 60:
        return "#FFFF00"
    elif pct <= 80:
        return "#FF8C00"
    else:
        return "#FF0000"


_CANONICAL_COLORS = {"#008000", "#FFFF00", "#FF8C00", "#FF0000"}

_NOOP_PHRASES = [
    "already balanced",
    "no movement needed",
    "distribution is even",
    "no changes required",
    "no redirection necessary",
    "current distribution is good",
    "no action needed",
    "evenly distributed",
    "well distributed",
    "no need to move",
    "stay where you are",
    "maintain current positions",
]

_DIRECTIVE_PHRASES = [
    "move to",
    "proceed to",
    "walk towards",
    "shift to",
    "please go to",
    "relocate to",
    "head to",
    "moving to",
    "avoid",
    "spread out",
    "distribute yourselves",
    "consider moving",
    "head towards",
]


def _is_balanced(train_crowd: List[int], platform_crowd: List[int], num_coaches: int) -> bool:
    """Check whether the current crowd state is roughly balanced."""
    if _std([float(p) for p in platform_crowd]) >= 10:
        return False
    if _std([float(t) for t in train_crowd]) >= 15:
        return False
    effective = [(train_crowd[i] + platform_crowd[i]) / 2.0 for i in range(num_coaches)]
    return _std(effective) < 8


# ---------------------------------------------------------------------------
# Reward 1 — Politeness & Kindness
# ---------------------------------------------------------------------------

_POLITE_MARKERS = [
    "please",
    "kindly",
    "we request",
    "thank you",
    "for your comfort",
    "we appreciate",
    "dear passengers",
    "may we suggest",
    "if possible",
    "we recommend",
    "for your convenience",
    "would you mind",
    "we encourage",
    "your cooperation",
    "we suggest",
    "attention passengers",
    "for a comfortable",
    "for your safety",
]

_RUDE_MARKERS = [
    "move now",
    "get out",
    "you must",
    "immediately leave",
    "stupid",
    "idiot",
    "hurry up",
    "don't stand there",
    "what are you doing",
    "move it",
]


def compute_politeness(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    if not response_text.strip():
        return 0.0

    text_lower = response_text.lower()
    polite_count = sum(1 for m in _POLITE_MARKERS if m in text_lower)
    rude_count = sum(1 for m in _RUDE_MARKERS if m in text_lower)

    # Excessive exclamation marks count as rude
    if response_text.count("!") > 3:
        rude_count += 1

    base_score = min(1.0, polite_count / 3.0)
    penalty = 0.3 * rude_count
    return max(0.0, base_score - penalty)


# ---------------------------------------------------------------------------
# Shared helper for math-based rewards
# ---------------------------------------------------------------------------

def _compute_ideal(train_crowd: List[int], platform_crowd: List[int], num_coaches: int) -> List[int]:
    """Compute capacity-weighted ideal distribution with iterative cap redistribution."""
    train_capacity = [100 - train_crowd[i] for i in range(num_coaches)]
    total_capacity = sum(train_capacity)
    if total_capacity == 0:
        return [int(sum(platform_crowd) / num_coaches)] * num_coaches

    total_platform = sum(platform_crowd)
    if total_platform == 0:
        return [0] * num_coaches

    weights = [c / total_capacity for c in train_capacity]
    ideal = [0.0] * num_coaches
    locked = [False] * num_coaches
    remaining = float(total_platform)

    for _ in range(num_coaches):
        unlocked_weight = sum(weights[i] for i in range(num_coaches) if not locked[i])
        if unlocked_weight <= 0:
            break
        for i in range(num_coaches):
            if not locked[i]:
                ideal[i] = remaining * (weights[i] / unlocked_weight)
        changed = False
        for i in range(num_coaches):
            if not locked[i] and ideal[i] > 100.0:
                ideal[i] = 100.0
                locked[i] = True
                remaining -= 100.0
                changed = True
        if not changed:
            break

    return [int(max(0.0, min(100.0, v))) for v in ideal]


# ---------------------------------------------------------------------------
# Reward 2a — Distribution Accuracy (MAE against ideal)
# ---------------------------------------------------------------------------

def compute_distribution_accuracy(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    proposed = _parse_distribution(response_text)

    if proposed is None:
        if _is_balanced(train_crowd, platform_crowd, num_coaches):
            return 0.5
        return 0.0

    if len(proposed) != num_coaches:
        return 0.0

    ideal = _compute_ideal(train_crowd, platform_crowd, num_coaches)

    mae = sum(abs(proposed[i] - ideal[i]) for i in range(num_coaches)) / num_coaches
    score = max(0.0, 1.0 - mae / 30.0)
    # print("distribution_accuracy: ", score)
    # print("train_crowd: ", train_crowd)
    # print("platform_crowd: ", platform_crowd)
    # print("proposed: ", proposed)
    # print("ideal: ", ideal)
    # print("mae: ", mae)
    return score


# ---------------------------------------------------------------------------
# Reward 2b — Conservation Accuracy (total crowd preserved)
# ---------------------------------------------------------------------------

def compute_conservation_accuracy(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    proposed = _parse_distribution(response_text)

    if proposed is None:
        if _is_balanced(train_crowd, platform_crowd, num_coaches):
            return 0.5
        return 0.0

    if len(proposed) != num_coaches:
        return 0.0

    total_proposed = sum(proposed)
    total_platform = sum(platform_crowd)

    if total_platform == 0:
        return 1.0 if total_proposed == 0 else 0.0

    return min(total_proposed, total_platform) / max(total_proposed, total_platform)


# ---------------------------------------------------------------------------
# Reward 2c — Feasibility Accuracy (values in valid range)
# ---------------------------------------------------------------------------

def compute_feasibility_accuracy(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    proposed = _parse_distribution(response_text)

    if proposed is None:
        if _is_balanced(train_crowd, platform_crowd, num_coaches):
            return 0.5
        return 0.0

    if len(proposed) != num_coaches:
        return 0.0

    if any(p < 0 or p > 100 for p in proposed):
        return 0.5

    return 1.0


# ---------------------------------------------------------------------------
# Reward 3 — Color Grading Accuracy
# ---------------------------------------------------------------------------

_PLATFORM_COLOR_PATTERN = r"Platform Zone Color Codes\s*:\s*\[([^\]]+)\]"
_TRAIN_COLOR_PATTERN = r"Train Coach Color Codes\s*:\s*\[([^\]]+)\]"


def _score_color_list(model_colors: List[str], expected: List[str], num_coaches: int) -> float:
    if len(model_colors) != num_coaches:
        return 0.0
    correct = 0
    for i in range(num_coaches):
        mc = model_colors[i]
        # Normalise: strip whitespace, ensure uppercase, ensure leading #
        if not mc.startswith("#"):
            mc = "#" + mc
        mc = mc.upper()
        if mc == expected[i]:
            correct += 1
        elif mc in _CANONICAL_COLORS:
            # Valid color but wrong band — no partial credit
            pass
    return correct / num_coaches


def compute_color_grading(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    expected_platform = [_pct_to_hex(platform_crowd[i]) for i in range(num_coaches)]
    expected_train = [_pct_to_hex(train_crowd[i]) for i in range(num_coaches)]

    model_platform = _parse_hex_list(response_text, _PLATFORM_COLOR_PATTERN)
    model_train = _parse_hex_list(response_text, _TRAIN_COLOR_PATTERN)

    platform_score = _score_color_list(model_platform, expected_platform, num_coaches)
    train_score = _score_color_list(model_train, expected_train, num_coaches)

    return (platform_score + train_score) / 2.0


# ---------------------------------------------------------------------------
# Reward 4 — Language Consistency
# ---------------------------------------------------------------------------

_NON_LATIN_PATTERN = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0600-\u06FF\u4E00-\u9FFF]"
)

_NON_ENGLISH_WORDS = {
    "kripya", "yahan", "wahan", "bheed",
    "chaliye", "rukiye", "jaldi", "dhanyavaad",
    "bitte", "por favor", "gracias",
    "danke", "arigatou",
}


def compute_language_consistency(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    if not response_text.strip():
        return 0.0

    total_chars = len(response_text)
    non_latin_chars = len(_NON_LATIN_PATTERN.findall(response_text))

    words = response_text.lower().split()
    non_english_count = sum(1 for w in words if w in _NON_ENGLISH_WORDS)

    non_english_ratio = (non_latin_chars + non_english_count * 5) / total_chars
    if non_english_ratio < 0.01:
        return 1.0
    return max(0.0, 1.0 - non_english_ratio * 10)


# ---------------------------------------------------------------------------
# Reward 5 — No-Op Detection
# ---------------------------------------------------------------------------

def compute_noop_detection(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    balanced = _is_balanced(train_crowd, platform_crowd, num_coaches)
    text_lower = response_text.lower()

    has_noop = any(p in text_lower for p in _NOOP_PHRASES)
    has_directive = any(p in text_lower for p in _DIRECTIVE_PHRASES)

    if balanced:
        if has_noop and not has_directive:
            return 1.0
        if has_directive and not has_noop:
            return 0.0
        return 0.3  # ambiguous
    else:
        if has_directive and not has_noop:
            return 1.0
        if has_noop and not has_directive:
            return 0.0
        return 0.5  # partial


# ---------------------------------------------------------------------------
# Reward 6 — Clarity
# ---------------------------------------------------------------------------

_JARGON_TERMS = [
    "redistribution",
    "optimization",
    "algorithm",
    "percentile",
    "standard deviation",
    "equilibrium",
    "load balancing",
    "throughput",
    "utilization",
    "coefficient",
    "parameter",
    "infrastructure",
    "congestion index",
    "density metric",
    "probabilistic",
    "stochastic",
    "heuristic",
]


def compute_clarity(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    announcement = _extract_announcement(response_text)
    if not announcement.strip():
        return 0.0

    # Component A: Sentence length (30%)
    sentences = [s.strip() for s in re.split(r"[.!?]+", announcement) if s.strip()]
    if not sentences:
        return 0.0

    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_words <= 15:
        length_score = 1.0
    elif avg_words <= 25:
        length_score = 0.7
    else:
        length_score = max(0.0, 1.0 - (avg_words - 25) * 0.05)

    # Component B: Jargon penalty (30%)
    text_lower = announcement.lower()
    jargon_count = sum(1 for t in _JARGON_TERMS if t in text_lower)
    jargon_score = max(0.0, 1.0 - jargon_count * 0.25)

    # Component C: Structure bonus (30%)
    has_numbered = bool(re.search(r"\d+[.)]\s", announcement))
    has_bullets = bool(re.search(r"[-*\u2022]\s", announcement))
    has_headers = bool(re.search(r"(?:Coaches?|Zones?)\s*[A-J]\s*:", announcement))
    structure_features = sum([has_numbered, has_bullets, has_headers])
    structure_score = min(1.0, structure_features * 0.5)

    # Component D: Minimum length (10%)
    length_ok = 1.0 if len(announcement) >= 50 else len(announcement) / 50.0

    return 0.3 * length_score + 0.3 * jargon_score + 0.3 * structure_score + 0.1 * length_ok


# ---------------------------------------------------------------------------
# Reward 7 — Sequential Direction
# ---------------------------------------------------------------------------

_SEQUENTIAL_MARKERS = [
    "first",
    "then",
    "next",
    "after that",
    "following that",
    "subsequently",
    "step 1",
    "step 2",
    "step 3",
]

_SIMULTANEOUS_MARKERS = [
    "everyone move",
    "all passengers",
    "at the same time",
    "simultaneously",
    "all at once",
    "everyone should",
    "all of you",
]


def compute_sequential_direction(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    announcement = _extract_announcement(response_text)
    balanced = _is_balanced(train_crowd, platform_crowd, num_coaches)
    text_lower = announcement.lower()

    has_noop = any(p in text_lower for p in _NOOP_PHRASES)
    has_directive = any(p in text_lower for p in _DIRECTIVE_PHRASES)

    # Balanced no-op — sequential ordering is irrelevant
    if balanced and has_noop and not has_directive:
        return 1.0

    # Find zone/coach references in text order
    zone_positions = []
    for match in re.finditer(r"(?:Zones?|Coaches?)\s*([A-J])", announcement, re.IGNORECASE):
        zone_positions.append(match.group(1).upper())

    if len(zone_positions) < 2:
        return 0.0

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for label in zone_positions:
        if label not in seen:
            seen.add(label)
            unique.append(label)

    expected = sorted(unique)
    if unique == expected:
        sequential_score = 1.0
    else:
        matches = sum(1 for a, b in zip(unique, expected) if a == b)
        sequential_score = matches / len(expected)

    # Transition word bonus
    seq_count = sum(1 for m in _SEQUENTIAL_MARKERS if re.search(m, text_lower))
    transition_bonus = min(0.2, seq_count * 0.1)

    # Simultaneous language penalty
    sim_count = sum(1 for m in _SIMULTANEOUS_MARKERS if m in text_lower)
    simultaneous_penalty = min(0.5, sim_count * 0.25)

    return max(0.0, min(1.0, sequential_score + transition_bonus - simultaneous_penalty))


# ---------------------------------------------------------------------------
# Reward 8 — Factual Accuracy of Announcements
# ---------------------------------------------------------------------------

_CROWDED_PHRASES = [
    "crowded", "full", "quite full", "very full", "busy",
    "packed", "most crowded", "avoid",
]

_SPACIOUS_PHRASES = [
    "space", "room", "least crowded", "plenty", "less crowded",
    "spacious", "not crowded", "more space",
]


def compute_factual_accuracy(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    """Check whether crowd descriptions in the announcement match actual data."""
    announcement = _extract_announcement(response_text)
    text_lower = announcement.lower()

    labels = [chr(ord("a") + i) for i in range(num_coaches)]
    claims = 0
    correct = 0

    for i, label in enumerate(labels):
        for match in re.finditer(rf"coach\s*{label}\b", text_lower):
            context = text_lower[max(0, match.start() - 40):match.end() + 60]
            is_crowded_claim = any(p in context for p in _CROWDED_PHRASES)
            is_spacious_claim = any(p in context for p in _SPACIOUS_PHRASES)

            if is_crowded_claim:
                claims += 1
                if train_crowd[i] >= 60:
                    correct += 1
            elif is_spacious_claim:
                claims += 1
                if train_crowd[i] <= 50:
                    correct += 1

    if claims == 0:
        return 0.5
    return correct / claims


# ---------------------------------------------------------------------------
# Reward 9 — Platform Number Mention
# ---------------------------------------------------------------------------

def compute_platform_mention(
    response_text: str,
    platform_number: int,
) -> float:
    """Check whether the announcement correctly mentions the platform number."""
    announcement = _extract_announcement(response_text)
    if not announcement.strip():
        return 0.0

    matches = re.findall(r"platform\s*(?:number\s*)?(\d+)", announcement, re.IGNORECASE)
    if not matches:
        return 0.0

    for m in matches:
        if int(m) == platform_number:
            return 1.0

    return 0.0

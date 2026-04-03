# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rule-based reward functions for the MetroCrowdManager environment.

All 7 reward functions are deterministic heuristics (not LLM-as-judge).
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
]


def _is_balanced(train_crowd: List[int], platform_crowd: List[int], num_coaches: int) -> bool:
    """Check whether the current crowd state is roughly balanced."""
    if _std([float(p) for p in platform_crowd]) < 10:
        return True
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
# Reward 2 — Mathematical Accuracy
# ---------------------------------------------------------------------------

def compute_math_accuracy(
    response_text: str,
    train_crowd: List[int],
    platform_crowd: List[int],
    num_coaches: int,
) -> float:
    proposed = _parse_distribution(response_text)

    if proposed is None:
        # No distribution provided — neutral score for balanced scenarios
        if _is_balanced(train_crowd, platform_crowd, num_coaches):
            return 0.5
        return 0.0

    if len(proposed) != num_coaches:
        return 0.0

    # Compute ideal distribution weighted by available train capacity
    capacity = [100 - train_crowd[i] for i in range(num_coaches)]
    total_capacity = sum(capacity)
    if total_capacity == 0:
        total_capacity = 1  # avoid division by zero (fully packed train)

    avg_platform = sum(platform_crowd) / num_coaches
    avg_capacity = total_capacity / num_coaches
    if avg_capacity == 0:
        avg_capacity = 1

    ideal = [
        avg_platform * (capacity[i] / avg_capacity)
        for i in range(num_coaches)
    ]
    ideal = [max(0.0, min(100.0, p)) for p in ideal]

    # Mean Absolute Error in percentage points
    mae = sum(abs(proposed[i] - ideal[i]) for i in range(num_coaches)) / num_coaches
    score = max(0.0, 1.0 - mae / 50.0)

    # Feasibility penalty for impossible percentages
    if any(p < 0 or p > 100 for p in proposed):
        score *= 0.5

    return score


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
    if not response_text.strip():
        return 0.0

    # Component A: Sentence length (30%)
    sentences = [s.strip() for s in re.split(r"[.!?]+", response_text) if s.strip()]
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
    text_lower = response_text.lower()
    jargon_count = sum(1 for t in _JARGON_TERMS if t in text_lower)
    jargon_score = max(0.0, 1.0 - jargon_count * 0.25)

    # Component C: Structure bonus (30%)
    has_numbered = bool(re.search(r"\d+[.)]\s", response_text))
    has_bullets = bool(re.search(r"[-*\u2022]\s", response_text))
    has_headers = bool(re.search(r"(?:Coach|Zone)\s*[A-F]\s*:", response_text))
    structure_features = sum([has_numbered, has_bullets, has_headers])
    structure_score = min(1.0, structure_features * 0.5)

    # Component D: Minimum length (10%)
    length_ok = 1.0 if len(response_text) >= 50 else len(response_text) / 50.0

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
    balanced = _is_balanced(train_crowd, platform_crowd, num_coaches)
    text_lower = response_text.lower()

    has_noop = any(p in text_lower for p in _NOOP_PHRASES)
    has_directive = any(p in text_lower for p in _DIRECTIVE_PHRASES)

    # Balanced no-op — sequential ordering is irrelevant
    if balanced and has_noop and not has_directive:
        return 1.0

    # Find zone/coach references in text order
    zone_positions = []
    for match in re.finditer(r"(?:Zone|Coach)\s*([A-F])", response_text, re.IGNORECASE):
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

"""Estimator-matched sample-size simulation for Work Planner.

Normative simulations preserve the paired-binary outcome space. They never add a
continuous effect to a binary difference. Each generated pair is one of
(0,0), (0,1), (1,0), or (1,1), so differences are always -1, 0, or 1.
"""
from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta

from analysis.decision_gates import (
    clustered_stage1a_bootstrap,
    hierarchical_planner_bootstrap,
    paired_task_bootstrap,
    paired_tost,
)

CONSERVATIVE_DISCORDANCE = 0.10
CONSERVATIVE_ZERO_SD = math.sqrt(CONSERVATIVE_DISCORDANCE)
PAIR_CATEGORIES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _round_up(value: int | float, multiple: int = 50) -> int:
    return int(math.ceil(float(value) / multiple) * multiple)


def _validate_pair(row: dict[str, Any]) -> tuple[int, int]:
    left, right = row.get("left"), row.get("right")
    if left not in (0, 1) or right not in (0, 1):
        raise ValueError("sample-size pairs must contain binary left/right outcomes")
    expected = int(left) - int(right)
    if float(row.get("difference", 999)) != float(expected):
        raise ValueError("pair difference must equal left-right")
    return int(left), int(right)


def _pair_differences(rows: Iterable[dict[str, Any]]) -> list[float]:
    return [float(l - r) for l, r in (_validate_pair(row) for row in rows)]


def _flatten_pairs(requirement: dict[str, Any]) -> list[dict[str, Any]]:
    kind = requirement["analysis_type"]
    if kind == "PLANNER_HIERARCHICAL":
        return [row for rows in requirement["seed_groups"].values() for row in rows]
    if kind == "STAGE1A_CLUSTERED":
        return [row for rows in requirement["task_clusters"].values() for row in rows]
    if kind == "TASK_PAIRED":
        return list(requirement["pairs"])
    raise ValueError(f"unsupported sample-size analysis_type: {kind}")


def final_analysis_units(requirement: dict[str, Any]) -> list[float]:
    kind = requirement["analysis_type"]
    if kind == "PLANNER_HIERARCHICAL":
        return [x for rows in requirement["seed_groups"].values() for x in _pair_differences(rows)]
    if kind == "STAGE1A_CLUSTERED":
        return [float(np.mean(_pair_differences(rows))) for rows in requirement["task_clusters"].values()]
    if kind == "TASK_PAIRED":
        return _pair_differences(requirement["pairs"])
    raise ValueError(f"unsupported sample-size analysis_type: {kind}")


def paired_sd(requirement_or_differences: dict[str, Any] | list[float]) -> float:
    values = (
        final_analysis_units(requirement_or_differences)
        if isinstance(requirement_or_differences, dict)
        else [float(x) for x in requirement_or_differences]
    )
    if len(values) < 2:
        raise ValueError("at least two pilot analysis units required")
    value = stdev(values)
    return CONSERVATIVE_ZERO_SD if value == 0 else value


def _observed_effect(requirement: dict[str, Any]) -> float:
    kind = requirement["analysis_type"]
    if kind == "PLANNER_HIERARCHICAL":
        return float(np.mean([np.mean(_pair_differences(rows)) for rows in requirement["seed_groups"].values()]))
    if kind == "STAGE1A_CLUSTERED":
        return float(np.mean([np.mean(_pair_differences(rows)) for rows in requirement["task_clusters"].values()]))
    return float(np.mean(_pair_differences(requirement["pairs"])))


def _binary_joint_model(rows: Iterable[dict[str, Any]], target_effect: float) -> np.ndarray:
    """Return probabilities for 00,01,10,11 with E[left-right]=target_effect.

    The model preserves at least the pilot discordance rate. A locked 10% symmetric
    discordance sensitivity floor is used when the pilot is degenerate. If the
    requested effect needs more discordance, discordance is conservatively raised
    to |effect| + 0.02. Agreement is split using the pilot 11 share.
    """
    materialized = [_validate_pair(row) for row in rows]
    if not materialized:
        raise ValueError("empty pilot pair set")
    counts = {cat: 0 for cat in PAIR_CATEGORIES}
    for pair in materialized:
        counts[pair] += 1
    total = len(materialized)
    observed_discordance = (counts[(0, 1)] + counts[(1, 0)]) / total
    discordance = max(observed_discordance, CONSERVATIVE_DISCORDANCE, abs(target_effect) + 0.02)
    if discordance > 1.0:
        raise ValueError("target effect is incompatible with paired binary outcomes")
    if abs(target_effect) > discordance + 1e-12:
        raise ValueError("target effect exceeds modeled discordance")
    p10 = (discordance + target_effect) / 2.0
    p01 = (discordance - target_effect) / 2.0
    agreements = counts[(0, 0)] + counts[(1, 1)]
    p11_share = counts[(1, 1)] / agreements if agreements else 0.5
    p11 = (1.0 - discordance) * p11_share
    p00 = (1.0 - discordance) * (1.0 - p11_share)
    probs = np.asarray([p00, p01, p10, p11], dtype=float)
    if np.any(probs < -1e-12) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("invalid paired-binary probability model")
    return np.clip(probs, 0.0, 1.0)


def _draw_binary_differences(
    rows: Iterable[dict[str, Any]], n: int, rng: np.random.Generator, target_effect: float
) -> list[float]:
    probs = _binary_joint_model(rows, target_effect)
    indices = rng.choice(len(PAIR_CATEGORIES), size=n, replace=True, p=probs)
    return [float(PAIR_CATEGORIES[int(i)][0] - PAIR_CATEGORIES[int(i)][1]) for i in indices]


def _draw_requirement(
    requirement: dict[str, Any], n: int, rng: np.random.Generator, effect: float
) -> dict[str, Any]:
    """Generate one estimator-matched binary dataset with n top-level tasks."""
    kind = requirement["analysis_type"]
    if kind == "PLANNER_HIERARCHICAL":
        source_groups = {str(k): list(v) for k, v in requirement["seed_groups"].items()}
        keys = sorted(source_groups)
        overall = _observed_effect(requirement)
        out: dict[str, list[float]] = {}
        # Final analysis always has exactly five final seeds. Resampling a source
        # seed preserves empirical between-seed heterogeneity without inventing
        # continuous task outcomes.
        for output_seed in keys:
            source_key = keys[int(rng.integers(0, len(keys)))]
            source_rows = source_groups[source_key]
            source_effect = float(np.mean(_pair_differences(source_rows)))
            target_seed_effect = float(np.clip(effect + source_effect - overall, -0.98, 0.98))
            out[output_seed] = _draw_binary_differences(source_rows, n, rng, target_seed_effect)
        return {"analysis_type": kind, "seed_groups": out}

    if kind == "STAGE1A_CLUSTERED":
        source = list(requirement["task_clusters"].values())
        if not source:
            raise ValueError("no Stage 1A pilot clusters")
        overall = _observed_effect(requirement)
        all_rows = _flatten_pairs(requirement)
        clusters: dict[str, list[float]] = {}
        for i in range(n):
            source_rows = source[int(rng.integers(0, len(source)))]
            source_mean = float(np.mean(_pair_differences(source_rows)))
            target_task_effect = float(np.clip(effect + source_mean - overall, -0.98, 0.98))
            # Preserve the sampled task's snapshot count and task-level residual;
            # use the global pilot joint table for stable binary cell estimates.
            clusters[f"sim-{i}"] = _draw_binary_differences(
                all_rows, len(source_rows), rng, target_task_effect
            )
        return {"analysis_type": kind, "task_clusters": clusters}

    if kind == "TASK_PAIRED":
        return {
            "analysis_type": kind,
            "pairs": _draw_binary_differences(requirement["pairs"], n, rng, effect),
        }
    raise ValueError(f"unsupported sample-size analysis_type: {kind}")


def _estimate_ci(dataset: dict[str, Any], *, resamples: int, seed: int) -> tuple[float, float, float]:
    kind = dataset["analysis_type"]
    if kind == "PLANNER_HIERARCHICAL":
        return hierarchical_planner_bootstrap(dataset["seed_groups"], resamples=resamples, seed=seed)
    if kind == "STAGE1A_CLUSTERED":
        return clustered_stage1a_bootstrap(dataset["task_clusters"], resamples=resamples, seed=seed)
    if kind == "TASK_PAIRED":
        return paired_task_bootstrap(dataset["pairs"], resamples=resamples, seed=seed)
    raise ValueError(kind)


COMPONENTS = {
    "primary_ci",
    "primary_power",
    "current_vs_shuffled_power",
    "random_code_power",
    "structured_noninferiority_power",
    "self_plan_power",
    "flops_direction_power",
}


def _component_effect(component: str, design_effect: float) -> float:
    if component == "primary_ci":
        raise ValueError("primary_ci uses observed pilot effect")
    if component == "structured_noninferiority_power":
        return 0.0
    if component == "flops_direction_power":
        return max(0.02, min(float(design_effect), 0.05))
    return float(design_effect)


def _passes_component(
    component: str,
    dataset: dict[str, Any],
    *,
    minimum_effect_gate: float,
    half_width: float,
    ci_resamples: int,
    seed: int,
) -> bool:
    estimate, lo, hi = _estimate_ci(dataset, resamples=ci_resamples, seed=seed)
    if component == "primary_ci":
        return (hi - lo) / 2 <= half_width
    if component == "primary_power":
        return estimate >= minimum_effect_gate and lo > 0.0
    if component in {"current_vs_shuffled_power", "random_code_power", "self_plan_power"}:
        return lo > 0.0
    if component == "structured_noninferiority_power":
        return lo >= -0.02
    if component == "flops_direction_power":
        return estimate >= 0.0 and lo >= -0.02
    raise ValueError(component)


def binomial_power_lower_bound(passed: int, simulations: int, confidence_level: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson lower confidence bound for simulated power."""
    if simulations < 1 or not 0 <= passed <= simulations:
        raise ValueError("invalid simulated-power counts")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("power confidence level must be between 0.5 and 1")
    if passed == 0:
        return 0.0
    alpha = 1.0 - confidence_level
    return float(beta.ppf(alpha, passed, simulations - passed + 1))


def _requirement_n(
    requirement: dict[str, Any],
    component: str,
    *,
    design_effect: float,
    minimum_effect_gate: float,
    half_width: float,
    target_power: float,
    simulations: int,
    ci_resamples: int,
    seed: int,
    minimum_n: int,
    round_multiple: int,
    maximum_n: int,
    power_confidence_level: float,
    power_confirmation_points: int,
) -> int:
    units = final_analysis_units(requirement)
    if len(units) < 20:
        raise ValueError(f"{component}: at least 20 pilot analysis units required")
    if power_confirmation_points < 1:
        raise ValueError("power_confirmation_points must be positive")
    if component == "primary_ci":
        effect = _observed_effect(requirement)
    else:
        effect = _component_effect(component, design_effect)
    if component == "primary_power" and effect <= minimum_effect_gate:
        raise ValueError("design effect must be strictly above the primary GO boundary")
    passing_streak: list[int] = []
    for n in range(_round_up(minimum_n, round_multiple), maximum_n + 1, round_multiple):
        rng = np.random.default_rng(seed + 10_000_019 * n)
        passed = 0
        for sim in range(simulations):
            dataset = _draw_requirement(requirement, n, rng, effect)
            passed += _passes_component(
                component, dataset, minimum_effect_gate=minimum_effect_gate,
                half_width=half_width, ci_resamples=ci_resamples,
                seed=seed + 100_000 * (sim + 1) + n,
            )
        lower = binomial_power_lower_bound(passed, simulations, power_confidence_level)
        if lower >= target_power:
            passing_streak.append(n)
            if len(passing_streak) >= power_confirmation_points:
                return passing_streak[0]
        else:
            passing_streak.clear()
    raise ValueError(f"{component}: required N exceeds maximum_n")


def calculate_components_from_structured_requirements(
    requirements: dict[str, dict[str, Any]],
    *,
    design_effect: float = 0.075,
    minimum_effect_gate: float = 0.05,
    half_width: float = 0.025,
    equivalence_margin: float | None = None,
    target_power: float = 0.90,
    simulations: int = 1000,
    ci_resamples: int = 1000,
    seed: int = 7302,
    minimum_n: int = 300,
    round_multiple: int = 50,
    maximum_n: int = 20000,
    power_confidence_level: float = 0.95,
    power_confirmation_points: int = 2,
) -> dict[str, int]:
    del equivalence_margin  # legacy API argument; confirmatory TOST is forbidden in v1.21
    requested = set(requirements)
    if not requested or not requested.issubset(COMPONENTS):
        raise ValueError(f"unsupported requirement set: {requested}; supported={COMPONENTS}")
    if not {"primary_ci", "primary_power"}.issubset(requested):
        raise ValueError("primary_ci and primary_power are mandatory for every stage")
    out: dict[str, int] = {}
    for offset, component in enumerate(sorted(requested)):
        out[component] = _requirement_n(
            requirements[component], component, design_effect=design_effect,
            minimum_effect_gate=minimum_effect_gate, half_width=half_width,
            target_power=target_power, simulations=simulations,
            ci_resamples=ci_resamples, seed=seed + offset, minimum_n=minimum_n,
            round_multiple=round_multiple, maximum_n=maximum_n,
            power_confidence_level=power_confidence_level,
            power_confirmation_points=power_confirmation_points,
        )
    out["minimum_n"] = _round_up(minimum_n, round_multiple)
    out["selected_n"] = max(out.values())
    return out


def calculate_components_from_requirements(requirements: dict[str, list[float]], **kwargs: Any) -> dict[str, int]:
    """Non-normative closed-form compatibility helper for legacy toy fixtures."""
    expected = {"primary_ci", "primary_power", "current_vs_shuffled_power", "equivalence_TOST_power"}
    if set(requirements) != expected:
        raise ValueError("requirement set mismatch")
    half_width = float(kwargs.get("half_width", 0.025))
    target_effect = float(kwargs.get("design_effect", 0.075))
    equivalence_margin = float(kwargs.get("equivalence_margin", 0.02))
    minimum_n = int(kwargs.get("minimum_n", 300))
    round_multiple = int(kwargs.get("round_multiple", 50))
    from scipy.stats import norm

    z975, z95, z90 = float(norm.ppf(.975)), float(norm.ppf(.95)), float(norm.ppf(.90))
    sds = {name: paired_sd(values) for name, values in requirements.items()}
    out = {
        "primary_ci": max(_round_up((z975 * sds["primary_ci"] / half_width) ** 2, round_multiple), _round_up(minimum_n, round_multiple)),
        "primary_power": max(_round_up(((z975 + z90) * sds["primary_power"] / target_effect) ** 2, round_multiple), _round_up(minimum_n, round_multiple)),
        "current_vs_shuffled_power": max(_round_up(((z975 + z90) * sds["current_vs_shuffled_power"] / target_effect) ** 2, round_multiple), _round_up(minimum_n, round_multiple)),
        "equivalence_TOST_power": max(_round_up(((z95 + z90) * sds["equivalence_TOST_power"] / equivalence_margin) ** 2, round_multiple), _round_up(minimum_n, round_multiple)),
        "minimum_n": _round_up(minimum_n, round_multiple),
    }
    out["selected_n"] = max(out.values())
    return out


def calculate_components(
    sd: float,
    *,
    target_effect: float = 0.05,
    half_width: float = 0.025,
    equivalence_margin: float = 0.02,
    minimum_n: int = 300,
    round_multiple: int = 50,
) -> dict[str, int]:
    """Non-normative closed-form smoke helper retained for toy preflight tests."""
    from scipy.stats import norm

    sd = max(float(sd), CONSERVATIVE_ZERO_SD)
    z975, z95, z90 = float(norm.ppf(0.975)), float(norm.ppf(0.95)), float(norm.ppf(0.90))
    rows = {
        "primary_ci": _round_up((z975 * sd / half_width) ** 2, round_multiple),
        "primary_power": _round_up(((z975 + z90) * sd / target_effect) ** 2, round_multiple),
        "current_vs_shuffled_power": _round_up(((z975 + z90) * sd / target_effect) ** 2, round_multiple),
        "equivalence_TOST_power": _round_up(((z95 + z90) * sd / equivalence_margin) ** 2, round_multiple),
        "minimum_n": _round_up(minimum_n, round_multiple),
    }
    rows["selected_n"] = max(rows.values())
    return rows

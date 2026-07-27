"""Locked statistics implementation for work-planner statistics/1.1."""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from statistics import fmean, stdev
from typing import Any

import numpy as np
from scipy.stats import norm, t as student_t

BOOTSTRAP_SEED = 7301


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty bootstrap distribution")
    values = sorted(values)
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def paired_task_bootstrap(differences: Iterable[float], *, resamples: int = 10000, seed: int = BOOTSTRAP_SEED, confidence: float = 0.95) -> tuple[float, float, float]:
    values = np.asarray([float(x) for x in differences],dtype=float)
    if values.size==0:
        raise ValueError("no complete paired differences")
    observed=float(values.mean())
    if np.allclose(values,values[0]):
        return observed,observed,observed
    rng=np.random.default_rng(seed); n=values.size
    # Chunked vectorisation keeps memory bounded for large confirmatory N.
    draws=[]; remaining=resamples
    while remaining:
        size=min(1000,remaining); remaining-=size
        idx=rng.integers(0,n,size=(size,n))
        draws.append(values[idx].mean(axis=1))
    distribution=np.concatenate(draws)
    alpha=1.0-confidence
    lo,hi=np.quantile(distribution,[alpha/2,1-alpha/2],method="linear")
    return observed,float(lo),float(hi)


def percentile_bootstrap_ci(differences: Iterable[float], *, resamples: int = 10000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Compatibility alias for paired task bootstrap."""
    return paired_task_bootstrap(differences, resamples=resamples, seed=seed)


def wilson_binary_rate_ci(values: Iterable[float], *, confidence: float = 0.95) -> tuple[float, float, float]:
    rows=[float(x) for x in values]
    if not rows:
        raise ValueError("no binary rate units")
    if any(x not in (0.0,1.0) for x in rows):
        raise ValueError("Wilson rate CI requires unit-level binary values")
    n=len(rows); estimate=float(sum(rows)/n); z=float(norm.ppf(0.5+confidence/2))
    denom=1.0+z*z/n
    center=(estimate+z*z/(2*n))/denom
    half=z*math.sqrt(estimate*(1-estimate)/n+z*z/(4*n*n))/denom
    return estimate,max(0.0,center-half),min(1.0,center+half)


def hierarchical_planner_bootstrap(seed_to_task_differences: Mapping[str | int, Iterable[float]], *, resamples: int = 10000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    groups = {str(k): np.asarray([float(v) for v in vals],dtype=float) for k, vals in seed_to_task_differences.items()}
    if not groups or any(vals.size==0 for vals in groups.values()):
        raise ValueError("every planner seed must contain complete paired task differences")
    keys=sorted(groups); observed=float(np.mean([groups[k].mean() for k in keys]))
    if all(np.allclose(groups[k],groups[k][0]) for k in keys) and len({float(groups[k][0]) for k in keys})==1:
        return observed,observed,observed
    rng=np.random.default_rng(seed); draws=np.empty(resamples,dtype=float)
    for r in range(resamples):
        sampled=[]
        for _ in keys:
            k=keys[int(rng.integers(0,len(keys)))]; vals=groups[k]
            sampled.append(float(vals[rng.integers(0,vals.size,size=vals.size)].mean()))
        draws[r]=float(np.mean(sampled))
    lo,hi=np.quantile(draws,[.025,.975],method="linear")
    return observed,float(lo),float(hi)


def clustered_stage1a_bootstrap(task_to_snapshot_differences: Mapping[str, Iterable[float]], *, resamples: int = 10000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    task_means = []
    for _, vals in sorted(task_to_snapshot_differences.items()):
        materialized = [float(x) for x in vals]
        if materialized:
            task_means.append(fmean(materialized))
    if not task_means:
        raise ValueError("no complete Stage 1A task clusters")
    return paired_task_bootstrap(task_means, resamples=resamples, seed=seed)


def paired_tost(differences: Iterable[float], margin: float, *, alpha: float = 0.05) -> dict[str, float | bool]:
    values = [float(x) for x in differences]
    if len(values) < 2:
        raise ValueError("paired TOST requires at least two complete pairs")
    mean = fmean(values)
    sd = stdev(values)
    if sd == 0:
        passed = -margin < mean < margin
        return {"estimate": mean, "ci_low": mean, "ci_high": mean, "p_lower": 0.0 if passed else 1.0, "p_upper": 0.0 if passed else 1.0, "pass": passed}
    se = sd / math.sqrt(len(values)); df = len(values) - 1
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = float(student_t.sf(t_lower, df))
    p_upper = float(student_t.cdf(t_upper, df))
    critical = float(student_t.ppf(1 - alpha, df))
    ci_low = mean - critical * se; ci_high = mean + critical * se
    return {"estimate": mean, "ci_low": ci_low, "ci_high": ci_high, "p_lower": p_lower, "p_upper": p_upper, "pass": p_lower < alpha and p_upper < alpha}



def positive_seed_count(seed_to_task_differences: Mapping[str | int, Iterable[float]]) -> int:
    groups={str(k):[float(v) for v in vals] for k,vals in seed_to_task_differences.items()}
    if not groups or any(not vals for vals in groups.values()):
        raise ValueError("every planner seed must contain paired task differences")
    return sum(1 for vals in groups.values() if fmean(vals) > 0.0)

def evaluate_gate(rule: str, estimate: float, ci_low: float, ci_high: float, threshold: float, *, raw_differences: Iterable[float] | None = None) -> bool:
    if rule == "ci_low_gte": return ci_low >= threshold
    if rule == "ci_low_gt": return ci_low > threshold
    if rule == "estimate_gte_and_ci_low_gt": return estimate >= threshold and ci_low > 0
    if rule == "estimate_gte": return estimate >= threshold
    if rule == "estimate_lte": return estimate <= threshold
    if rule == "minimum_positive_seed_count": return estimate >= threshold
    if rule == "paired_tost":
        if raw_differences is None: raise ValueError("paired_tost requires raw_differences")
        return bool(paired_tost(raw_differences, threshold)["pass"])
    raise ValueError(f"unknown locked rule: {rule}")


def stage_decision(stage: str, gates: list[dict[str, Any]], *, planner_stage1b_eligible: bool | None = None) -> str:
    passed = all(bool(g["pass"]) for g in gates)
    if stage == "PLANNER":
        architecture = [g for g in gates if g.get("gate_group") == "ARCHITECTURE"]
        eligibility = [g for g in gates if g.get("gate_group") == "STAGE1B_ELIGIBILITY"]
        if not architecture or not all(g["pass"] for g in architecture): return "STOP_PLANNER"
        return "GO_PLANNER_STAGE1B_ELIGIBLE" if eligibility and all(g["pass"] for g in eligibility) else "GO_PLANNER_DIAGNOSTIC_ONLY"
    if stage == "STAGE1A":
        if not passed: return "STOP_INTERFACE"
        return "GO_INTERFACE_STAGE1B_ELIGIBLE" if planner_stage1b_eligible else "GO_INTERFACE_DIAGNOSTIC_ONLY"
    if stage == "STAGE1B": return "GO_END_TO_END" if passed else "STOP_END_TO_END"
    raise ValueError(f"unknown stage: {stage}")

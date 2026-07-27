from __future__ import annotations
import json
from pathlib import Path

from analysis.decision_gates import paired_task_bootstrap,hierarchical_planner_bootstrap,clustered_stage1a_bootstrap,paired_tost
from analysis.sample_size import calculate_components_from_requirements

ROOT=Path(__file__).resolve().parents[1]
CASES=json.loads((ROOT/'docs/statistics/golden_cases_v1.1.json').read_text())['cases']

def test_statistics_golden_cases_are_byte_stable():
    c=CASES['paired_constant']; assert list(paired_task_bootstrap(c['differences']))==c['expected']
    c=CASES['planner_hierarchical_constant']; assert list(hierarchical_planner_bootstrap(c['seed_groups']))==c['expected']
    c=CASES['stage1a_clustered_constant']; assert list(clustered_stage1a_bootstrap(c['task_clusters']))==c['expected']
    c=CASES['paired_tost_zero']; r=paired_tost(c['differences'],c['margin'])
    for k,v in c['expected'].items(): assert r[k]==v
    c=CASES['sample_size_requirement_specific']; assert calculate_components_from_requirements(c['requirements'])==c['expected']

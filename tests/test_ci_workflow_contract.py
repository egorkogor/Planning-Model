from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"
FORMAL_PATH = ROOT / ".github" / "workflows" / "fixed-target-acceptance.yml"


def _ci_text() -> str:
    return CI_PATH.read_text(encoding="utf-8")


def test_hosted_same_environment_reproducibility_remains_fail_closed() -> None:
    workflow = _ci_text()

    assert 'python-version: "3.11.15"' in workflow
    assert "RUN_CANONICAL_QUALITY_DETERMINISM=1" in workflow
    assert "--output-dir .quality-eval-ci" in workflow
    assert "--output-dir .quality-eval-ci-independent" in workflow
    assert 'cmp ".quality-eval-ci/$artifact" ".quality-eval-ci-independent/$artifact"' in workflow
    assert (
        "cmp .quality-docs-ci/data/a2_a3_a4_heldout_summary.json \\\n"
        "            .quality-docs-ci-independent/data/a2_a3_a4_heldout_summary.json"
        in workflow
    )
    assert (
        "cmp .quality-docs-ci/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md \\\n"
        "            .quality-docs-ci-independent/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md"
        in workflow
    )


def test_hosted_cross_target_exactness_is_not_an_acceptance_gate() -> None:
    workflow = _ci_text()

    assert (
        "cmp .quality-docs-ci/data/a2_a3_a4_heldout_summary.json \\\n"
        "            docs/evaluations/data/a2_a3_a4_heldout_summary.json"
        not in workflow
    )
    assert (
        "cmp .quality-docs-ci/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md \\\n"
        "            docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md"
        not in workflow
    )
    assert 'cmp "a/$path" "b/$path"' not in workflow
    assert "HOSTED_CROSS_WORKER_MATCH" in workflow
    assert "HOSTED_CROSS_WORKER_DIFFERENCE_OBSERVED_NON_ACCEPTANCE" in workflow
    assert "|| true" not in workflow


def test_hosted_cross_worker_runs_remain_bound_and_diagnostic() -> None:
    workflow = _ci_text()

    assert "canonical-run-${{ matrix.worker }}" in workflow
    assert "worker: [a, b]" in workflow
    assert "cmp a/implementation-sha.txt b/implementation-sha.txt" in workflow
    assert "printf '%s\\n' \"$IMPLEMENTATION_SHA\" | cmp - a/implementation-sha.txt" in workflow
    assert 'test -f "a/$path"' in workflow
    assert 'test -f "b/$path"' in workflow
    assert "python -m scripts.compare_toy_quality_runs" in workflow
    assert "a b --output cross-worker-report.json" in workflow
    assert "HOSTED_CROSS_WORKER_REPORT_MALFORMED" in workflow
    assert "name: canonical-cross-worker-diagnostic" in workflow
    assert "path: cross-worker-report.json" in workflow


def test_formal_fixed_target_gate_remains_separate_and_authoritative() -> None:
    hosted = _ci_text()
    formal = FORMAL_PATH.read_text(encoding="utf-8")

    assert "planning-model-canonical-cpu-v1" not in hosted
    assert "final-gate" not in hosted
    assert "planning-model-canonical-cpu-v1" in formal
    assert "workflow_dispatch:" in formal
    assert "validate-bundle" in formal
    assert "Final runtime 1.1 acceptance gate" in formal
    assert "final-gate" in formal

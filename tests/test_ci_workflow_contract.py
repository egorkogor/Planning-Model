import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"
FORMAL_PATH = ROOT / ".github" / "workflows" / "fixed-target-acceptance.yml"
A2_DIAGNOSTIC_PATH = ROOT / ".github" / "workflows" / "a2-learnability-diagnostic.yml"
BOOTSTRAP_MANIFEST_PATH = ROOT / "release" / "BOOTSTRAP_MANIFEST.json"


def _ci_text() -> str:
    return CI_PATH.read_text(encoding="utf-8")


def _workflow_document(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


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


def test_hosted_ci_is_bootstrap_protected_and_formal_gate_remains_separate() -> None:
    bootstrap = json.loads(BOOTSTRAP_MANIFEST_PATH.read_text(encoding="utf-8"))
    hosted_digest = bootstrap["files"].get(".github/workflows/ci.yml")
    hosted = _ci_text()
    formal = FORMAL_PATH.read_text(encoding="utf-8")

    assert isinstance(hosted_digest, str)
    assert hosted_digest.startswith("sha256:")
    assert len(hosted_digest) == len("sha256:") + 64
    assert "planning-model-canonical-cpu-v1" not in hosted
    assert "final-gate" not in hosted
    assert "planning-model-canonical-cpu-v1" in formal
    assert "workflow_dispatch:" in formal
    assert "validate-bundle" in formal
    assert "Final runtime 1.1 acceptance gate" in formal
    assert "final-gate" in formal


def test_a2_diagnostic_uses_canonical_runner_image_provenance_binding() -> None:
    document = _workflow_document(A2_DIAGNOSTIC_PATH)
    workflow = A2_DIAGNOSTIC_PATH.read_text(encoding="utf-8")

    assert set(document["on"]) == {"workflow_dispatch"}
    job = document["jobs"]["a2-learnability-diagnostic"]
    assert job["name"] == "a2-learnability-diagnostic-development-only"
    assert job["runs-on"] == [
        "self-hosted",
        "linux",
        "x64",
        "planning-model-canonical-cpu-v1",
    ]

    steps = job["steps"]
    step_names = [step["name"] for step in steps]
    identity_index = step_names.index("Verify immutable runner identity file")
    preflight_index = step_names.index(
        "Validate dedicated target observation for diagnostic provenance"
    )
    diagnostic_index = step_names.index("Run development-only A2 learnability diagnostic")
    assert identity_index < preflight_index < diagnostic_index

    identity_run = steps[identity_index]["run"]
    preflight_run = steps[preflight_index]["run"]
    diagnostic_run = steps[diagnostic_index]["run"]

    assert "/etc/planning-model-runner-image-id" in identity_run
    assert "printf 'FIXED_TARGET_RUNNER_IMAGE=%s\\n'" in identity_run
    assert '>> "$GITHUB_ENV"' in identity_run
    assert "python -m scripts.run_fixed_target_acceptance preflight" in preflight_run
    assert (
        'test "$(cat /etc/planning-model-runner-image-id)" = '
        '"$FIXED_TARGET_RUNNER_IMAGE"'
        in diagnostic_run
    )

    run_scripts = "\n".join(step.get("run", "") for step in steps)
    assert "$A2_LEARNABILITY_RUNNER_IMAGE" not in run_scripts
    assert "${A2_LEARNABILITY_RUNNER_IMAGE}" not in run_scripts
    assert "A2_LEARNABILITY_RUNNER_IMAGE" not in job.get("env", {})
    for step in steps:
        assert "A2_LEARNABILITY_RUNNER_IMAGE" not in step.get("env", {})

    runner_image_env_writes = [
        line.strip()
        for step in steps
        for line in step.get("run", "").splitlines()
        if "RUNNER_IMAGE" in line and "$GITHUB_ENV" in line
    ]
    assert runner_image_env_writes == [
        "printf 'FIXED_TARGET_RUNNER_IMAGE=%s\\n' \"$runner_image\" >> \"$GITHUB_ENV\""
    ]

    assert "validate-bundle" not in workflow
    assert "final-gate" not in workflow
    assert "Final runtime 1.1 acceptance gate" not in workflow
    assert "rerun" not in workflow.lower()


def test_a2_diagnostic_uploads_complete_hidden_evidence_after_validation() -> None:
    document = _workflow_document(A2_DIAGNOSTIC_PATH)
    job = document["jobs"]["a2-learnability-diagnostic"]
    steps = job["steps"]
    step_names = [step["name"] for step in steps]

    validation_index = step_names.index("Independently validate diagnostic bundle")
    upload_index = step_names.index("Upload development diagnostic evidence")
    cleanup_index = step_names.index("Cleanup diagnostic workspace")
    assert validation_index < upload_index < cleanup_index

    canonical_uploads = [
        step for step in steps if step.get("name") == "Upload development diagnostic evidence"
    ]
    assert len(canonical_uploads) == 1
    upload = canonical_uploads[0]
    assert upload["uses"] == "actions/upload-artifact@v4"

    upload_config = upload["with"]
    assert upload_config["path"] == ".a2-learnability"
    assert upload_config["include-hidden-files"] == "true"
    assert upload_config["if-no-files-found"] == "error"
    assert upload_config["retention-days"] == "30"
    assert upload_config["name"] == "a2-learnability-development-${{ github.run_id }}"

    cleanup_run = steps[cleanup_index]["run"]
    assert "rm -rf .a2-learnability" in cleanup_run


def test_formal_workflow_keeps_canonical_runner_image_binding() -> None:
    formal = FORMAL_PATH.read_text(encoding="utf-8")

    assert "printf 'FIXED_TARGET_RUNNER_IMAGE=%s\\n'" in formal
    assert "A2_LEARNABILITY_RUNNER_IMAGE" not in formal

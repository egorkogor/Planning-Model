from planner_llm_mvp.reproducibility import MANUAL_GATES, PHASE_IDS, validate_release_index


def test_incomplete_release_index_is_rejected() -> None:
    assert validate_release_index({})


def test_complete_release_index_is_accepted() -> None:
    data = {
        "schema_version": "work-planner-release/1.0",
        "git_commit": "a" * 40,
        "phase_reports": {phase: f"reports/phase-{phase}.json" for phase in PHASE_IDS},
        "decisions": {gate: f"decisions/{gate}.json" for gate in MANUAL_GATES},
        "artifacts": [{"path": "release/bundle.tar.gz", "sha256": "sha256:" + "b" * 64}],
    }
    assert validate_release_index(data) == []


def test_missing_gate_or_phase_is_rejected() -> None:
    data = {
        "schema_version": "work-planner-release/1.0",
        "git_commit": "a" * 40,
        "phase_reports": {phase: "x" for phase in PHASE_IDS[:-1]},
        "decisions": {gate: "x" for gate in list(MANUAL_GATES)[:-2]},
        "artifacts": [{"path": "x", "sha256": "sha256:" + "b" * 64}],
    }
    errors = validate_release_index(data)
    assert any("P00 through P20" in error for error in errors)
    assert any("all seven mandatory manual gates" in error for error in errors)

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import validation.verify_lock as lock_module

SOURCE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_lock_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "repo"
    shutil.copytree(
        SOURCE_ROOT,
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"),
    )
    monkeypatch.setattr(lock_module, "ROOT", workspace)
    monkeypatch.setattr(
        lock_module,
        "POLICIES",
        {
            "scientific": workspace / "docs/operator/scientific_lock_v1.yaml",
            "implementation": workspace / "docs/operator/implementation_lock_v1.yaml",
        },
    )
    monkeypatch.setattr(
        lock_module,
        "DEFAULT_LOCKS",
        {
            "scientific": workspace / "locks/scientific.lock.json",
            "implementation": workspace / "locks/implementation.lock.json",
        },
    )
    return workspace


def test_contract_lock_detects_mutation(isolated_lock_repo: Path):
    lock = isolated_lock_repo / "locks/test-scientific.lock.json"
    lock_module.create("scientific", lock, "run-test")
    assert lock_module.verify("scientific", lock) == []
    target = isolated_lock_repo / "docs/statistics/statistics_contract_v1.yaml"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    errors = lock_module.verify("scientific", lock)
    assert any("hash mismatch" in error for error in errors)


def test_implementation_lock_detects_runtime_mutation(
    isolated_lock_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    lock = isolated_lock_repo / "locks/test-implementation.lock.json"
    candidate_path = isolated_lock_repo / "freezes/implementation-lock.candidate.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "path": p.relative_to(isolated_lock_repo).as_posix(),
            "sha256": lock_module.digest(p),
        }
        for p in lock_module.protected_files("implementation")
    ]
    candidate = {"candidate_hash": "sha256:" + "a" * 64, "protected_files": rows}
    candidate_path.write_text(json.dumps(candidate))
    monkeypatch.setattr(lock_module, "validate_candidate", lambda obj: [])
    lock_module.create("implementation", lock, "run-test", candidate_path)
    assert lock_module.verify("implementation", lock) == []
    target = isolated_lock_repo / "src/planner_llm_mvp/reproducibility.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any("hash mismatch" in error for error in lock_module.verify("implementation", lock))

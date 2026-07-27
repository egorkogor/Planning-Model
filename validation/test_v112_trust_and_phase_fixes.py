from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_operator_keys(base: Path) -> tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    public_raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    private_path = base / "operator-private.b64"
    public_path = base / "operator-public.b64"
    private_path.write_bytes(base64.b64encode(private_raw))
    public_path.write_bytes(base64.b64encode(public_raw))
    return private_path, public_path


def _minimal_trust_artifacts(workspace: Path) -> None:
    (workspace / "reports").mkdir(exist_ok=True)
    (workspace / "locks").mkdir(exist_ok=True)
    (workspace / "artifacts").mkdir(exist_ok=True)
    (workspace / "artifacts/scope.md").write_text("operator-approved scope\n")
    (workspace / "reports/resource-plan.json").write_text('{"plan":"operator-approved"}\n')
    (workspace / "locks/infrastructure-plan.json").write_text('{"infra":"separate-identities"}\n')
    (workspace / "locks/public-keys.json").write_text('{"keys":"attested"}\n')


def _configure_trust_module(monkeypatch, workspace: Path):
    import validation.trust_topology_validator as trust_module

    monkeypatch.setattr(trust_module, "ROOT", workspace)
    monkeypatch.setattr(trust_module, "POLICY", workspace / "docs/operator/trust_topology_lock_v1.yaml")
    monkeypatch.setattr(trust_module, "DEFAULT_LOCK", workspace / "locks/trust-topology.lock.json")
    return trust_module


def test_trust_topology_requires_external_key_and_detects_mutation(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "repo"
    shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    _minimal_trust_artifacts(workspace)
    private_path, public_path = _write_operator_keys(tmp_path)
    trust_module = _configure_trust_module(monkeypatch, workspace)
    trust_module.create("run-trust-test", "operator-test", private_path, public_path)
    assert trust_module.verify(public_key_path=public_path) == []
    (workspace / "reports/resource-plan.json").write_text('{"plan":"builder-substituted"}\n')
    assert "trust artifact hash mismatch: reports/resource-plan.json" in trust_module.verify(public_key_path=public_path)


def test_trust_topology_rejects_keys_inside_repository(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "repo-inside-key"
    shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    _minimal_trust_artifacts(workspace)
    private_path, public_path = _write_operator_keys(workspace / "locks")
    trust_module = _configure_trust_module(monkeypatch, workspace)
    try:
        trust_module.create("run-inside-key", "operator-test", private_path, public_path)
    except ValueError as exc:
        assert "outside the repository trust boundary" in str(exc)
    else:
        raise AssertionError("repository-local operator keys were accepted")



def test_trust_topology_lock_is_append_only(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "repo-immutable-trust"
    shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    _minimal_trust_artifacts(workspace)
    private_path, public_path = _write_operator_keys(tmp_path / "outside")
    trust_module = _configure_trust_module(monkeypatch, workspace)
    trust_module.create("run-trust-test", "operator-test", private_path, public_path)
    try:
        trust_module.create("run-trust-test", "operator-test", private_path, public_path)
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("existing Trust Topology lock was overwritten")

def test_scientific_lock_allows_p03_builders_but_protects_trust_and_runtime_manifests(tmp_path: Path) -> None:
    policy = yaml.safe_load((ROOT / "docs/operator/scientific_lock_v1.yaml").read_text())
    paths = set(policy["protected_paths"])
    assert "analysis/**" not in paths
    assert "analysis/decision_gates.py" in paths and "analysis/sample_size.py" in paths
    for rel in (
        "locks/trust-topology.lock.json", "reports/resource-plan.json", "locks/public-keys.json",
        "locks/environment.lock.json", "locks/llm_model_lock.json", "locks/semantic_target_model_lock.json",
    ):
        assert rel in paths


def test_later_phases_run_locked_code_instead_of_implementing_it() -> None:
    registry = yaml.safe_load((ROOT / "docs/operator/phase_registry_v1.yaml").read_text())
    by = {row["phase_id"]: row for row in registry["phases"]}
    p03 = " ".join(by["P03"]["actions"]).lower()
    p06 = " ".join(by["P06"]["actions"]).lower()
    p10 = " ".join(by["P10"]["actions"]).lower()
    p15 = " ".join(by["P15"]["actions"]).lower()
    assert "prototype-bank builder" in p03 and "control-certification engine" in p03
    assert "every protected executable path" in p06
    assert "do not add or modify" in p10
    assert "source-code changes are forbidden" in p15


def test_scripts_verify_gate_is_only_a_thin_wrapper() -> None:
    text = (ROOT / "scripts/verify_gate.py").read_text(encoding="utf-8")
    assert "from validation.verify_gate import main" in text
    assert "def main" not in text


def test_scientific_lock_accepts_p03_builders_and_rejects_trust_runtime_mutations(
    tmp_path: Path, monkeypatch
) -> None:
    import validation.verify_lock as lock_module

    workspace = tmp_path / "repo-lock"
    shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    monkeypatch.setattr(lock_module, "ROOT", workspace)
    monkeypatch.setattr(lock_module, "POLICIES", {
        "scientific": workspace / "docs/operator/scientific_lock_v1.yaml",
        "implementation": workspace / "docs/operator/implementation_lock_v1.yaml",
    })
    monkeypatch.setattr(lock_module, "DEFAULT_LOCKS", {
        "scientific": workspace / "locks/scientific.lock.json",
        "implementation": workspace / "locks/implementation.lock.json",
    })
    runtime_files = {
        "locks/trust-topology.lock.json": b'{"trust":"signed-before-P02"}\n',
        "reports/resource-plan.json": b'{"roles":"operator-approved"}\n',
        "locks/infrastructure-plan.json": b'{"infra":"isolated"}\n',
        "locks/public-keys.json": b'{"keys":"attested"}\n',
        "locks/environment.lock.json": b'{"runtime":"pinned"}\n',
        "locks/llm_model_lock.json": b'{"model":"immutable"}\n',
        "locks/semantic_target_model_lock.json": b'{"semantic":"immutable"}\n',
    }
    for rel, raw in runtime_files.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    lock_module.create("scientific", lock_module.DEFAULT_LOCKS["scientific"], "run-lock-test")
    # P03 creates these after Scientific lock; they must not alter the Scientific path set.
    (workspace / "analysis/build_analysis_input.py").write_text("# implementation-locked builder\n")
    (workspace / "analysis/build_sample_size_input.py").write_text("# implementation-locked builder\n")
    assert lock_module.verify("scientific", lock_module.DEFAULT_LOCKS["scientific"]) == []
    # Every trust/runtime root artifact is covered and mutation fails closed.
    for rel, original in runtime_files.items():
        path = workspace / rel
        path.write_bytes(original + b"mutation")
        errors = lock_module.verify("scientific", lock_module.DEFAULT_LOCKS["scientific"])
        assert f"hash mismatch: {rel}" in errors
        path.write_bytes(original)
        assert lock_module.verify("scientific", lock_module.DEFAULT_LOCKS["scientific"]) == []


def test_compatibility_entrypoints_load_canonical_modules() -> None:
    expected = {
        "scripts/verify_gate.py": "from validation.verify_gate import main",
        "scripts/verify_trust_topology.py": "from validation.trust_topology_validator import main",
        "scripts/phase_check_runner.py": "from validation.phase_check_runner import main",
    }
    for rel, import_line in expected.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert import_line in text
        assert "def main" not in text
        assert "raise SystemExit(main())" in text

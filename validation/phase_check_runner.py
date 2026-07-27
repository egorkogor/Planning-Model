from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.role_validator import validate_role_independence
from validation.statistical_audit_validator import validate_statistical_audit
from validation.implementation_audit_validator import validate_implementation_audit
from validation.implementation_candidate_validator import validate_candidate
from validation.signature_validator import verify_signed_manifest
from validation.hashing import hash_json
from validation.verify_lock import verify as verify_lock
from validation.trust_topology_validator import verify as verify_trust_topology
from validation.full_plan_lineage_validator import validate_lineage_index
from validation.capacity_validator import validate_capacity_preflight, validate_compute_profile
from validation.random_codebook_validator import validate_random_codebook, validate_signature_bank


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def schema_registry():
    registry = Registry(); schemas = {}
    for path in sorted((ROOT / "docs/schemas").glob("*.json")):
        obj = load_json(path); schemas[path.name] = obj
        registry = registry.with_resource(obj["$id"], Resource.from_contents(obj))
    return schemas, registry


def validate_schema(obj, name: str) -> list[str]:
    schemas, registry = schema_registry()
    return [e.message for e in Draft202012Validator(schemas[name], registry=registry, format_checker=FormatChecker()).iter_errors(obj)]


def report_check(report: dict, check_id: str) -> dict | None:
    return next((x for x in report.get("checks", []) if x.get("check_id") == check_id), None)


def core_check(kind: str, phase: str, check_id: str, report: dict) -> list[str]:
    errors: list[str] = []
    if kind == "bootstrap_manifest":
        cp = subprocess.run([sys.executable, str(ROOT / "validation/verify_release_manifest.py")], cwd=ROOT, capture_output=True)
        if cp.returncode: errors.append("bootstrap manifest mismatch")
    elif kind == "scope_version":
        path = ROOT / "artifacts/scope.md"
        if not path.is_file(): errors.append("scope artifact missing")
        else:
            text = path.read_text(encoding="utf-8")
            if "work-planner/1.14" not in text or "runbook 2.14" not in text: errors.append("scope version markers missing")
    elif kind == "confirmatory_absent":
        forbidden = []
        for base in (ROOT / "results", ROOT / "sealed"):
            if base.exists():
                forbidden.extend(p for p in base.rglob("*") if p.is_file() and p.name != ".gitkeep")
        if forbidden: errors.append("confirmatory artifacts already present")
    elif kind == "evidence_sealed":
        row = report_check(report, check_id)
        if row is None or not row.get("evidence"): errors.append("sealed evidence missing")
        else:
            arts = {a["path"]: a["sha256"] for a in report.get("artifacts", [])}
            for rel in row["evidence"]:
                path = ROOT / rel
                if rel not in arts or not path.is_file() or sha(path) != arts[rel]: errors.append(f"unsealed evidence: {rel}")
    elif kind == "resource_plan_schema":
        path = ROOT / "reports/resource-plan.json"
        if not path.is_file(): errors.append("resource plan missing")
        else: errors.extend(validate_schema(load_json(path), "resource_plan.schema.json"))
    elif kind == "role_independence":
        try: errors.extend(validate_role_independence(load_json(ROOT / "reports/resource-plan.json"), load_json(ROOT / "locks/public-keys.json")))
        except Exception as exc: errors.append(str(exc))
    elif kind == "budget_limit":
        try:
            plan = load_json(ROOT / "reports/resource-plan.json")
            contract = yaml.safe_load((ROOT / "docs/infrastructure/provisioning_contract_v1.yaml").read_text(encoding="utf-8"))
            hard = contract.get("budget", {}).get("maximum_total")
            if hard is None: errors.append("hard budget is not defined")
            elif float(plan["estimated_cost"]) > float(hard): errors.append("estimated cost exceeds hard budget")
        except Exception as exc: errors.append(str(exc))
    elif kind == "public_key_binding":
        try:
            plan=load_json(ROOT / "reports/resource-plan.json"); keys=load_json(ROOT / "locks/public-keys.json")
            errors.extend(validate_schema(keys,"public_key_registry.schema.json"))
            errors.extend(validate_role_independence(plan,keys))
        except Exception as exc: errors.append(str(exc))
    elif kind == "immutable_model_revisions":
        for rel in ("locks/llm_model_lock.json", "locks/semantic_target_model_lock.json"):
            path=ROOT/rel
            if not path.is_file(): errors.append(f"missing {rel}"); continue
            obj=load_json(path); errors.extend(validate_schema(obj,"model_lock.schema.json"))
            rev=str(obj.get("revision", ""))
            if not rev or rev.lower() in {"main","latest","master"}: errors.append(f"mutable model revision in {rel}")
    elif kind == "bundle_tests":
        env = {**__import__("os").environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        static = subprocess.run(
            [sys.executable, "validation/validate_bundle.py", "--skip-nested-pytest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        if static.returncode:
            errors.append(f"bundle static validation failed: {static.stderr.strip() or static.stdout.strip()}")
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "validation", "--basetemp=.phase-p02-pytest"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        if tests.returncode:
            errors.append(f"bundle pytest failed: {tests.stderr.strip() or tests.stdout.strip()}")
        import shutil
        shutil.rmtree(ROOT / ".phase-p02-pytest", ignore_errors=True)
        shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)
    elif kind == "scientific_lock":
        lock=ROOT/"locks/scientific.lock.json"
        if not lock.is_file(): errors.append("scientific lock missing")
        else: errors.extend(verify_lock("scientific", lock, report.get("run_id")))
    elif kind == "implementation_lock":
        lock=ROOT/"locks/implementation.lock.json"
        if not lock.is_file(): errors.append("implementation lock missing")
        else: errors.extend(verify_lock("implementation", lock, report.get("run_id")))
    elif kind == "trust_topology_lock":
        errors.extend(verify_trust_topology(expected_run_id=report.get("run_id")))
    elif kind == "operator_identity":
        try:
            trust=load_json(ROOT / "locks/trust-topology.lock.json")
            if report.get("executor_identity_sha256") != trust.get("operator_public_key_sha256"):
                errors.append("operator executor identity differs from Trust Topology operator key")
        except Exception as exc:
            errors.append(str(exc))
    elif kind == "runtime_stack_lock":
        try:
            path = ROOT / "locks/environment.lock.json"
            if not path.is_file():
                errors.append("runtime stack lock missing")
            else:
                obj = load_json(path)
                errors.extend(validate_schema(obj, "runtime_stack_lock.schema.json"))
                payload = dict(obj); payload.pop("lock_hash", None)
                if obj.get("lock_hash") != hash_json(payload): errors.append("runtime stack lock_hash mismatch")
                names = {row["name"]: row for row in obj.get("packages", [])}
                for required_name in ("torch", "transformers", "sentence-transformers"):
                    if required_name not in names: errors.append(f"runtime stack missing {required_name}")
                runtime = obj.get("model_runtime", {})
                if runtime.get("trust_remote_code") is not False:
                    errors.append("trust_remote_code must be false")
                expected_profiles = {"stage1a_executor": 64, "stage1b_executor": 128, "self_plan_generator": 128}
                profiles = runtime.get("generation_profiles", {})
                for profile_name, expected_tokens in expected_profiles.items():
                    profile = profiles.get(profile_name, {})
                    if profile.get("max_new_tokens") != expected_tokens:
                        errors.append(f"{profile_name} max_new_tokens must equal {expected_tokens}")
                    if profile.get("do_sample") is not False or profile.get("num_beams") != 1:
                        errors.append(f"{profile_name} decoding must be deterministic")
                if obj.get("inference_backend") == "VLLM" and "vllm" not in names:
                    errors.append("VLLM backend selected without pinned vllm package")
                for rel, field in (("locks/llm_model_lock.json", "llm_model_aggregate_sha256"), ("locks/semantic_target_model_lock.json", "semantic_model_aggregate_sha256")):
                    model = load_json(ROOT / rel)
                    if obj.get(field) != model.get("aggregate_sha256"): errors.append(f"{field} not bound to {rel}")
        except Exception as exc: errors.append(str(exc))
    elif kind == "implementation_candidate":
        try:
            path = ROOT / "freezes/implementation-lock.candidate.json"
            if not path.is_file():
                errors.append("implementation-lock candidate missing")
            else:
                obj = load_json(path)
                errors.extend(validate_schema(obj, "implementation_lock_candidate.schema.json"))
                errors.extend(validate_candidate(obj))
                if obj.get("run_id") != report.get("run_id"):
                    errors.append("implementation candidate run_id differs from phase report")
                if obj.get("reviewed_commit") != report.get("implementation_commit"):
                    errors.append("implementation candidate reviewed_commit differs from phase implementation_commit")
        except Exception as exc: errors.append(str(exc))
    elif kind in {"statistical_audit", "implementation_audit"}:
        try:
            resource = load_json(ROOT / "reports/resource-plan.json")
            if kind == "statistical_audit":
                rel, schema, role = "reports/statistical-implementation-audit.json", "statistical_audit.schema.json", "STATISTICAL_REVIEWER"
                obj = load_json(ROOT / rel); errors.extend(validate_schema(obj, schema)); errors.extend(validate_statistical_audit(obj, resource))
            else:
                rel, schema, role = "reports/independent-implementation-audit.json", "implementation_audit.schema.json", "AUDITOR"
                obj = load_json(ROOT / rel); errors.extend(validate_schema(obj, schema)); errors.extend(validate_implementation_audit(obj, resource))
            errors.extend(verify_signed_manifest(obj, role, hash_field="report_hash"))
            if obj.get("decision") != "APPROVE": errors.append(f"{kind} decision is not APPROVE")
            if obj.get("run_id") != report.get("run_id"):
                errors.append(f"{kind} run_id differs from phase report")
            if obj.get("reviewed_commit") != report.get("implementation_commit"):
                errors.append(f"{kind} reviewed_commit differs from phase implementation_commit")
        except Exception as exc: errors.append(str(exc))
    elif kind == "random_codebook":
        try:
            bank_path = ROOT / "semantic_bank/signatures/manifest.json"
            codebook_path = ROOT / "semantic_bank/random-codebook/manifest.json"
            if not bank_path.is_file() or not codebook_path.is_file():
                errors.append("semantic signature bank or random codebook manifest missing")
            else:
                bank = load_json(bank_path); codebook = load_json(codebook_path)
                errors.extend(validate_schema(bank, "semantic_signature_bank.schema.json"))
                errors.extend(validate_signature_bank(bank))
                errors.extend(validate_schema(codebook, "random_codebook_manifest.schema.json"))
                errors.extend(validate_random_codebook(ROOT, codebook))
                if bank.get("run_id") != report.get("run_id") or codebook.get("run_id") != report.get("run_id"):
                    errors.append("random codebook lineage run_id differs from phase report")
        except Exception as exc:
            errors.append(str(exc))
    elif kind == "capacity_preflight":
        try:
            path = ROOT / "reports/preflight-final.json"
            if not path.is_file():
                errors.append("capacity preflight report missing")
            else:
                obj = load_json(path)
                errors.extend(validate_schema(obj, "capacity_preflight.schema.json"))
                errors.extend(validate_capacity_preflight(ROOT, obj))
                if obj.get("run_id") != report.get("run_id"):
                    errors.append("capacity preflight run_id differs from phase report")
                compute = load_json(ROOT / "reports/compute-profile.json")
                errors.extend(validate_schema(compute, "compute_profile.schema.json"))
                errors.extend(validate_compute_profile(compute, ROOT))
        except Exception as exc:
            errors.append(str(exc))
    elif kind in {"planner_confirmatory_full_plan_lineage", "stage1b_full_plan_lineage"}:
        try:
            if kind == "planner_confirmatory_full_plan_lineage":
                rel, expected_stage = "results/planner-confirmatory/lineage-index.json", "PLANNER"
            else:
                rel, expected_stage = "results/stage1b-confirmatory/lineage-index.json", "STAGE1B"
            path = ROOT / rel
            if not path.is_file():
                errors.append(f"full-plan lineage index missing: {rel}")
            else:
                obj = load_json(path)
                errors.extend(validate_schema(obj, "full_plan_lineage_index.schema.json"))
                errors.extend(validate_lineage_index(ROOT, obj, expected_stage=expected_stage))
                if obj.get("run_id") != report.get("run_id"):
                    errors.append("lineage index run_id differs from phase report")
        except Exception as exc:
            errors.append(str(exc))
    elif kind == "bootstrap_covers_trust_verifiers":
        try:
            manifest=load_json(ROOT/"release/BOOTSTRAP_MANIFEST.json")
            paths=set(manifest["files"]) if isinstance(manifest["files"], dict) else {x["path"] for x in manifest["files"]}
            required={
                "validation/verify_release_manifest.py",
                "validation/verify_lock.py",
                "validation/verify_gate.py",
                "validation/phase_check_runner.py",
                "validation/implementation_audit_validator.py",
                "validation/trust_topology_validator.py",
                "validation/operator_decision_validator.py",
                "validation/confirmatory_lineage_validator.py",
                "validation/code_fingerprint.py",
                "validation/full_plan_lineage_validator.py",
                "validation/capacity_validator.py",
                "validation/random_codebook_validator.py",
            }
            missing=required-paths
            if missing: errors.append(f"bootstrap omits trust verifiers: {sorted(missing)}")
        except Exception as exc: errors.append(str(exc))
    else:
        errors.append(f"unknown core check kind: {kind}")
    return errors


def runtime_check(phase: str, check_id: str, report_path: Path, contract: dict) -> list[str]:
    template=contract["runtime_checker"]["command_template"]
    command=template.format(phase_id=phase,check_id=check_id)
    cp=subprocess.run(command.split(),cwd=ROOT,text=True,capture_output=True)
    if cp.returncode: return [f"runtime checker failed: {command}: {cp.stderr.strip() or cp.stdout.strip()}"]
    try: obj=json.loads(cp.stdout)
    except Exception as exc: return [f"runtime checker did not return JSON: {exc}"]
    errors=validate_schema(obj,"machine_check_result.schema.json")
    if obj.get("phase_id")!=phase or obj.get("check_id")!=check_id: errors.append("runtime checker phase/check mismatch")
    if obj.get("status")!="PASS": errors.append("runtime checker returned FAIL")
    return errors


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",required=True); ap.add_argument("--check",required=True); ap.add_argument("--report",type=Path,required=True); args=ap.parse_args()
    contract=yaml.safe_load((ROOT/"docs/operator/phase_check_contract_v1.yaml").read_text(encoding="utf-8"))
    try: report=load_json(ROOT/args.report if not args.report.is_absolute() else args.report)
    except Exception as exc: print(f"phase report unavailable: {exc}"); return 2
    kind=contract.get("core_checks",{}).get(args.check)
    errors=core_check(kind,args.phase,args.check,report) if kind else runtime_check(args.phase,args.check,args.report,contract)
    if errors:
        print("\n".join(errors)); return 2
    print("MACHINE_CHECK_VERIFIED"); return 0

if __name__ == "__main__":
    raise SystemExit(main())

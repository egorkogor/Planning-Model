from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

try:
    from .domain_oracle import oracle_distance as default_oracle_distance
    from .hashing import (
        canonical_task_hash,
        goal_hash,
        manifest_artifact_hash,
        manifest_content_hash,
        pair_group_hash,
        plan_artifact_hash,
        plan_content_hash,
        prompt_bytes_hash,
        state_hash,
        token_vector_hash,
    )
except ImportError:  # pragma: no cover - allows direct script use
    from domain_oracle import oracle_distance as default_oracle_distance
    from hashing import (
        canonical_task_hash,
        goal_hash,
        manifest_artifact_hash,
        manifest_content_hash,
        pair_group_hash,
        plan_artifact_hash,
        plan_content_hash,
        prompt_bytes_hash,
        state_hash,
        token_vector_hash,
    )


@dataclass(frozen=True)
class ContractViolation:
    code: str
    path: str
    message: str


class CrossObjectContractValidator:
    """Normative validation for relations that JSON Schema cannot express.

    JSON Schema remains the local shape/type layer. This validator is the
    mandatory second layer and must run before an artifact is persisted.
    """

    VARIANT_REPRESENTATION = {
        "A1": "TOKEN_GRAMMAR",
        "A2": "TYPED_ONLY",
        "A2b": "DISCRETE_INTENT",
        "A2c": "STRUCTURED_DISCRETE",
        "A3": "CONTINUOUS_LATENT",
        "A4": "CONTINUOUS_LATENT",
        "A5": "CONTINUOUS_LATENT",
    }
    INTERFACE_ARMS = {
        "I0_EQUAL_TOKENS_RAW",
        "I1_ORACLE_CURRENT_RAW",
        "I2_SHUFFLED_RAW",
        "I3_PLANNER_CURRENT_RAW",
    }
    END_TO_END_ARMS = {
        "E0_EQUAL_TOKENS_RAW",
        "E1_PLANNER_CURRENT_RAW",
        "E2_SHUFFLED_RAW",
        "E3_ORACLE_REPLAN_RAW",
        "P_REPLAY_RAW",
        "E0_EQUAL_TOKENS_OPERATIONAL",
        "E1_PLANNER_CURRENT_OPERATIONAL",
    }
    PLANNER_INTENT_ARMS = {
        "I3_PLANNER_CURRENT_RAW",
        "E1_PLANNER_CURRENT_RAW",
        "E1_PLANNER_CURRENT_OPERATIONAL",
    }
    ORACLE_INTENT_ARMS = {"I1_ORACLE_CURRENT_RAW", "E3_ORACLE_REPLAN_RAW"}
    SHUFFLED_ARMS = {"I2_SHUFFLED_RAW", "E2_SHUFFLED_RAW"}
    EQUAL_TOKEN_ARMS = {
        "I0_EQUAL_TOKENS_RAW",
        "E0_EQUAL_TOKENS_RAW",
        "E0_EQUAL_TOKENS_OPERATIONAL",
    }
    LLM_ARMS = (INTERFACE_ARMS | END_TO_END_ARMS) - {"P_REPLAY_RAW"}
    EXPECTED_SOURCE = {
        "I0_EQUAL_TOKENS_RAW": "EQUAL_TOKEN_LLM",
        "I1_ORACLE_CURRENT_RAW": "ORACLE_INTENT_LLM",
        "I2_SHUFFLED_RAW": "SHUFFLED_INTENT_LLM",
        "I3_PLANNER_CURRENT_RAW": "PLANNER_INTENT_LLM",
        "E0_EQUAL_TOKENS_RAW": "EQUAL_TOKEN_LLM",
        "E1_PLANNER_CURRENT_RAW": "PLANNER_INTENT_LLM",
        "E2_SHUFFLED_RAW": "SHUFFLED_INTENT_LLM",
        "E3_ORACLE_REPLAN_RAW": "ORACLE_INTENT_LLM",
        "P_REPLAY_RAW": "PLAN_REPLAY",
        "E0_EQUAL_TOKENS_OPERATIONAL": "EQUAL_TOKEN_LLM",
        "E1_PLANNER_CURRENT_OPERATIONAL": "PLANNER_INTENT_LLM",
    }

    def __init__(
        self,
        oracle_distance_fn: Callable[[Iterable[Iterable[str]], Iterable[Iterable[str]], int], int | None] = default_oracle_distance,
        artifact_bytes_loader: Callable[[str], bytes] | None = None,
    ) -> None:
        self.oracle_distance_fn = oracle_distance_fn
        self.artifact_bytes_loader = artifact_bytes_loader

    @staticmethod
    def _v(code: str, path: str, message: str) -> ContractViolation:
        return ContractViolation(code, path, message)

    @staticmethod
    def _predicate_refs(predicates: Iterable[Sequence[Any]]) -> set[str]:
        return {
            value
            for fact in predicates
            for value in fact[1:]
            if isinstance(value, str) and value.startswith("@B")
        }

    @staticmethod
    def _sha_from_semantic_ref(ref: str) -> str:
        return "sha256:" + urlparse(ref).netloc

    @staticmethod
    def _nullable_fields_present(obj: Mapping[str, Any], fields: Iterable[str]) -> bool:
        return any(obj.get(field) is not None for field in fields)

    def validate_domain_contract(self, domain: Mapping[str, Any]) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        expected_predicates = {
            "ON": (2, ("moving", "support")),
            "ON_TABLE": (1, ("block",)),
            "CLEAR": (1, ("block",)),
            "HOLDING": (1, ("block",)),
            "HAND_EMPTY": (0, ()),
        }
        actual: dict[str, tuple[int, tuple[str, ...]]] = {}
        for index, predicate in enumerate(domain["state_representation"]["positive_predicates"]):
            if not isinstance(predicate.get("name"), str):
                out.append(self._v("CONTRACT_VIOLATION", f"positive_predicates[{index}]", "predicate name must be a string"))
                continue
            actual[predicate["name"]] = (predicate["arity"], tuple(predicate["roles"]))
        if actual != expected_predicates:
            out.append(self._v("CONTRACT_VIOLATION", "positive_predicates", f"expected {expected_predicates}, got {actual}"))

        expected_actions = {
            "PICK_UP": {
                "rank": 0,
                "args": (("block", "BLOCK"),),
                "pre": (("ON_TABLE", ("block",)), ("CLEAR", ("block",)), ("HAND_EMPTY", ())),
                "add": (("HOLDING", ("block",)),),
                "delete": (("ON_TABLE", ("block",)), ("CLEAR", ("block",)), ("HAND_EMPTY", ())),
            },
            "UNSTACK": {
                "rank": 1,
                "args": (("moving", "BLOCK"), ("support", "BLOCK")),
                "pre": (("ON", ("moving", "support")), ("CLEAR", ("moving",)), ("HAND_EMPTY", ())),
                "add": (("HOLDING", ("moving",)), ("CLEAR", ("support",))),
                "delete": (("ON", ("moving", "support")), ("CLEAR", ("moving",)), ("HAND_EMPTY", ())),
            },
            "PUT_DOWN": {
                "rank": 2,
                "args": (("block", "BLOCK"),),
                "pre": (("HOLDING", ("block",)),),
                "add": (("ON_TABLE", ("block",)), ("CLEAR", ("block",)), ("HAND_EMPTY", ())),
                "delete": (("HOLDING", ("block",)),),
            },
            "STACK": {
                "rank": 3,
                "args": (("moving", "BLOCK"), ("support", "BLOCK")),
                "pre": (("HOLDING", ("moving",)), ("CLEAR", ("support",))),
                "add": (("ON", ("moving", "support")), ("CLEAR", ("moving",)), ("HAND_EMPTY", ())),
                "delete": (("HOLDING", ("moving",)), ("CLEAR", ("support",))),
            },
            "END": {"rank": 4, "args": (), "pre": (), "add": (), "delete": ()},
        }

        def facts(items: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
            converted: list[tuple[str, tuple[str, ...]]] = []
            for item in items:
                predicate = item.get("predicate")
                args = item.get("args")
                if not isinstance(predicate, str) or not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                    converted.append(("<INVALID>", ()))
                else:
                    converted.append((predicate, tuple(args)))
            return tuple(converted)

        action_map = {a["name"]: a for a in domain["actions"]}
        if set(action_map) != set(expected_actions):
            out.append(self._v("CONTRACT_VIOLATION", "actions", "action registry differs from normative registry"))
        for name, expected in expected_actions.items():
            action = action_map.get(name)
            if action is None:
                continue
            actual_action = {
                "rank": action["rank"],
                "args": tuple((a["role"], a["type"]) for a in action["arguments"]),
                "pre": facts(action.get("preconditions", [])),
                "add": facts(action.get("add_effects", [])),
                "delete": facts(action.get("delete_effects", [])),
            }
            if actual_action != expected:
                out.append(self._v("CONTRACT_VIOLATION", f"actions.{name}", f"expected {expected}, got {actual_action}"))
        if domain["goal_semantics"].get("allowed_predicates") != ["ON", "ON_TABLE"]:
            out.append(self._v("CONTRACT_VIOLATION", "goal_semantics.allowed_predicates", "must be exactly [ON, ON_TABLE]"))
        return out

    def _validate_goal(self, goal: list[list[str]], ledger: set[str]) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        on: list[tuple[str, str]] = []
        on_table: set[str] = set()
        for index, fact in enumerate(goal):
            if fact[0] not in {"ON", "ON_TABLE"}:
                out.append(self._v("CONTRACT_VIOLATION", f"goal[{index}]", "goal permits only ON/ON_TABLE"))
                continue
            for ref in fact[1:]:
                if ref not in ledger:
                    out.append(self._v("UNKNOWN_REF", f"goal[{index}]", f"{ref} absent from ledger"))
            if fact[0] == "ON":
                moving, support = fact[1], fact[2]
                if moving == support:
                    out.append(self._v("INVALID_ACTION", f"goal[{index}]", "ON(x,x) is invalid"))
                on.append((moving, support))
            else:
                on_table.add(fact[1])
        by_moving: dict[str, set[str]] = {}
        by_support: dict[str, set[str]] = {}
        for moving, support in on:
            by_moving.setdefault(moving, set()).add(support)
            by_support.setdefault(support, set()).add(moving)
        for moving, supports in by_moving.items():
            if len(supports) > 1:
                out.append(self._v("CONTRACT_VIOLATION", "goal", f"{moving} has multiple goal supports"))
            if moving in on_table:
                out.append(self._v("CONTRACT_VIOLATION", "goal", f"{moving} is both ON_TABLE and ON"))
        for support, moving_blocks in by_support.items():
            if len(moving_blocks) > 1:
                out.append(self._v("CONTRACT_VIOLATION", "goal", f"multiple blocks are directly ON {support}"))
        parent = {moving: next(iter(supports)) for moving, supports in by_moving.items() if len(supports) == 1}
        for start in ledger:
            seen: set[str] = set()
            current = start
            while current in parent:
                if current in seen:
                    out.append(self._v("CONTRACT_VIOLATION", "goal", f"ON cycle contains {current}"))
                    break
                seen.add(current)
                current = parent[current]
        return out

    def _validate_state(self, facts: set[tuple[Any, ...]], ledger: set[str], path: str) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        for fact in facts:
            for ref in fact[1:]:
                if ref not in ledger:
                    out.append(self._v("UNKNOWN_REF", path, f"{ref} absent from ledger"))
        on = [(f[1], f[2]) for f in facts if f[0] == "ON"]
        on_table = {f[1] for f in facts if f[0] == "ON_TABLE"}
        holding = {f[1] for f in facts if f[0] == "HOLDING"}
        clear = {f[1] for f in facts if f[0] == "CLEAR"}
        hand_empty = ("HAND_EMPTY",) in facts
        moving_supports: dict[str, set[str]] = {}
        support_movers: dict[str, set[str]] = {}
        for moving, support in on:
            if moving == support:
                out.append(self._v("CONTRACT_VIOLATION", path, "ON graph contains self edge"))
            moving_supports.setdefault(moving, set()).add(support)
            support_movers.setdefault(support, set()).add(moving)
        for moving, supports in moving_supports.items():
            if len(supports) > 1:
                out.append(self._v("CONTRACT_VIOLATION", path, f"{moving} has {len(supports)} supports"))
        for support, moving_blocks in support_movers.items():
            if len(moving_blocks) > 1:
                out.append(self._v("CONTRACT_VIOLATION", path, f"{len(moving_blocks)} blocks are directly ON {support}"))
        if len(holding) > 1:
            out.append(self._v("CONTRACT_VIOLATION", path, "at most one HOLDING fact is allowed"))
        if hand_empty != (len(holding) == 0):
            out.append(self._v("CONTRACT_VIOLATION", path, "HAND_EMPTY iff no HOLDING"))
        parent = {moving: next(iter(supports)) for moving, supports in moving_supports.items() if len(supports) == 1}
        for start in ledger:
            seen: set[str] = set()
            current = start
            while current in parent:
                if current in seen:
                    out.append(self._v("CONTRACT_VIOLATION", path, f"ON cycle contains {current}"))
                    break
                seen.add(current)
                current = parent[current]
        for block in ledger:
            modes = int(block in on_table) + int(block in moving_supports) + int(block in holding)
            if modes != 1:
                out.append(self._v("CONTRACT_VIOLATION", path, f"{block} has {modes} location modes"))
            should_clear = not support_movers.get(block) and block not in holding
            if (block in clear) != should_clear:
                out.append(self._v("CONTRACT_VIOLATION", path, f"CLEAR({block}) is inconsistent"))
        return out

    def validate_task(self, task: dict[str, Any], *, verify_hash: bool = True) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        ledger_entries = task["ledger"]
        ledger = set(ledger_entries)
        expected_refs = {f"@B{i}" for i in range(len(ledger_entries))}
        if ledger != expected_refs:
            out.append(self._v("CONTRACT_VIOLATION", "ledger", f"refs must be contiguous {sorted(expected_refs)}"))
        aliases = [entry["surface_alias"] for entry in ledger_entries.values()]
        if len(aliases) != len(set(aliases)):
            out.append(self._v("AMBIGUOUS_ALIAS", "ledger", "surface aliases must be unique"))
        for ref, entry in ledger_entries.items():
            if ref.startswith("@B"):
                expected_alias = f"block_{int(ref[2:])}"
                if entry["surface_alias"] != expected_alias:
                    out.append(self._v("CONTRACT_VIOLATION", f"ledger.{ref}.surface_alias", f"must equal {expected_alias}"))
        difficulty = task["difficulty"]
        if difficulty["block_count"] != len(ledger):
            out.append(self._v("CONTRACT_VIOLATION", "difficulty.block_count", "must equal len(ledger)"))
        ranges = {"short_1_5": (1, 5), "medium_6_10": (6, 10), "long_11_16": (11, 16)}
        low, high = ranges[difficulty["horizon_bucket"]]
        if not low <= difficulty["oracle_length"] <= high:
            out.append(self._v("CONTRACT_VIOLATION", "difficulty", "oracle length lies outside bucket"))
        out.extend(self._validate_state({tuple(x) for x in task["initial"]}, ledger, "initial"))
        out.extend(self._validate_goal(task["goal"], ledger))
        if verify_hash and task.get("canonical_task_hash") != canonical_task_hash(task):
            out.append(self._v("HASH_MISMATCH", "canonical_task_hash", "does not match canonical task payload"))
        if not out:
            actual = self.oracle_distance_fn(task["initial"], task["goal"], 16)
            if actual is None:
                out.append(self._v("GOAL_NOT_REACHED", "goal", "normative oracle found no plan within 16 actions"))
            elif actual != difficulty["oracle_length"]:
                out.append(self._v("CONTRACT_VIOLATION", "difficulty.oracle_length", f"reported {difficulty['oracle_length']}, actual {actual}"))
        return out

    def validate_typed_action(self, action: dict[str, Any], ledger: set[str] | None = None) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        refs = [arg["ref"] for arg in action["args"]]
        if ledger is not None:
            for ref in refs:
                if ref not in ledger:
                    out.append(self._v("UNKNOWN_REF", "args", f"{ref} absent from ledger"))
        if action["action"] in {"STACK", "UNSTACK"} and len(refs) == 2 and refs[0] == refs[1]:
            out.append(self._v("INVALID_ACTION", "args", "moving and support must differ"))
        return out

    def validate_planner_step(self, step: dict[str, Any], ledger: set[str] | None = None) -> list[ContractViolation]:
        out = self.validate_typed_action(step["typed_action"], ledger)
        expected_step_id = f"S{step['step_index']:02d}"
        if step["step_id"] != expected_step_id:
            out.append(self._v("CONTRACT_VIOLATION", "step_id", f"expected {expected_step_id}"))
        expected_representation = self.VARIANT_REPRESENTATION[step["planner_variant"]]
        if step["representation"] != expected_representation:
            out.append(self._v("CONTRACT_VIOLATION", "representation", f"expected {expected_representation}"))
        action = step["typed_action"]["action"]
        signature = step.get("semantic_signature")
        if signature is not None and step.get("intent_id") != signature.get("intent_id"):
            out.append(self._v("CONTRACT_VIOLATION", "semantic_signature.intent_id", "must equal top-level intent_id"))
        if action == "END":
            if self._nullable_fields_present(step, ("semantic_ref", "intent_id", "semantic_signature", "semantic_similarity", "semantic_margin")):
                out.append(self._v("CONTRACT_VIOLATION", "planner_step", "END semantic fields must be null"))
            return out
        rep = step["representation"]
        if rep == "CONTINUOUS_LATENT":
            if step.get("semantic_ref") is None or signature is None or step.get("intent_id") is None:
                out.append(self._v("CONTRACT_VIOLATION", "semantic_*", "continuous step requires ref, intent and signature"))
            elif urlparse(step["semantic_ref"]).fragment != step["step_id"]:
                out.append(self._v("CONTRACT_VIOLATION", "semantic_ref", "URI fragment must equal step_id"))
        elif rep == "STRUCTURED_DISCRETE":
            if step.get("semantic_ref") is not None or signature is None or step.get("intent_id") is None:
                out.append(self._v("CONTRACT_VIOLATION", "semantic_*", "A2c requires signature+intent and forbids semantic_ref"))
        elif rep == "DISCRETE_INTENT":
            if step.get("semantic_ref") is not None or step.get("intent_id") is None or signature is not None:
                out.append(self._v("CONTRACT_VIOLATION", "semantic_*", "A2b requires only intent_id"))
        else:
            if self._nullable_fields_present(step, ("semantic_ref", "intent_id", "semantic_signature", "semantic_similarity", "semantic_margin")):
                out.append(self._v("CONTRACT_VIOLATION", "semantic_*", "A1/A2 semantic fields must be null"))
        return out

    def validate_plan_manifest(self, plan: dict[str, Any], manifest: dict[str, Any] | None) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        steps = plan["steps"]
        indices = [step["step_index"] for step in steps]
        if indices != list(range(len(steps))):
            out.append(self._v("CONTRACT_VIOLATION", "steps", "indices must be contiguous from zero"))
        if len(steps) > 17:
            out.append(self._v("HORIZON_EXCEEDED", "steps", "at most 16 actions plus terminal END"))
        end_indices = [index for index, step in enumerate(steps) if step["typed_action"]["action"] == "END"]
        if end_indices != [len(steps) - 1]:
            out.append(self._v("CONTRACT_VIOLATION", "steps", "exactly one END must be the final PlannerStep"))
        if sum(step["typed_action"]["action"] != "END" for step in steps) > 16:
            out.append(self._v("HORIZON_EXCEEDED", "steps", "more than 16 non-END actions"))
        for step in steps:
            out.extend(self.validate_planner_step(step))
            if step["planner_variant"] != plan["planner_variant"] or step["representation"] != plan["representation"]:
                out.append(self._v("CONTRACT_VIOLATION", f"steps.{step['step_id']}", "step variant/representation differs from plan"))
        if plan["plan_content_hash"] != plan_content_hash(plan):
            out.append(self._v("HASH_MISMATCH", "plan_content_hash", "does not match normative payload"))
        if plan["plan_artifact_hash"] != plan_artifact_hash(plan):
            out.append(self._v("HASH_MISMATCH", "plan_artifact_hash", "does not match full artifact"))

        continuous = plan["representation"] == "CONTINUOUS_LATENT"
        if not continuous:
            if manifest is not None or plan["semantic_artifact_manifest_sha256"] is not None:
                out.append(self._v("CONTRACT_VIOLATION", "semantic_artifact_manifest", "non-continuous plan must not have semantic manifest"))
            return out
        if manifest is None:
            out.append(self._v("CONTRACT_VIOLATION", "semantic_artifact_manifest", "continuous plan requires manifest"))
            return out
        for field in ("task_id", "canonical_task_hash", "state_hash", "planner_checkpoint_sha256", "planner_config_sha256"):
            if plan[field] != manifest[field]:
                out.append(self._v("CONTRACT_VIOLATION", f"plan|manifest.{field}", f"{field} differs"))
        if manifest["manifest_content_hash"] != manifest_content_hash(manifest):
            out.append(self._v("HASH_MISMATCH", "manifest_content_hash", "does not match normative payload"))
        if manifest["manifest_hash"] != manifest_artifact_hash(manifest):
            out.append(self._v("HASH_MISMATCH", "manifest_hash", "does not match full manifest"))
        if plan["semantic_artifact_manifest_sha256"] != manifest["manifest_hash"]:
            out.append(self._v("HASH_MISMATCH", "semantic_artifact_manifest_sha256", "must equal manifest_hash"))
        artifacts = {artifact["step_id"]: artifact for artifact in manifest["artifacts"]}
        non_end_steps = [step for step in steps if step["typed_action"]["action"] != "END"]
        if set(artifacts) != {step["step_id"] for step in non_end_steps}:
            out.append(self._v("CONTRACT_VIOLATION", "manifest.artifacts", "must contain exactly one artifact per non-END step"))
        for step in non_end_steps:
            artifact = artifacts.get(step["step_id"])
            if artifact is None:
                continue
            expected = {
                "semantic_ref": step["semantic_ref"],
                "task_id": plan["task_id"],
                "state_hash": plan["state_hash"],
                "planner_checkpoint_sha256": plan["planner_checkpoint_sha256"],
                "planner_config_sha256": plan["planner_config_sha256"],
            }
            for field, value in expected.items():
                if artifact[field] != value:
                    out.append(self._v("CONTRACT_VIOLATION", f"artifacts.{step['step_id']}.{field}", f"expected {value}"))
            if self._sha_from_semantic_ref(artifact["semantic_ref"]) != artifact["tensor_sha256"]:
                out.append(self._v("HASH_MISMATCH", f"artifacts.{step['step_id']}.semantic_ref", "URI digest must equal tensor_sha256"))
        return out

    def _validate_llm_presence(self, attempt: dict[str, Any], *, llm_called: bool) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        mandatory_when_called = (
            "prompt_hash",
            "base_prompt_hash",
            "prompt_artifact",
            "prompt_token_ids_hash",
            "attention_mask_hash",
            "position_ids_hash",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "chat_template_hash",
            "prompt_tokens_total",
            "attended_prompt_tokens",
            "padded_sequence_length",
            "added_block_tokens",
            "guidance_block_token_ids_hash",
            "tokens_in",
            "tokens_out",
            "queue_ms",
            "inference_ms",
            "total_ms",
        )
        if llm_called:
            for field in mandatory_when_called:
                if attempt.get(field) is None:
                    out.append(self._v("CONTRACT_VIOLATION", field, "LLM call requires field"))
        else:
            for field in mandatory_when_called + ("raw_output", "raw_output_bytes_hash", "parsed_llm_response"):
                if attempt.get(field) is not None:
                    out.append(self._v("CONTRACT_VIOLATION", field, "field must be null when LLM is not called"))
        return out

    def validate_attempt(self, attempt: dict[str, Any]) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        stage, arm = attempt["stage"], attempt["arm"]
        if stage == "STAGE1A_INTERFACE":
            if arm not in self.INTERFACE_ARMS:
                out.append(self._v("CONTRACT_VIOLATION", "arm", "not a Stage1A arm"))
            if attempt["state_source"] != "ORACLE_SNAPSHOT" or attempt["snapshot_id"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "state_source", "Stage1A requires oracle snapshot"))
        elif stage == "STAGE1B_END_TO_END":
            if arm not in self.END_TO_END_ARMS:
                out.append(self._v("CONTRACT_VIOLATION", "arm", "not a Stage1B arm"))
            if attempt["state_source"] != "ACTUAL_TRAJECTORY" or attempt["snapshot_id"] is not None:
                out.append(self._v("CONTRACT_VIOLATION", "state_source", "Stage1B requires actual trajectory"))
        expected_policy = "oracle_snapshot" if stage == "STAGE1A_INTERFACE" else "independent_receding_horizon"
        if attempt["trajectory_policy"] != expected_policy:
            out.append(self._v("CONTRACT_VIOLATION", "trajectory_policy", f"expected {expected_policy}"))
        expected_pair_hash = pair_group_hash(
            stage=stage,
            task_id=attempt["task_id"],
            base_task_id=attempt["base_task_id"],
            split=attempt["split"],
            snapshot_id=attempt["snapshot_id"],
            trajectory_policy=attempt["trajectory_policy"],
            experiment_freeze_hash=attempt["experiment_freeze_hash"],
        )
        if attempt["pair_group_hash"] != expected_pair_hash:
            out.append(self._v("HASH_MISMATCH", "pair_group_hash", "does not match stage/task/snapshot"))
        if attempt["rollout_mode"] == "RAW" and (
            attempt["attempt_index"] != 0 or attempt["mask_applied"] or attempt["mask_hash"] is not None
        ):
            out.append(self._v("CONTRACT_VIOLATION", "rollout_mode", "RAW cannot retry or mask"))
        if (attempt["rollout_mode"] == "OPERATIONAL") != arm.endswith("OPERATIONAL"):
            out.append(self._v("CONTRACT_VIOLATION", "arm", "arm suffix and rollout mode disagree"))

        unresolved = attempt["semantic_resolution_status"] == "UNRESOLVED"
        expected_source = self.EXPECTED_SOURCE.get(arm)
        if unresolved and arm in self.PLANNER_INTENT_ARMS:
            expected_source = "PLANNER_RESOLUTION_FAILURE"
        if expected_source is not None and attempt["candidate_source"] != expected_source:
            out.append(self._v("CONTRACT_VIOLATION", "candidate_source", f"expected {expected_source}"))

        parse_status = attempt["parse_status"]
        validation_status = attempt["validation_status"]
        issue = attempt["issue_code"]
        if parse_status == "PARSED":
            if attempt["parsed_llm_response"] is None or attempt["parsed_typed_action"] is None or attempt["candidate_typed_action"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "parse_status", "PARSED requires parsed response, parsed TypedAction and candidate"))
            elif attempt["parsed_typed_action"] != attempt["candidate_typed_action"]:
                out.append(self._v("CONTRACT_VIOLATION", "candidate_typed_action", "LLM parsed action must equal candidate"))
        elif parse_status == "PARSE_FAILED":
            if attempt["parsed_llm_response"] is not None or attempt["parsed_typed_action"] is not None or attempt["candidate_typed_action"] is not None or validation_status != "NOT_APPLICABLE" or issue != "PARSE_FAILED":
                out.append(self._v("CONTRACT_VIOLATION", "parse_status", "invalid parse-failure flow"))
        elif parse_status == "NOT_APPLICABLE" and (
            attempt["parsed_llm_response"] is not None or attempt["parsed_typed_action"] is not None
        ):
            out.append(self._v("CONTRACT_VIOLATION", "parse_status", "N/A parser must have null parser-derived fields"))
        if validation_status == "VALID" and issue is not None:
            out.append(self._v("CONTRACT_VIOLATION", "issue_code", "VALID cannot carry issue"))
        if validation_status == "INVALID" and issue is None:
            out.append(self._v("CONTRACT_VIOLATION", "issue_code", "INVALID requires issue"))

        if arm in self.ORACLE_INTENT_ARMS:
            if attempt["semantic_resolution_status"] != "NOT_APPLICABLE":
                out.append(self._v("CONTRACT_VIOLATION", "semantic_resolution_status", "oracle bypasses resolver"))
            if self._nullable_fields_present(
                attempt,
                (
                    "semantic_ref",
                    "semantic_artifact_hash",
                    "semantic_resolution_method",
                    "semantic_bank_hash",
                    "semantic_top1_intent_id",
                    "semantic_top1_similarity",
                    "semantic_top2_intent_id",
                    "semantic_top2_similarity",
                    "semantic_margin",
                    "semantic_min_similarity",
                    "semantic_min_margin",
                ),
            ):
                out.append(self._v("CONTRACT_VIOLATION", "semantic_*", "oracle arm must not carry resolver fields"))
        if arm in self.PLANNER_INTENT_ARMS:
            required = (
                "planner_checkpoint_sha256",
                "planner_config_sha256",
                "planner_seed",
                "planner_support_status",
                "planner_support_signature_hash",
                "semantic_ref",
                "semantic_artifact_hash",
                "semantic_resolution_method",
                "semantic_bank_hash",
                "semantic_top1_intent_id",
                "semantic_top1_similarity",
                "semantic_top2_intent_id",
                "semantic_top2_similarity",
                "semantic_margin",
                "semantic_min_similarity",
                "semantic_min_margin",
            )
            if any(attempt.get(field) is None for field in required):
                out.append(self._v("CONTRACT_VIOLATION", "planner/semantic", "planner intent arm lacks identity or resolver fields"))
            if attempt.get("semantic_top1_intent_id") == attempt.get("semantic_top2_intent_id"):
                out.append(self._v("CONTRACT_VIOLATION", "semantic_top2_intent_id", "top-2 intent class must differ"))
        elif attempt["planner_support_status"] != "NOT_APPLICABLE" or attempt["planner_support_signature_hash"] is not None:
            if arm != "P_REPLAY_RAW":
                out.append(self._v("CONTRACT_VIOLATION", "planner_support_status", "non-planner arm must be NOT_APPLICABLE"))

        if arm in self.SHUFFLED_ARMS:
            fields = (
                "control_source_intent_id",
                "compatible_intent_ids",
                "control_mapping_hash",
                "intent_compatibility_hash",
                "control_certification_hash",
            )
            if any(attempt.get(field) is None for field in fields):
                out.append(self._v("CONTRACT_VIOLATION", "control_*", "shuffled arm requires frozen control identity"))
            elif attempt["control_source_intent_id"] in attempt["compatible_intent_ids"]:
                out.append(self._v("CONTRACT_VIOLATION", "control_source_intent_id", "wrong intent must be incompatible"))
        elif self._nullable_fields_present(
            attempt,
            (
                "control_source_intent_id",
                "compatible_intent_ids",
                "control_mapping_hash",
                "intent_compatibility_hash",
                "control_certification_hash",
            ),
        ):
            out.append(self._v("CONTRACT_VIOLATION", "control_*", "non-shuffled arm must not carry control fields"))

        if arm in self.EQUAL_TOKEN_ARMS and attempt["intent_text_hash"] is not None:
            out.append(self._v("CONTRACT_VIOLATION", "intent_text_hash", "neutral arm has no intent text"))
        if arm not in self.EQUAL_TOKEN_ARMS and arm != "P_REPLAY_RAW" and not unresolved and attempt["intent_text_hash"] is None:
            out.append(self._v("CONTRACT_VIOLATION", "intent_text_hash", "intent-guided LLM arm requires intent text hash"))

        llm_called = arm in self.LLM_ARMS and not unresolved and issue != "LLM_TIMEOUT"
        out.extend(self._validate_llm_presence(attempt, llm_called=llm_called))
        if llm_called:
            if attempt["added_block_tokens"] != 32:
                out.append(self._v("CONTRACT_VIOLATION", "added_block_tokens", "guidance block must be 32 attended tokens"))
            if attempt["tokens_in"] != attempt["attended_prompt_tokens"]:
                out.append(self._v("CONTRACT_VIOLATION", "tokens_in", "must equal attended unpadded input length"))
            if attempt["attended_prompt_tokens"] != attempt["prompt_tokens_total"]:
                out.append(self._v("CONTRACT_VIOLATION", "attended_prompt_tokens", "canonical prompt artifact must be unpadded"))
            if attempt["padded_sequence_length"] < attempt["prompt_tokens_total"]:
                out.append(self._v("CONTRACT_VIOLATION", "padded_sequence_length", "batch padding cannot shorten prompt"))
            if attempt.get("padding_side") not in {"NONE", "LEFT"}:
                out.append(self._v("CONTRACT_VIOLATION", "padding_side", "only unpadded or left-padded batching is allowed"))
            if attempt["raw_output"] is not None and attempt["raw_output_bytes_hash"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "raw_output_bytes_hash", "raw output text requires bytes hash"))
        if stage == "STAGE1A_INTERFACE" and llm_called:
            if attempt["prompt_tokens_total"] != attempt["matched_control_tokens"]:
                out.append(self._v("CONTRACT_VIOLATION", "prompt_tokens_total", "must equal frozen pair budget"))
        if stage == "STAGE1B_END_TO_END" and llm_called:
            if attempt["prompt_tokens_total"] > 512 or attempt["padded_sequence_length"] > 512:
                out.append(self._v("PROMPT_BUDGET_EXCEEDED", "prompt_tokens_total", "Stage1B unpadded/batch length must be <=512"))
            if attempt["matched_control_tokens"] != 32:
                out.append(self._v("CONTRACT_VIOLATION", "matched_control_tokens", "Stage1B equalizes guidance only at 32 tokens"))

        if arm == "P_REPLAY_RAW":
            if attempt["planner_checkpoint_sha256"] is None or attempt["planner_config_sha256"] is None or attempt["plan_step_id"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "P_REPLAY", "requires planner checkpoint/config and plan_step_id"))
            if attempt["raw_unmasked_action"] is None or attempt["raw_unmasked_args"] is None or attempt["candidate_typed_action"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "P_REPLAY", "requires raw unmasked planner action/args and candidate TypedAction"))
            if parse_status != "NOT_APPLICABLE":
                out.append(self._v("CONTRACT_VIOLATION", "parse_status", "P_REPLAY parser is N/A"))
            if attempt["planner_support_status"] == "NOT_APPLICABLE" or attempt["planner_support_signature_hash"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "planner_support_status", "P_REPLAY requires support diagnostics"))

        if unresolved and arm in self.PLANNER_INTENT_ARMS:
            if issue != "SEMANTIC_UNRESOLVED" or parse_status != "NOT_APPLICABLE" or validation_status != "NOT_APPLICABLE":
                out.append(self._v("CONTRACT_VIOLATION", "unresolved", "must terminate before LLM/parser/validator"))
            if attempt["raw_output"] is not None or attempt["parsed_llm_response"] is not None or attempt["parsed_typed_action"] is not None or attempt["candidate_typed_action"] is not None:
                out.append(self._v("CONTRACT_VIOLATION", "unresolved", "unresolved flow must have no LLM output"))

        if stage == "STAGE1A_INTERFACE":
            if attempt["oracle_distance_before"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "oracle_distance_before", "Stage1A requires distance"))
            if validation_status == "VALID":
                if attempt["oracle_distance_after"] is None or attempt["progress_success"] is None:
                    out.append(self._v("CONTRACT_VIOLATION", "progress_success", "valid action requires after distance"))
                elif attempt["progress_success"] != (
                    attempt["oracle_distance_after"] < attempt["oracle_distance_before"]
                ):
                    out.append(self._v("CONTRACT_VIOLATION", "progress_success", "flag does not match distance delta"))
        return out

    def validate_attempt_artifacts(self, attempt: dict[str, Any]) -> list[ContractViolation]:
        """Validates prompt/raw hashes when a byte loader is configured."""
        if self.artifact_bytes_loader is None:
            return []
        out: list[ContractViolation] = []
        if attempt.get("prompt_artifact") is not None:
            rendered = self.artifact_bytes_loader(attempt["prompt_artifact"])
            if prompt_bytes_hash(rendered) != attempt["prompt_hash"]:
                out.append(self._v("HASH_MISMATCH", "prompt_hash", "prompt bytes do not match"))
        if attempt.get("raw_output") is not None:
            raw = attempt["raw_output"].encode("utf-8")
            if prompt_bytes_hash(raw) != attempt["raw_output_bytes_hash"]:
                out.append(self._v("HASH_MISMATCH", "raw_output_bytes_hash", "raw UTF-8 bytes do not match"))
        return out

    def validate_pair_group(self, attempts: list[dict[str, Any]]) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        if not attempts:
            return [self._v("INCOMPLETE_PAIR", "attempts", "pair group is empty")]
        stages = {a["stage"] for a in attempts}
        if len(stages) != 1:
            return [self._v("INCOMPLETE_PAIR", "stage", "mixed stages in pair group")]
        stage = next(iter(stages))
        if stage == "STAGE1A_INTERFACE":
            expected_arms = self.INTERFACE_ARMS
            actual_arms = {a["arm"] for a in attempts}
            if actual_arms != expected_arms or len(attempts) != len(expected_arms):
                out.append(self._v("INCOMPLETE_PAIR", "arm", f"expected exactly {sorted(expected_arms)}"))
            identity = {
                (a["task_id"], a["snapshot_id"], a["state_before_hash"], a["goal_hash"], a["pair_group_hash"], a["base_prompt_hash"])
                for a in attempts
            }
            if len(identity) != 1:
                out.append(self._v("CONTRACT_VIOLATION", "pair_group", "Stage1A identity/base prompt differs across arms"))
            budgets = {(a["prompt_tokens_total"], a["attended_prompt_tokens"], a["added_block_tokens"]) for a in attempts}
            if len(budgets) != 1:
                out.append(self._v("CONTRACT_VIOLATION", "pair_group", "Stage1A token budgets differ across arms"))
        else:
            if len({a["task_id"] for a in attempts}) != 1 or len({a["pair_group_hash"] for a in attempts}) != 1:
                out.append(self._v("INCOMPLETE_PAIR", "pair_group", "Stage1B group must share task and pair hash"))
        return out

    def validate_episode(self, episode: dict[str, Any]) -> list[ContractViolation]:
        out: list[ContractViolation] = []
        if episode["stage"] == "STAGE1A_INTERFACE":
            if episode["progress_success"] is None or episode["goal_success"] is not None or episode["snapshot_id"] is None:
                out.append(self._v("CONTRACT_VIOLATION", "episode", "invalid Stage1A summary fields"))
        elif episode["stage"] == "STAGE1B_END_TO_END":
            if episode["goal_success"] is None or episode["progress_success"] is not None or episode["snapshot_id"] is not None:
                out.append(self._v("CONTRACT_VIOLATION", "episode", "invalid Stage1B summary fields"))
        expected_pair_hash = pair_group_hash(
            stage=episode["stage"],
            task_id=episode["task_id"],
            base_task_id=episode["base_task_id"],
            split=episode["split"],
            snapshot_id=episode["snapshot_id"],
            trajectory_policy=episode["trajectory_policy"],
            experiment_freeze_hash=episode["experiment_freeze_hash"],
        )
        if episode["pair_group_hash"] != expected_pair_hash:
            out.append(self._v("HASH_MISMATCH", "pair_group_hash", "episode pair hash mismatch"))
        if episode["rollout_mode"] == "RAW" and episode["retries_total"] != 0:
            out.append(self._v("CONTRACT_VIOLATION", "retries_total", "RAW must have zero retries"))
        if episode["goal_success"] is True and episode["terminal_error"] is not None:
            out.append(self._v("CONTRACT_VIOLATION", "terminal_error", "successful episode cannot have terminal error"))
        if episode["goal_success"] is False and episode["terminal_error"] is None:
            out.append(self._v("CONTRACT_VIOLATION", "terminal_error", "failed Stage1B episode requires terminal error"))
        return out

    def validate_episode_attempts(self, episode: dict[str, Any], attempts: list[dict[str, Any]]) -> list[ContractViolation]:
        out = self.validate_episode(episode)
        for index, attempt in enumerate(attempts):
            for field in ("run_id", "episode_id", "trajectory_id", "stage", "task_id", "base_task_id", "snapshot_id", "canonical_task_hash", "split", "arm", "rollout_mode", "trajectory_policy", "experiment_freeze_hash", "pair_group_hash"):
                if attempt[field] != episode[field]:
                    out.append(self._v("CONTRACT_VIOLATION", f"attempts[{index}].{field}", "differs from episode"))
        ordered = sorted(attempts, key=lambda a: (a["step_index"], a["attempt_index"]))
        if attempts != ordered:
            out.append(self._v("CONTRACT_VIOLATION", "attempts", "must be ordered by step_index, attempt_index"))
        for previous, current in zip(attempts, attempts[1:]):
            if previous["state_after_hash"] is not None and current["state_before_hash"] != previous["state_after_hash"]:
                out.append(self._v("CONTRACT_VIOLATION", "attempts.state", "trajectory state hashes are not continuous"))
        if len(attempts) != episode["attempts_total"]:
            out.append(self._v("CONTRACT_VIOLATION", "attempts_total", "does not equal attempt rows"))
        executed = [
            a
            for a in attempts
            if a["validation_status"] == "VALID"
            and a["candidate_typed_action"] is not None
            and a["candidate_typed_action"]["action"] != "END"
        ]
        if len(executed) != episode["steps_accepted"] or len(executed) != episode["executed_length"]:
            out.append(self._v("CONTRACT_VIOLATION", "steps_accepted|executed_length", "does not equal executed valid non-END actions"))
        retries = sum(a["attempt_index"] > 0 for a in attempts)
        if retries != episode["retries_total"]:
            out.append(self._v("CONTRACT_VIOLATION", "retries_total", "does not equal retry rows"))
        planner_calls = sum(a["arm"] in self.PLANNER_INTENT_ARMS or a["arm"] == "P_REPLAY_RAW" for a in attempts)
        if planner_calls != episode["planner_calls"]:
            out.append(self._v("CONTRACT_VIOLATION", "planner_calls", "does not equal planner attempts"))
        resolver_calls = sum(a["semantic_resolution_status"] in {"RESOLVED", "UNRESOLVED"} for a in attempts)
        if resolver_calls != episode["semantic_resolver_calls"]:
            out.append(self._v("CONTRACT_VIOLATION", "semantic_resolver_calls", "does not equal resolver attempts"))
        unresolved = sum(a["semantic_resolution_status"] == "UNRESOLVED" for a in attempts)
        if unresolved != episode["semantic_unresolved_count"]:
            out.append(self._v("CONTRACT_VIOLATION", "semantic_unresolved_count", "does not equal attempts"))
        unseen = sum(a["planner_support_status"] == "UNSEEN_SIGNATURE" for a in attempts)
        if unseen != episode["unseen_support_signature_count"]:
            out.append(self._v("CONTRACT_VIOLATION", "unseen_support_signature_count", "does not equal attempts"))
        token_in = sum(a["tokens_in"] or 0 for a in attempts)
        attended = sum(a["attended_prompt_tokens"] or 0 for a in attempts)
        token_out = sum(a["tokens_out"] or 0 for a in attempts)
        latency = sum(a["total_ms"] or 0 for a in attempts)
        if episode["total_tokens_in"] != token_in:
            out.append(self._v("CONTRACT_VIOLATION", "total_tokens_in", "does not equal attempt sum"))
        if episode["total_attended_tokens"] != attended:
            out.append(self._v("CONTRACT_VIOLATION", "total_attended_tokens", "does not equal attempt sum"))
        if episode["total_tokens_out"] != token_out:
            out.append(self._v("CONTRACT_VIOLATION", "total_tokens_out", "does not equal attempt sum"))
        if abs((episode["total_latency_ms"] or 0) - latency) > 1e-9:
            out.append(self._v("CONTRACT_VIOLATION", "total_latency_ms", "does not equal attempt sum"))
        budget_violations = sum(
            a["stage"] == "STAGE1B_END_TO_END"
            and a["arm"] != "P_REPLAY_RAW"
            and a["semantic_resolution_status"] != "UNRESOLVED"
            and (a["prompt_tokens_total"] > 512 or a["padded_sequence_length"] > 512 or a.get("padding_side") not in {"NONE", "LEFT"})
            for a in attempts
        )
        if budget_violations != episode["prompt_budget_violation_count"]:
            out.append(self._v("CONTRACT_VIOLATION", "prompt_budget_violation_count", "does not equal attempts"))
        final_hash = attempts[-1]["state_after_hash"] or attempts[-1]["state_before_hash"] if attempts else episode["final_state_hash"]
        if attempts and episode["final_state_hash"] != final_hash:
            out.append(self._v("HASH_MISMATCH", "final_state_hash", "does not match final attempt state"))
        if episode["terminal_error"] is not None and attempts:
            last_issue = attempts[-1]["issue_code"]
            allowed_loop_errors = {"HORIZON_EXCEEDED", "GOAL_NOT_REACHED", "PLANNER_OOD_STATE"}
            if episode["terminal_error"] != last_issue and episode["terminal_error"] not in allowed_loop_errors:
                out.append(self._v("CONTRACT_VIOLATION", "terminal_error", "does not match terminal attempt"))
        return out

    @staticmethod
    def assert_valid(violations: list[ContractViolation]) -> None:
        if violations:
            raise ValueError("; ".join(f"{v.code}@{v.path}: {v.message}" for v in violations))

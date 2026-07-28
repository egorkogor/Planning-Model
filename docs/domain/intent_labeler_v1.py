"""Normative deterministic intent labeler for Work Planner / BlocksWorld v1.20.

The module is deliberately dependency-free. Facts are tuples such as
("ON", "@B0", "@B1") and actions are dictionaries with keys ``action`` and
``args`` where args are refs in normative order.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

INTENT_KEYS = {
    0: "CLEAR_MOVING_BLOCK",
    1: "CLEAR_TARGET_SUPPORT",
    2: "MAKE_MOVING_BLOCK_AVAILABLE",
    3: "TEMPORARY_TABLE_PLACEMENT",
    4: "PLACE_GOAL_BLOCK",
    5: "RESTORE_HAND_EMPTY",
    6: "GOAL_ALREADY_SATISFIED",
}

INTENT_TEXT = {
    0: "clear the block that must move next",
    1: "clear the support required by the goal",
    2: "make the relevant block available for placement",
    3: "place an obstructing block aside temporarily",
    4: "place the relevant block on its required support",
    5: "restore an empty hand before the next placement",
    6: "finish because the requested configuration is satisfied",
}


@dataclass(frozen=True)
class SemanticSignature:
    intent_id: int
    hand_mode: str
    goal_relation: str
    moving_clear: str
    support_clear: str
    obstruction_depth_bucket: str
    remaining_distance_bucket: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fact_set(facts: Iterable[Sequence[str]]) -> set[tuple[str, ...]]:
    return {tuple(str(x) for x in fact) for fact in facts}


def _goal_satisfied(state: set[tuple[str, ...]], goal: set[tuple[str, ...]]) -> bool:
    return goal.issubset(state)


def _action_tuple(action: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    name = str(action["action"])
    args = tuple(str(x) for x in action.get("args", []))
    return name, args


def _apply(state: set[tuple[str, ...]], action: Mapping[str, object]) -> set[tuple[str, ...]]:
    name, args = _action_tuple(action)
    out = set(state)
    if name == "END":
        return out
    if name == "PICK_UP":
        (b,) = args
        out -= {("ON_TABLE", b), ("CLEAR", b), ("HAND_EMPTY",)}
        out.add(("HOLDING", b))
    elif name == "UNSTACK":
        moving, support = args
        out -= {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)}
        out |= {("HOLDING", moving), ("CLEAR", support)}
    elif name == "PUT_DOWN":
        (b,) = args
        out.discard(("HOLDING", b))
        out |= {("ON_TABLE", b), ("CLEAR", b), ("HAND_EMPTY",)}
    elif name == "STACK":
        moving, support = args
        out -= {("HOLDING", moving), ("CLEAR", support)}
        out |= {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)}
    else:
        raise ValueError(f"unknown action: {name}")
    return out


def _blocks_above(state: set[tuple[str, ...]], block: str) -> int:
    parent: dict[str, str] = {}
    for fact in state:
        if fact[0] == "ON":
            parent[fact[2]] = fact[1]
    count = 0
    current = block
    seen: set[str] = set()
    while current in parent:
        current = parent[current]
        if current in seen:
            raise ValueError("cyclic state")
        seen.add(current)
        count += 1
    return count


def _bucket_depth(depth: int | None) -> str:
    if depth is None:
        return "NOT_APPLICABLE"
    if depth == 0:
        return "ZERO"
    if depth == 1:
        return "ONE"
    return "TWO_PLUS"


def _bucket_distance(distance: int) -> str:
    if distance == 0:
        return "ZERO"
    if distance <= 2:
        return "ONE_TWO"
    if distance <= 5:
        return "THREE_FIVE"
    if distance <= 10:
        return "SIX_TEN"
    if distance <= 16:
        return "ELEVEN_SIXTEEN"
    raise ValueError("remaining_oracle_length must be 0..16")


def _relevant_goal(goal: set[tuple[str, ...]], selected_action: Mapping[str, object]) -> tuple[str, ...] | None:
    name, args = _action_tuple(selected_action)
    moving = args[0] if args else None
    candidates = sorted(g for g in goal if moving is not None and len(g) >= 2 and g[1] == moving)
    if candidates:
        return candidates[0]
    return sorted(goal)[0] if goal else None


def label_intent(
    state_facts: Iterable[Sequence[str]],
    goal_facts: Iterable[Sequence[str]],
    all_shortest_first_actions: Sequence[Mapping[str, object]],
    selected_action: Mapping[str, object],
    remaining_oracle_length: int,
) -> dict[str, object]:
    """Return exact intent/signature/canonical text for the selected oracle action."""
    state = _fact_set(state_facts)
    goal = _fact_set(goal_facts)
    name, args = _action_tuple(selected_action)
    if not all_shortest_first_actions:
        raise ValueError("all_shortest_first_actions must be non-empty")
    if _action_tuple(all_shortest_first_actions[0]) != (name, args):
        raise ValueError("selected_action must be first under normative tie-break")

    if _goal_satisfied(state, goal):
        if name != "END":
            raise ValueError("goal-satisfied state requires END")
        intent_id = 6
    else:
        after = _apply(state, selected_action)
        created_goal = bool((after - state) & goal)
        moving = args[0] if args else None
        support = args[1] if len(args) > 1 else None
        unsatisfied = goal - state
        goal_subjects = {g[1] for g in unsatisfied if len(g) >= 2}
        goal_supports = {g[2] for g in unsatisfied if g[0] == "ON"}

        if created_goal:
            intent_id = 4
        elif name == "UNSTACK" and support in goal_supports:
            intent_id = 1
        elif name in {"UNSTACK", "PICK_UP"} and moving in goal_subjects and _blocks_above(state, moving) == 0:
            intent_id = 2
        elif name == "UNSTACK" and moving not in goal_subjects:
            intent_id = 0
        elif name == "PUT_DOWN" and ("ON_TABLE", moving) not in goal:
            intent_id = 3
        else:
            intent_id = 5

    relevant = _relevant_goal(goal, selected_action)
    if intent_id == 6:
        goal_relation = "SATISFIED"
        moving_clear = support_clear = "NOT_APPLICABLE"
        obstruction_depth = None
    else:
        goal_relation = relevant[0] if relevant and relevant[0] in {"ON", "ON_TABLE"} else "ON_TABLE"
        moving = args[0] if args else None
        support = relevant[2] if relevant and relevant[0] == "ON" else None
        moving_clear = "YES" if moving and ("CLEAR", moving) in state else ("NO" if moving else "NOT_APPLICABLE")
        support_clear = "YES" if support and ("CLEAR", support) in state else ("NO" if support else "NOT_APPLICABLE")
        relevant_blocks = [b for b in (moving, support) if b]
        obstruction_depth = max((_blocks_above(state, b) for b in relevant_blocks), default=0)

    signature = SemanticSignature(
        intent_id=intent_id,
        hand_mode="HAND_EMPTY" if ("HAND_EMPTY",) in state else "HOLDING",
        goal_relation=goal_relation,
        moving_clear=moving_clear,
        support_clear=support_clear,
        obstruction_depth_bucket=_bucket_depth(obstruction_depth),
        remaining_distance_bucket=_bucket_distance(remaining_oracle_length),
    )
    key = INTENT_KEYS[intent_id]
    canonical_text = (
        f"intent={key}; hand={signature.hand_mode}; goal_relation={signature.goal_relation}; "
        f"moving_clear={signature.moving_clear}; support_clear={signature.support_clear}; "
        f"obstruction_depth={signature.obstruction_depth_bucket}; "
        f"remaining_distance={signature.remaining_distance_bucket}"
    )
    return {
        "intent_id": intent_id,
        "intent_key": key,
        "intent_text": INTENT_TEXT[intent_id],
        "semantic_signature": signature.to_dict(),
        "canonical_text": canonical_text,
    }

"""BlocksWorld semantics implementing docs/domain/blocks_world_v1.yaml."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from .canonical import canonical_task_hash

Fact = tuple[str, ...]
Action = tuple[str, ...]
ACTION_RANK = {"PICK_UP": 0, "UNSTACK": 1, "PUT_DOWN": 2, "STACK": 3, "END": 4}


def canonical_facts(facts: Iterable[Fact]) -> tuple[Fact, ...]:
    return tuple(sorted((tuple(fact) for fact in facts), key=lambda fact: (fact[0], fact[1:])))


@dataclass(frozen=True)
class Task:
    blocks: tuple[str, ...]
    initial: tuple[Fact, ...]
    goal: tuple[Fact, ...]
    task_id: str = ""

    def payload(self) -> dict:
        return {
            "domain_id": "blocks_world_v1",
            "blocks": list(self.blocks),
            "initial": [list(fact) for fact in canonical_facts(self.initial)],
            "goal": [list(fact) for fact in canonical_facts(self.goal)],
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_task_hash(self.payload())


def validate_state(blocks: tuple[str, ...], state: Iterable[Fact]) -> tuple[Fact, ...]:
    facts = canonical_facts(state)
    fact_set = set(facts)
    held = [fact[1] for fact in facts if fact[0] == "HOLDING"]
    if len(held) > 1 or (("HAND_EMPTY",) in fact_set) != (not held):
        raise ValueError("invalid hand mode")
    for block in blocks:
        locations = sum(
            [
                ("ON_TABLE", block) in fact_set,
                any(f[:2] == ("ON", block) for f in facts),
                ("HOLDING", block) in fact_set,
            ]
        )
        if locations != 1:
            raise ValueError(f"block {block} must have exactly one location")
        expected_clear = (
            not any(f[0] == "ON" and f[2] == block for f in facts) and block not in held
        )
        if (("CLEAR", block) in fact_set) != expected_clear:
            raise ValueError(f"incorrect CLEAR fact for {block}")
    refs = {arg for fact in facts for arg in fact[1:]}
    if not refs <= set(blocks):
        raise ValueError("unknown block reference")
    edges = {(f[1], f[2]) for f in facts if f[0] == "ON"}
    if len({support for _, support in edges}) != len(edges):
        raise ValueError("more than one block on a support")
    for start, _ in edges:
        seen = {start}
        node = start
        while True:
            next_nodes = [support for moving, support in edges if moving == node]
            if not next_nodes:
                break
            node = next_nodes[0]
            if node in seen:
                raise ValueError("cyclic ON graph")
            seen.add(node)
    return facts


def applicable(blocks: tuple[str, ...], state: tuple[Fact, ...], action: Action) -> bool:
    facts = set(state)
    name, *args = action
    if name == "PICK_UP" and len(args) == 1:
        block = args[0]
        return block in blocks and {("ON_TABLE", block), ("CLEAR", block), ("HAND_EMPTY",)} <= facts
    if name == "UNSTACK" and len(args) == 2:
        moving, support = args
        return (
            moving != support
            and {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)} <= facts
        )
    if name == "PUT_DOWN" and len(args) == 1:
        return ("HOLDING", args[0]) in facts
    if name == "STACK" and len(args) == 2:
        moving, support = args
        return moving != support and {("HOLDING", moving), ("CLEAR", support)} <= facts
    return False


def apply_action(
    blocks: tuple[str, ...], state: tuple[Fact, ...], action: Action
) -> tuple[Fact, ...]:
    if not applicable(blocks, state, action):
        raise ValueError(f"inapplicable action: {action}")
    facts = set(state)
    name, *args = action
    if name == "PICK_UP":
        block = args[0]
        facts -= {("ON_TABLE", block), ("CLEAR", block), ("HAND_EMPTY",)}
        facts.add(("HOLDING", block))
    elif name == "UNSTACK":
        moving, support = args
        facts -= {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)}
        facts |= {("HOLDING", moving), ("CLEAR", support)}
    elif name == "PUT_DOWN":
        block = args[0]
        facts.remove(("HOLDING", block))
        facts |= {("ON_TABLE", block), ("CLEAR", block), ("HAND_EMPTY",)}
    else:
        moving, support = args
        facts -= {("HOLDING", moving), ("CLEAR", support)}
        facts |= {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)}
    return validate_state(blocks, facts)


def goal_satisfied(state: tuple[Fact, ...], goal: tuple[Fact, ...]) -> bool:
    return set(goal) <= set(state)


def actions(blocks: tuple[str, ...], state: tuple[Fact, ...]) -> list[Action]:
    candidates: list[Action] = []
    for block in blocks:
        candidates.extend([("PICK_UP", block), ("PUT_DOWN", block)])
    for moving in blocks:
        for support in blocks:
            if moving != support:
                candidates.extend([("UNSTACK", moving, support), ("STACK", moving, support)])
    return sorted(
        (action for action in candidates if applicable(blocks, state, action)),
        key=lambda action: (ACTION_RANK[action[0]],)
        + tuple(blocks.index(arg) for arg in action[1:]),
    )


def shortest_plan(task: Task) -> tuple[Action, ...]:
    initial = validate_state(task.blocks, task.initial)
    if goal_satisfied(initial, task.goal):
        return ()
    queue = deque([(initial, ())])
    seen = {initial}
    while queue:
        state, plan = queue.popleft()
        for action in actions(task.blocks, state):
            successor = apply_action(task.blocks, state, action)
            if successor in seen:
                continue
            candidate = plan + (action,)
            if goal_satisfied(successor, task.goal):
                return candidate
            seen.add(successor)
            queue.append((successor, candidate))
    raise ValueError("unreachable goal")

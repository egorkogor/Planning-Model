from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

Fact = tuple[str, ...]
State = frozenset[Fact]
Action = tuple[str, tuple[str, ...]]
ACTION_RANK = {"PICK_UP": 0, "UNSTACK": 1, "PUT_DOWN": 2, "STACK": 3}


def ref_rank(ref: str) -> int:
    return int(ref[2:])


def canonical_state(facts: Iterable[Iterable[str]]) -> State:
    return frozenset(tuple(x) for x in facts)


def canonical_state_list(state: State) -> list[list[str]]:
    return [list(x) for x in sorted(state)]


def goal_pass(state: State, goal: Iterable[Iterable[str]]) -> bool:
    return all(tuple(f) in state for f in goal)


def _objects(state: State) -> list[str]:
    return sorted({x for f in state for x in f[1:] if x.startswith("@B")}, key=ref_rank)


def action_sort_key(action: Action) -> tuple[int, ...]:
    name, args = action
    padded = tuple(ref_rank(x) for x in args) + (-1,) * (2 - len(args))
    return (ACTION_RANK[name], *padded)


def applicable_actions(state: State) -> list[Action]:
    objs = _objects(state)
    out: list[Action] = []
    hand_empty = ("HAND_EMPTY",) in state
    holding = {f[1] for f in state if f[0] == "HOLDING"}
    clear = {f[1] for f in state if f[0] == "CLEAR"}
    on_table = {f[1] for f in state if f[0] == "ON_TABLE"}
    on = {(f[1], f[2]) for f in state if f[0] == "ON"}
    if hand_empty:
        for b in objs:
            if b in on_table and b in clear:
                out.append(("PICK_UP", (b,)))
        for moving, support in on:
            if moving in clear:
                out.append(("UNSTACK", (moving, support)))
    else:
        for b in holding:
            out.append(("PUT_DOWN", (b,)))
            for support in objs:
                if support != b and support in clear:
                    out.append(("STACK", (b, support)))
    return sorted(out, key=action_sort_key)


def apply_action(state: State, action: Action) -> State:
    name, args = action
    if action not in applicable_actions(state):
        raise ValueError(f"action is not applicable: {action}")
    nxt = set(state)
    if name == "PICK_UP":
        (block,) = args
        nxt -= {("ON_TABLE", block), ("CLEAR", block), ("HAND_EMPTY",)}
        nxt.add(("HOLDING", block))
    elif name == "PUT_DOWN":
        (block,) = args
        nxt.remove(("HOLDING", block))
        nxt |= {("ON_TABLE", block), ("CLEAR", block), ("HAND_EMPTY",)}
    elif name == "UNSTACK":
        moving, support = args
        nxt -= {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)}
        nxt |= {("HOLDING", moving), ("CLEAR", support)}
    elif name == "STACK":
        moving, support = args
        nxt -= {("HOLDING", moving), ("CLEAR", support)}
        nxt |= {("ON", moving, support), ("CLEAR", moving), ("HAND_EMPTY",)}
    else:
        raise ValueError(name)
    return frozenset(nxt)


def shortest_plan(initial: Iterable[Iterable[str]], goal: Iterable[Iterable[str]], max_depth: int = 16) -> list[Action] | None:
    start = canonical_state(initial)
    if goal_pass(start, goal):
        return []
    queue = deque([start])
    path: dict[State, tuple[State | None, Action | None]] = {start: (None, None)}
    depth: dict[State, int] = {start: 0}
    while queue:
        state = queue.popleft()
        d = depth[state]
        if d >= max_depth:
            continue
        for action in applicable_actions(state):
            nxt = apply_action(state, action)
            if nxt in path:
                continue
            path[nxt] = (state, action)
            depth[nxt] = d + 1
            if goal_pass(nxt, goal):
                result: list[Action] = []
                cur = nxt
                while path[cur][0] is not None:
                    prev, act = path[cur]
                    if prev is None or act is None:
                        raise RuntimeError("oracle predecessor chain is incomplete")
                    result.append(act)
                    cur = prev
                result.reverse()
                return result
            queue.append(nxt)
    return None


def oracle_distance(initial: Iterable[Iterable[str]], goal: Iterable[Iterable[str]], max_depth: int = 16) -> int | None:
    plan = shortest_plan(initial, goal, max_depth)
    return None if plan is None else len(plan)


def all_shortest_first_actions(initial: Iterable[Iterable[str]], goal: Iterable[Iterable[str]], max_depth: int = 16) -> list[Action]:
    start = canonical_state(initial)
    best = oracle_distance(initial, goal, max_depth)
    if best is None or best == 0:
        return []
    out: list[Action] = []
    for action in applicable_actions(start):
        nxt = apply_action(start, action)
        remainder = oracle_distance(canonical_state_list(nxt), goal, max_depth - 1)
        if remainder is not None and 1 + remainder == best:
            out.append(action)
    return sorted(out, key=action_sort_key)


def reachable_states(initial: Iterable[Iterable[str]], max_steps: int = 16) -> set[State]:
    start = canonical_state(initial)
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        if depth >= max_steps:
            continue
        for action in applicable_actions(state):
            nxt = apply_action(state, action)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    return seen

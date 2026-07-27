from docs.domain.intent_labeler_v1 import label_intent


def test_goal_satisfied_end():
    out = label_intent(
        [("ON_TABLE", "@B0"), ("CLEAR", "@B0"), ("HAND_EMPTY",)],
        [("ON_TABLE", "@B0")],
        [{"action": "END", "args": []}],
        {"action": "END", "args": []},
        0,
    )
    assert out["intent_id"] == 6
    assert "@B0" not in out["canonical_text"]


def test_stack_creates_goal_relation():
    out = label_intent(
        [("HOLDING", "@B0"), ("ON_TABLE", "@B1"), ("CLEAR", "@B1")],
        [("ON", "@B0", "@B1")],
        [{"action": "STACK", "args": ["@B0", "@B1"]}],
        {"action": "STACK", "args": ["@B0", "@B1"]},
        1,
    )
    assert out["intent_id"] == 4
    assert out["semantic_signature"]["goal_relation"] == "ON"


def test_temporary_put_down():
    out = label_intent(
        [("HOLDING", "@B2"), ("ON_TABLE", "@B0"), ("CLEAR", "@B0")],
        [("ON", "@B0", "@B2")],
        [{"action": "PUT_DOWN", "args": ["@B2"]}],
        {"action": "PUT_DOWN", "args": ["@B2"]},
        3,
    )
    assert out["intent_id"] == 3


def test_deterministic():
    args = (
        [("ON", "@B2", "@B1"), ("ON_TABLE", "@B1"), ("CLEAR", "@B2"), ("HAND_EMPTY",)],
        [("ON", "@B0", "@B1")],
        [{"action": "UNSTACK", "args": ["@B2", "@B1"]}],
        {"action": "UNSTACK", "args": ["@B2", "@B1"]},
        4,
    )
    assert label_intent(*args) == label_intent(*args)

from validation.domain_oracle import canonical_state, applicable_actions, apply_action, oracle_distance

def table_state():
    return [['ON_TABLE','@B0'],['ON_TABLE','@B1'],['CLEAR','@B0'],['CLEAR','@B1'],['HAND_EMPTY']]

def test_oracle_distance_stack_two_blocks():
    assert oracle_distance(table_state(),[['ON','@B0','@B1']],16)==2

def test_pickup_creates_holding_state():
    state=canonical_state(table_state())
    nxt=apply_action(state,('PICK_UP',('@B0',)))
    assert ('HOLDING','@B0') in nxt
    assert ('HAND_EMPTY',) not in nxt
    assert ('STACK',('@B0','@B1')) in applicable_actions(nxt)

def test_apply_then_stack_reaches_goal():
    state=canonical_state(table_state())
    state=apply_action(state,('PICK_UP',('@B0',)))
    state=apply_action(state,('STACK',('@B0','@B1')))
    assert ('ON','@B0','@B1') in state

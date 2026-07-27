from validation.validate_task_encoding import SPEC, encode

def test_at_least_three_examples():
    assert len(SPEC['golden_examples']) >= 3

def test_golden_vectors_byte_exact():
    for ex in SPEC['golden_examples']:
        actual=encode(ex)
        assert actual['token_ids']==ex['expected_token_ids']
        assert actual['segment_ids']==ex['expected_segment_ids']
        assert actual['argument_position_ids']==ex['expected_argument_position_ids']
        assert actual['attention_mask']==ex['expected_attention_mask']
        assert actual['ref_slot_positions']==ex['expected_ref_slot_positions']

def test_holding_state_is_covered():
    assert any(any(f[0]=='HOLDING' for f in ex['initial']) for ex in SPEC['golden_examples'])

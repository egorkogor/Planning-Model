from __future__ import annotations
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
SPEC=yaml.safe_load((ROOT/'docs'/'architecture'/'task_encoding_v1.yaml').read_text(encoding='utf-8'))
T=SPEC['tokens']; SEG=SPEC['segments']; MAX=SPEC['max_length']

def ref_token(ref:str)->int: return T['REF_SLOT_0']+int(ref[2:])

def encode(ex:dict):
    ids=[]; seg=[]; arg=[]; ref_positions={}
    def add(token:int, segment:int, argpos:int=0): ids.append(token); seg.append(segment); arg.append(argpos)
    add(T['BOS'],SEG['SPECIAL']); add(T['DOMAIN_BLOCKS'],SEG['SPECIAL']); add(T['LEDGER'],SEG['LEDGER'])
    for ref in ex['ledger_refs']:
        ref_positions[ref]=len(ids); add(ref_token(ref),SEG['LEDGER']); add(T['TYPE_BLOCK'],SEG['LEDGER'])
    add(T['STATE'],SEG['STATE'])
    for fact in sorted(ex['initial'],key=lambda x:tuple(x)):
        add(T['PRED_OPEN'],SEG['STATE']); add(T[fact[0]],SEG['STATE'],1)
        for i,r in enumerate(fact[1:]): add(ref_token(r),SEG['STATE'],2+i)
        add(T['PRED_CLOSE'],SEG['STATE'])
    add(T['GOAL'],SEG['GOAL'])
    for fact in sorted(ex['goal'],key=lambda x:tuple(x)):
        add(T['PRED_OPEN'],SEG['GOAL']); add(T[fact[0]],SEG['GOAL'],1)
        for i,r in enumerate(fact[1:]): add(ref_token(r),SEG['GOAL'],2+i)
        add(T['PRED_CLOSE'],SEG['GOAL'])
    add(T['EOS'],SEG['SPECIAL'])
    mask=[1]*len(ids); pad=MAX-len(ids)
    if pad<0: raise ValueError('encoding exceeds max_length')
    ids += [T['PAD']]*pad; seg += [SEG['SPECIAL']]*pad; arg += [0]*pad; mask += [0]*pad
    return {'token_ids':ids,'segment_ids':seg,'argument_position_ids':arg,'attention_mask':mask,'ref_slot_positions':ref_positions}

def validate_all()->None:
    assert len(SPEC['golden_examples'])>=3
    for ex in SPEC['golden_examples']:
        actual=encode(ex)
        assert actual['token_ids']==ex['expected_token_ids'],ex['id']+' token_ids'
        assert actual['segment_ids']==ex['expected_segment_ids'],ex['id']+' segment_ids'
        assert actual['argument_position_ids']==ex['expected_argument_position_ids'],ex['id']+' argument positions'
        assert actual['attention_mask']==ex['expected_attention_mask'],ex['id']+' attention mask'
        assert actual['ref_slot_positions']==ex['expected_ref_slot_positions'],ex['id']+' ref positions'

if __name__=='__main__':
    validate_all(); print(f"PASS: {len(SPEC['golden_examples'])} complete task-encoding golden examples")

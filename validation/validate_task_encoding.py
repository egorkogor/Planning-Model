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

def validate_all() -> None:
    examples = SPEC.get("golden_examples", [])
    if len(examples) < 3:
        raise RuntimeError("at least three task-encoding golden examples are required")
    fields = (
        ("token_ids", "token_ids"),
        ("segment_ids", "segment_ids"),
        ("argument_position_ids", "argument positions"),
        ("attention_mask", "attention mask"),
        ("ref_slot_positions", "ref positions"),
    )
    for ex in examples:
        actual = encode(ex)
        for field, label in fields:
            expected_field = f"expected_{field}"
            if actual[field] != ex[expected_field]:
                raise RuntimeError(f"{ex['id']} {label} mismatch")

if __name__=='__main__':
    validate_all(); print(f"PASS: {len(SPEC['golden_examples'])} complete task-encoding golden examples")

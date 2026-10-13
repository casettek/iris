#!/usr/bin/env python3
"""Check generated CFS dataflow against the ranked-BPE authoring contract.

Run `cargo raster cfs` in both tokenizer projects first. This supplements,
and does not replace, source review, guest builds, and execution audits.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def scope(n): return {'SequenceScope': {'input_index': n}}
def prior(n): return {'PriorItemOutput': {'intra_sequence_item_index': n}}
def load(name):
    cfs = json.loads((root / 'raster-stages' / name / 'target/raster/cfs.json').read_text())
    return cfs, {s['id']: s for s in cfs['sequences']}

merge, m = load('prompt-merge')
final, f = load('prompt-prepare')
assert len(merge['tiles']) == 7 and len(final['tiles']) == 12
assert m['main']['entry_arguments'] == ['tokenizer', 'initial_pieces']
assert f['main']['entry_arguments'] == ['tokenizer', 'merged_pieces']
assert len(m['main']['items']) == 8
for i, item in enumerate(m['main']['items']):
    assert item == {'Sequence': {'id': 'merge_once', 'sources': ['EntryArgument' if i == 0 else prior(i-1), 'EntryArgument', 'EntryArgument']}}
assert m['merge_once']['items'] == [
    {'Sequence': {'id': 'find_best', 'sources': [scope(0), scope(1), scope(2)]}},
    {'RecurTile': {'id': 'apply_ranked_merge', 'sources': [scope(0), {'Direct': 'Inline'}, prior(0)], 'chunk': 256}},
]
for sequences in [m, f]:
    assert sequences['find_best']['items'] == [
        {'Tile': {'id': 'empty_merge_scan', 'sources': []}},
        {'RecurSequence': {'id': 'scan_pair', 'sources': [scope(0), prior(0), scope(1), scope(2)]}},
    ]
    lookup = sequences['scan_pair']['items'][3]['RecurTile']
    assert lookup['id'] == 'scan_merge_rules'
    assert lookup['sources'][0] == {'Indexed': {'value': scope(2), 'indexes': [prior(1)]}}
assert f['main']['items'][1] == {'Tile': {'id': 'assert_merges_converged', 'sources': [prior(0)]}}
# Inline slots are exclusively sanctioned new drafts. Computed inline tile
# arguments would have no authenticated lineage and must fail this check.
for name, sequences in [('prompt-merge', m), ('prompt-prepare', f)]:
    for sequence in sequences.values():
        for index, item in enumerate(sequence['items']):
            op = next(iter(item.values()))
            for slot, source in enumerate(op['sources']):
                if 'Inline' in json.dumps(source):
                    assert (name, sequence['id'], index, slot) in [
                        ('prompt-merge', 'merge_once', 1, 1), ('prompt-prepare', 'main', 2, 1)]
print('PASS: eight explicit merges, immutable search/apply inputs, bounded 256-piece draft tiles, dictionary record scans, convergence assertion, and authorized bindings')

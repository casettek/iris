#!/usr/bin/env python3
"""Generate adversarial BPE fixtures with the independent tokenizers 0.22.2 oracle."""
import hashlib
import json
from pathlib import Path
import tokenizers

assert tokenizers.__version__ == '0.22.2'
out = Path(__file__).resolve().parent
source = out.parents[1] / 'model-bundles/parity-gemma/tokenizer.json'
data = json.loads(source.read_text())
vocab = data['model']['vocab']
rules = [['a', 'a'], ['b', 'c'], ['a', 'b']]
alphabet = 'ABCDEFGHIJKLMNOPQR'
rules += [[alphabet[:n-1], alphabet[n-1:n]] for n in range(2, 19)]
for left, right in rules:
    if left + right not in vocab:
        vocab[left + right] = max(vocab.values()) + 1
data['model']['merges'] = rules
path = out / 'ranked-tokenizer.json'
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
t = tokenizers.Tokenizer.from_file(str(path))
texts = {'empty': '', 'competing-ranks': 'abc', 'overlap-leftmost': 'aaaaa',
         'before-eight': alphabet[:8], 'exact-eight': alphabet[:9],
         'after-eight': alphabet[:10], 'exact-sixteen': alphabet[:17],
         'after-sixteen': alphabet, 'identity-after-convergence': 'a!b!c!d!e!f!g!h!i!',
         'cross-256': '!' * 254 + 'abc', 'special-boundary': 'a<bos>bc',
         'whitespace': '  a  b\t\r\n', 'unicode': 'e\u0301🙂\U0001FAE8',
         'literal-marker': '</w>'}
result = {'reference_library': 'huggingface/tokenizers', 'reference_version': tokenizers.__version__,
          'tokenizer_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'cases': [{'name': name, 'text': text, 'token_ids': t.encode(text).ids} for name, text in texts.items()]}
(out / 'ranked-reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print([(c['name'], len(c['token_ids'])) for c in result['cases']])

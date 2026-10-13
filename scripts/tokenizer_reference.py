#!/usr/bin/env python3
"""Regenerate independent exact-ID fixtures with tokenizers==0.22.2.
Usage: tokenizer_reference.py TOKENIZER_JSON OUTPUT_JSON [--production]
The production artifact is local; its content is pinned by SHA-256 in the output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tokenizers

parser = argparse.ArgumentParser()
parser.add_argument('tokenizer', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--production', action='store_true')
args = parser.parse_args()
assert tokenizers.__version__ == '0.22.2', tokenizers.__version__
tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
texts = {
    'empty': '', 'single': 'a', 'competing-ranks': 'abc',
    'overlap': 'aaaaa', 'new-candidates': 'Raster', 'literal-marker': '</w>',
    'whitespace': '  a  b\t\r\n', 'unicode': 'café e\u0301 中🙂\U0001FAE8',
    'special-boundary': 'a<bos>bc<eos>a', 'raw-special': '<bos>',
    'wrapped': '<bos><|turn>user\nHi,  café! 🙂\nRaster?<turn|>\n<|turn>model\n',
}
if args.production:
    article = ('Accurate tokenization preserves punctuation, repeated spaces, Unicode, and the order of ranked merges. '
               'Each checkpoint records the next bounded batch of work. ')
    text = article
    while len(tokenizer.encode(text).ids) < 3000:
        text += article
    texts['long-3000'] = text
else:
    texts['long-3000'] = 'abc!' * 1000
result = {
    'reference_library': 'huggingface/tokenizers', 'reference_version': tokenizers.__version__,
    'tokenizer_sha256': hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
    'cases': [{'name': name, 'text': text, 'token_ids': tokenizer.encode(text).ids} for name, text in texts.items()],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(args.output, [(c['name'], len(c['token_ids'])) for c in result['cases']])

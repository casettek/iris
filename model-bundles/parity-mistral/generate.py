#!/usr/bin/env python3
"""Regenerate the pinned `mistral-dense` parity bundle using only Python's standard library.

Integer xorshift32 weights and a small independent tokenizer oracle make the
bytes independent of platform floating point, Python hash seeds, and ML tools.
The oracle's Unicode categories are Python's, so a Python built on another
Unicode version could cut a text differently; none of the corpus's characters
is one whose category has changed.
Run with --check to verify reproducibility without writing files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import unicodedata

ROOT = Path(__file__).resolve().parent
SEED = 0x3A7E51C9
ONE = 65536
Q8_0 = 8  # detwgt v3 element width for Q8_0 blocks
HIDDEN, FFN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 128, 2048, 4, 4, 2, 32, 512

# The tokenizer's word split, as `tokenizer.json` writes it.
PATTERN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
           r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
           r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def scale_q24(bits):
    """A half-precision scale's exact value in units of 2^-24 (det_num::q8)."""
    exponent, mantissa = (bits >> 10) & 0x1F, bits & 0x3FF
    magnitude = mantissa if exponent == 0 else (1024 + mantissa) << (exponent - 1)
    return -magnitude if bits & 0x8000 else magnitude


def round_half_even(value, shift):
    floor = value >> shift
    remainder = value - (floor << shift)
    half = 1 << (shift - 1)
    return floor + 1 if remainder > half or (remainder == half and floor & 1) else floor


def byte_char(byte):
    """The character a byte-level tokenizer writes a byte as."""
    if byte <= 0x20:
        return chr(0x100 + byte)
    if 0x7F <= byte <= 0xA0:
        return chr(0x121 + byte - 0x7F)
    return chr(0x143) if byte == 0xAD else chr(byte)


def as_bytes(text):
    return "".join(byte_char(byte) for byte in text.encode())


def word_pattern():
    """`PATTERN` for Python's `re`, which has no `\\p{..}`: each category as a character class."""
    def klass(wanted):
        runs, start = [], None
        for code in range(0x110000):
            hit = wanted(unicodedata.category(chr(code)))
            if hit and start is None:
                start = code
            if not hit and start is not None:
                runs.append((start, code - 1))
                start = None
        if start is not None:
            runs.append((start, 0x10FFFF))
        return "".join(re.escape(chr(a)) if a == b else re.escape(chr(a)) + "-" + re.escape(chr(b))
                       for a, b in runs)

    # The one category that stands outside brackets needs its own.
    pattern = PATTERN.replace(r"|\p{N}|", r"|[\p{N}]|")
    for name, wanted in [
        (r"\p{Lu}", lambda c: c == "Lu"), (r"\p{Lt}", lambda c: c == "Lt"),
        (r"\p{Lm}", lambda c: c == "Lm"), (r"\p{Lo}", lambda c: c == "Lo"),
        (r"\p{Ll}", lambda c: c == "Ll"), (r"\p{M}", lambda c: c[0] == "M"),
        (r"\p{L}", lambda c: c[0] == "L"), (r"\p{N}", lambda c: c[0] == "N"),
    ]:
        pattern = pattern.replace(name, klass(wanted))
    return re.compile(pattern)


def build():
    state = SEED
    tensors = []

    def next_state():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state

    def tensor(name, dims, amplitude, center=0):
        values = []
        for _ in range(math.prod(dims)):
            # Nonzero, signed fractional coefficients at several magnitudes.
            value = next_state() % (2 * amplitude) - amplitude
            values.append(center + (value if value else 1))
        tensors.append((name, dims, 32, values))

    def blocks(name, dims, scale_base):
        """Q8_0 rows: per 32 values a half-precision scale near `scale_base`
        (bits), then 32 signed bytes."""
        payload = bytearray()
        for _ in range(math.prod(dims) // 32):
            payload += struct.pack("<H", scale_base + next_state() % 64)
            payload += bytes((next_state() % 255 - 127) & 0xFF for _ in range(32))
        tensors.append((name, dims, Q8_0, bytes(payload)))

    prefix = "model.language_model."
    # The head is the embedding table: this family ties them.
    blocks(prefix + "embed_tokens.weight", [VOCAB, HIDDEN], 0x1400)
    tensor(prefix + "norm.weight", [HIDDEN], 8192, ONE)
    for layer in range(LAYERS):
        at = prefix + f"layers.{layer}."
        for name, dims in [
            ("self_attn.q_proj.weight", [HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.k_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.v_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.o_proj.weight", [HIDDEN, HEADS * HEAD_DIM]),
            # 2048-wide so each MLP matrix fills one Q8_0 page and spills
            # into a second, partial one.
            ("mlp.gate_proj.weight", [FFN, HIDDEN]),
            ("mlp.up_proj.weight", [FFN, HIDDEN]),
            ("mlp.down_proj.weight", [HIDDEN, FFN]),
        ]:
            blocks(at + name, dims, 0x1000)
        for name in ["input_layernorm.weight", "pre_feedforward_layernorm.weight"]:
            tensor(at + name, [HIDDEN], 8192, ONE)

    weights = bytearray(b"DNWGTV0\0" + struct.pack("<IIQ", 3, 1, len(tensors)))
    for name, dims, width, data in tensors:
        name = name.encode()
        row_width = dims[-1]
        if width == Q8_0:
            # Row mass of the dequantized values: q · scale in Q16.16, rounded
            # half to even, exactly as det_num::q8::dequantize_row.
            row_bytes = row_width // 32 * 34
            mass = 0
            for start in range(0, len(data), row_bytes):
                row = data[start:start + row_bytes]
                total = 0
                for block in range(0, row_bytes, 34):
                    scale = scale_q24(struct.unpack_from("<H", row, block)[0])
                    total += sum(abs(round_half_even(q * scale, 8))
                                 for q in struct.unpack_from("<32b", row, block + 2))
                mass = max(mass, total)
            payload = data
        else:
            mass = max(sum(abs(x) for x in data[i:i + row_width])
                       for i in range(0, len(data), row_width))
            payload = struct.pack("<" + "i" * len(data), *data)
        weights += struct.pack("<I", len(name)) + name + struct.pack("<I", len(dims))
        weights += struct.pack("<" + "Q" * len(dims), *dims)
        weights += struct.pack("<QIQQ", math.prod(dims), width, len(payload), mass)
        weights += b"\0" * (-len(weights) % 64)
        weights += payload

    # A byte-level vocabulary: the special tokens, one token a byte, and the
    # results of the merges. A token's text is its bytes, one character each.
    specials = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]"]
    words = specials + [byte_char(byte) for byte in range(256)]
    merges = []

    def merge(left, right):
        if left + right not in words:
            merges.append([left, right])
            words.append(left + right)

    # Words that grow from the left, one byte a merge. Some are whole words of
    # the corpus, which the tokenizer takes as tokens without merging.
    for word in ["Hi", " cafés", "Raster", " the", " and", "abc", " \U0001F642", "\n\n", "  ", " d"]:
        text = as_bytes(word)
        for end in range(2, len(text) + 1):
            merge(text[:end - 1], text[end - 1])
    # A run of forty dashes is one token, of more pieces than any Gemma token.
    for left, right in [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 8)]:
        merge("-" * left, "-" * right)
    for upper in "ZYXWVUTSRQ":
        for lower in "abcdefghijklmnopqrstuvwxyz":
            if len(words) < VOCAB:
                merge(upper, lower)
    vocab = {word: i for i, word in enumerate(words)}
    assert len(vocab) == VOCAB and len(words) == VOCAB
    tokenizer = {"version": "1.0", "truncation": None, "padding": None, "normalizer": None,
                 "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                     {"type": "Split", "pattern": {"Regex": PATTERN}, "behavior": "Isolated", "invert": False},
                     {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
                 "post_processor": None, "decoder": None,
                 "model": {"type": "BPE", "vocab": vocab, "merges": merges, "dropout": None,
                           "unk_token": None, "continuing_subword_prefix": None,
                           "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                           "ignore_merges": True},
                 "added_tokens": [{"id": vocab[word], "content": word, "special": True,
                                   "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
                                  for word in specials]}
    config = {"model_type": "mistral3", "text_config": {
        "model_type": "ministral3", "hidden_act": "silu",
        "hidden_size": HIDDEN, "intermediate_size": FFN, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "vocab_size": VOCAB, "max_position_embeddings": 262144, "sliding_window": None,
        "rms_norm_eps": 0.00001, "tie_word_embeddings": True,
        "rope_parameters": {
            "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0, "llama_4_scaling_beta": 0.1,
            "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
            "rope_theta": 1000000.0, "rope_type": "yarn"}}}

    split = word_pattern()
    rules = {tuple(pair): (rank, "".join(pair)) for rank, pair in enumerate(merges)}

    def token_ids(prompt, raw):
        if not raw:
            prompt = "<s>[INST]" + prompt.strip() + "[/INST]"
        ids, plain = [], ""

        def flush():
            # Every character is in exactly one word.
            assert "".join(split.findall(plain)) == plain, plain
            for word in split.findall(plain):
                pieces = [byte_char(byte) for byte in word.encode()]
                # A whole word that is a token is that token.
                if "".join(pieces) in vocab:
                    pieces = ["".join(pieces)]
                # Independent exhaustive oracle: choose globally by (rank, position).
                while True:
                    candidates = [(rules[tuple(pieces[i:i + 2])][0], i, rules[tuple(pieces[i:i + 2])][1])
                                  for i in range(len(pieces) - 1) if tuple(pieces[i:i + 2]) in rules]
                    if not candidates:
                        break
                    _, i, merged = min(candidates)
                    pieces[i:i + 2] = [merged]
                ids.extend(vocab[piece] for piece in pieces)

        while prompt:
            special = next((s for s in sorted(specials, key=len, reverse=True) if prompt.startswith(s)), None)
            if special:
                flush()
                plain = ""
                ids.append(vocab[special])
                prompt = prompt[len(special):]
            else:
                plain, prompt = plain + prompt[0], prompt[1:]
        flush()
        return ids

    cases = []
    for name, prompt, raw, tokens in [
        ("short", "Hi, é!", True, 1),
        ("near-window", "abcdefghijklmn?!", True, 8),
        # A long merged token, blanks after line breaks, and capitals after a
        # letter with no case: what the word scanner has to hold back for.
        # Long enough for seven ranges of six tokens. It starts with a word of
        # three tokens, so the first range has neighbouring tokens of one
        # segment, and the model answers it with bytes that are not text.
        ("over-window", "QRS Hi,  cafés! \U0001F642\nRaster? " + "-" * 40
         + "\n\n  中ABC d the and abc ZaZb 7 8", False, 8),
    ]:
        cases.append({"name": name, "prompt": prompt, "raw_prompt": raw,
                      "tokens": tokens, "input_token_ids": token_ids(prompt, raw)})
    assert vocab["-" * 40] in cases[2]["input_token_ids"]
    assert len(cases[2]["input_token_ids"]) > 36
    encode = lambda obj: (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
    files = {"model.detwgt": bytes(weights), "config.json": encode(config),
             "tokenizer.json": encode(tokenizer), "cases.json": encode(cases)}
    files["checksums.json"] = encode({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in build().items():
        path = ROOT / name
        if args.check:
            if path.read_bytes() != data:
                raise SystemExit(f"model bundle differs: {path}")
        else:
            path.write_bytes(data)
    print("Parity model bundle verified" if args.check else "Parity model bundle regenerated")

#!/usr/bin/env python3
"""Regenerate the pinned parity bundle using only Python's standard library.

Integer xorshift32 weights and a small independent tokenizer oracle make the
bytes independent of platform floating point, Python hash seeds, and ML tools.
Run with --check to verify reproducibility without writing files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
SEED = 0x51A7E123
ONE = 65536
Q8_0 = 8  # detwgt v3 element width for Q8_0 blocks


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
    # Scale bits 0x1000 ≈ 2^-11 and 0x1400 ≈ 2^-10: weights up to about
    # 0.0625 and 0.125, the magnitudes of the Q16.16 amplitudes 4096 and 8192.
    blocks(prefix + "embed_tokens.weight", [512, 128], 0x1000)
    blocks(prefix + "embed_tokens_per_layer.weight", [512, 128], 0x1400)
    tensor(prefix + "per_layer_model_projection.weight", [128, 128], 4096)
    tensor(prefix + "per_layer_projection_norm.weight", [32], 8192, ONE)
    tensor(prefix + "norm.weight", [128], 8192, ONE)
    blocks(prefix + "lm_head.weight", [512, 128], 0x1400)
    for layer in range(4):
        at = prefix + f"layers.{layer}."
        for name, dims in [
            ("self_attn.q_proj.weight", [128, 128]),
            ("self_attn.k_proj.weight", [64, 128]),
            ("self_attn.v_proj.weight", [64, 128]),
            ("self_attn.o_proj.weight", [128, 128]),
            # 2048-wide so each MLP matrix fills one Q8_0 page and spills
            # into a second, partial one.
            ("mlp.gate_proj.weight", [2048, 128]),
            ("mlp.up_proj.weight", [2048, 128]),
            ("mlp.down_proj.weight", [128, 2048]),
        ]:
            blocks(at + name, dims, 0x1000)
        for name, dims in [
            ("per_layer_input_gate.weight", [32, 128]),
            ("per_layer_projection.weight", [128, 32]),
        ]:
            tensor(at + name, dims, 4096)
        for name, width in [
            ("input_layernorm.weight", 128),
            ("post_attention_layernorm.weight", 128),
            ("pre_feedforward_layernorm.weight", 128),
            ("post_feedforward_layernorm.weight", 128),
            ("post_per_layer_input_norm.weight", 128),
            ("self_attn.q_norm.weight", 32),
            ("self_attn.k_norm.weight", 32),
        ]:
            tensor(at + name, [width], 8192, ONE)
        tensors.append((at + "layer_scalar", [1], 32, [ONE]))

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

    specials = ["<unk>", "<pad>", "<bos>", "<|turn>", "<turn|>", "<eos>"]
    words = specials + ["</w>", "\n", "▁", "é"]
    words += [chr(i) for i in range(33, 127)]
    words += [f"<0x{i:02X}>" for i in range(256)]
    merges = [["b", "c"], ["a", "b"], ["H", "i"], ["R", "a"], ["Ra", "s"], ["Ras", "t"], ["Rast", "e"], ["Raste", "r"]]
    words += [a + b for a, b in merges]
    words += [f"word{i}" for i in range(512 - len(words))]
    vocab = {word: i for i, word in enumerate(words)}
    assert len(vocab) == 512
    tokenizer = {"version": "1.0", "truncation": None, "padding": None,
                 "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
                 "pre_tokenizer": None, "post_processor": None, "decoder": None,
                 "model": {"type": "BPE", "vocab": vocab, "merges": merges, "dropout": None,
                           "unk_token": "<unk>", "continuing_subword_prefix": None,
                           "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False},
                 "added_tokens": [{"id": vocab[word], "content": word, "special": True,
                                   "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
                                  for word in specials]}
    config = {"text_config": {
        "hidden_size": 128, "intermediate_size": 2048, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "hidden_size_per_layer_input": 32, "vocab_size": 512, "sliding_window": 16,
        "num_kv_shared_layers": 2, "rms_norm_eps": 0.000001,
        "tie_word_embeddings": False, "final_logit_softcapping": 1.0,
        "eos_token_id": [vocab["<eos>"]],
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000},
            "full_attention": {"rope_theta": 1000000, "partial_rotary_factor": 0.5}}}}

    def token_ids(prompt, raw):
        if not raw:
            prompt = "<bos><|turn>user\n" + prompt.strip() + "<turn|>\n<|turn>model\n"
        pieces = []
        while prompt:
            special = next((s for s in specials if prompt.startswith(s)), None)
            if special:
                pieces.append(special)
                prompt = prompt[len(special):]
            else:
                ch, prompt = prompt[0], prompt[1:]
                ch = "▁" if ch == " " else ch
                pieces.extend([ch] if ch in vocab else [f"<0x{b:02X}>" for b in ch.encode()])
        rules = {tuple(pair): (rank, "".join(pair)) for rank, pair in enumerate(merges)}
        # Independent exhaustive oracle: choose globally by (rank, position).
        while True:
            candidates = [(rules[tuple(pieces[i:i+2])][0], i, rules[tuple(pieces[i:i+2])][1])
                          for i in range(len(pieces)-1) if tuple(pieces[i:i+2]) in rules
                          and pieces[i] not in specials and pieces[i+1] not in specials]
            if not candidates:
                return [vocab[p] for p in pieces]
            _, i, merged = min(candidates)
            pieces[i:i+2] = [merged]

    cases = []
    for name, prompt, raw, tokens in [
        ("short", "Hi, é!", True, 1),
        ("near-window", "abcdefghijklmn?!", True, 8),
        ("over-window", "Hi,  cafés! 🙂\nRaster?", False, 8),
    ]:
        cases.append({"name": name, "prompt": prompt, "raw_prompt": raw,
                      "tokens": tokens, "input_token_ids": token_ids(prompt, raw)})
    assert len(cases[1]["input_token_ids"]) == 15
    assert len(cases[2]["input_token_ids"]) > 16
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

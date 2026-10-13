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


def build():
    state = SEED
    tensors = []

    def tensor(name, dims, amplitude, center=0):
        nonlocal state
        values = []
        for _ in range(math.prod(dims)):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            # Nonzero, signed fractional coefficients at several magnitudes.
            value = state % (2 * amplitude) - amplitude
            values.append(center + (value if value else 1))
        tensors.append((name, dims, values))

    prefix = "model.language_model."
    tensor(prefix + "embed_tokens.weight", [512, 128], 4096)
    tensor(prefix + "embed_tokens_per_layer.weight", [512, 32], 8192)
    tensor(prefix + "per_layer_model_projection.weight", [32, 128], 4096)
    tensor(prefix + "per_layer_projection_norm.weight", [8], 8192, ONE)
    tensor(prefix + "norm.weight", [128], 8192, ONE)
    tensor(prefix + "lm_head.weight", [512, 128], 8192)
    for layer in range(4):
        at = prefix + f"layers.{layer}."
        for name, dims in [
            ("self_attn.q_proj.weight", [128, 128]),
            ("self_attn.k_proj.weight", [64, 128]),
            ("self_attn.v_proj.weight", [64, 128]),
            ("self_attn.o_proj.weight", [128, 128]),
            ("mlp.gate_proj.weight", [512, 128]),
            ("mlp.up_proj.weight", [512, 128]),
            ("mlp.down_proj.weight", [128, 512]),
            ("per_layer_input_gate.weight", [8, 128]),
            ("per_layer_projection.weight", [128, 8]),
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
        tensors.append((at + "layer_scalar", [1], [ONE]))

    weights = bytearray(b"DNWGTV0\0" + struct.pack("<IIQ", 2, 1, len(tensors)))
    for name, dims, values in tensors:
        name = name.encode()
        row_width = dims[-1]
        mass = max(sum(abs(x) for x in values[i:i + row_width])
                   for i in range(0, len(values), row_width))
        weights += struct.pack("<I", len(name)) + name + struct.pack("<I", len(dims))
        weights += struct.pack("<" + "Q" * len(dims), *dims)
        weights += struct.pack("<QIQQ", len(values), 32, len(values) * 4, mass)
        weights += b"\0" * (-len(weights) % 64)
        weights += struct.pack("<" + "i" * len(values), *values)

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
        "hidden_size": 128, "intermediate_size": 512, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "hidden_size_per_layer_input": 8, "vocab_size": 512, "sliding_window": 16,
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
        ("over-window", "Hi,  café! 🙂\nRaster?", False, 8),
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

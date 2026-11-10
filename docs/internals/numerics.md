# Numerics

Inference is integer arithmetic with every rounding and overflow rule fixed,
so direct inference, the native staged pass, and Raster's RISC Zero replay
produce bit-identical results. Floating point appears only once, when a model
is converted into a bundle; everything committed after that is integers.

## Values

- **Activations** are Q16.16: signed 32-bit, 16 fractional bits.
- **Large matrices** are `Q8_0` blocks, the layout GGUF calls `Q8_0`: per 32
  weights, a half-precision scale then 32 signed bytes (34 bytes). These are
  the token and per-layer embedding tables, the logits head, and every
  attention and MLP projection.
- **Everything else** (norm weights, the per-layer-embedding gate and
  projections, scalars) is Q16.16.

[`det_num`](../../crates/det-num) defines every operation. Its `q8` module is
the `Q8_0` arithmetic that all three paths call.

## `Q8_0` arithmetic

Every half-precision value is an integer multiple of `2^-24`, so a block's
scale decodes exactly to an integer in units of `2^-24`. Its weights are then
exact as well:

- **Products.** A row's dot product with Q16.16 activations sums each block
  exactly in `i64`, multiplies by the block's scale, and sums the blocks in
  `i128`. That total is the exact real dot product. It is rounded once to Q16.16
  (round half to even) and saturated. Summation order cannot change the result.
- **Lookups.** An embedding row is dequantized value by value: `q · scale`,
  rounded once to Q16.16 and saturated.

A dequantized matrix is never used for a product. Its per-weight rounding
would give different bits from the exact product.

Q16.16 products accumulate in wrapping Q32.32 and requantize once, as before.
Norms, softmax, RoPE and GELU are `det_num`'s fixed-point routines.

## `Q8_0` products on a GPU

Built with the `gpu-kernels` feature, native-direct and native-staged run the
transformer layer's `Q8_0` matrix products on an NVIDIA GPU
(`crates/inference/kernels/cuda/q8_product.cu`). The kernel computes the same
integers as `det_num::q8::dot_row`, with no floating point:

- An activation is split into `v >> 16` and `v & 0xFFFF`. With weights in
  `[-128, 127]`, thirty-two products of either half stay below `2^31`, so each
  block sums exactly in 32-bit arithmetic and the halves recombine to the exact
  block sum.
- Each block sum times its scale is added to a 128-bit total held as two 64-bit
  words, then rounded once to Q16.16 (ties to even) and saturated.

`row_dot` remains the definition of every value. The layer kernel has one place
that multiplies by a weight matrix, `linear_slab_from_source`, and the GPU sits
behind it. After each product, 66 of its outputs (the first, the last and 64
chosen pseudo-randomly) are recomputed with `row_dot`; a disagreement fails the
stage instead of publishing its output.

The GPU is never required. A product stays on the CPU when it is small (a
decode step's single token), when the matrix is not `Q8_0`, when a block's scale
is not finite (the CPU reports that error), or when the device cannot be used.
A device failure is reported once on stderr and the rest of the run stays on the
CPU. Attention, norms, RoPE, GELU and the per-layer-embedding products stay on
the CPU, as do Raster stages: nothing a stage commits to changes.

## Weight files

`model.detwgt` v3 holds both encodings. A tensor's `element_width` is 16 or 32
for Q16.16 values, or 8 for `Q8_0` rows of blocks. A `Q8_0` tensor's rows must
be whole blocks. Readers keep the two kinds separate: `Q8_0` tensors are read
as block rows, never as Q16.16 values.

`iris model convert` writes a v3 bundle from a GGUF file
([model import](../workflows/model-import.md#converting-a-gguf-model)). It
copies `Q8_0` blocks byte for byte and converts float tensors to Q16.16 with
`det_num`'s exact conversion.

## Raster pages

A Raster byte region is split into fixed-size pages, and a stage's tiles read
one page at a time, so a page must hold whole rows.

| Region | Page size | Why |
| --- | --- | --- |
| Q16.16 matrices | 196,608 | Whole rows of `width · 4` bytes |
| `Q8_0` matrices and embedding tables | 195,840 | Whole rows of `width / 32 · 34` bytes for every width in use |
| `Q8_0` logits head | 52,224 | Whole rows, and at most 384 rows of a 128-wide head in one tile call |

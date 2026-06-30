# Disaggregated Inference — Dell (prefill) → KV-stream → M1 (decode)
### Scope · cable-filling · NO cross-platform collective
*Dell-CC, 2026-06-29*

## Is this the NVIDIA↔Apple Gloo thing? NO — and that's the point.
This uses **no Gloo, no NCCL, no cross-platform collective.** The only thing crossing the cable is
a **one-way byte stream of the KV cache** — a plain socket/HTTP transfer, Dell → M1. It sidesteps
the single biggest unknown in the TP build (does an all-reduce span NVIDIA↔Apple over TB?). If the
TP Gloo probe fails, **this still works.**

## Goal
Split one inference into its two halves and run each on the machine it suits:
- **Prefill on the Dell** (CUDA, fast compute) — chew the whole prompt, build the KV cache.
- **Stream the KV cache one-way** Dell → M1 over the cable (this is what fills the cable).
- **Decode on the M1** (big unified memory) — generate tokens from the streamed KV.

## Why this split (plays to each machine's strength)
- **Prefill = compute-bound** (big matmuls over N prompt tokens) → Dell's ~58 TFLOPS wins.
- **Decode = memory-bandwidth-bound** (1 token at a time, read all weights) → M1's unified memory.
- **The KV cache is big** → the one-way stream genuinely fills the cable (vs pipeline's tiny
  per-token activation).
- **For LONG prompts** (R1 reasoning chains), the Dell's fast prefill beats the M1 doing it alone →
  single-request latency drops AND the cable lights up.

## Architecture
1. Both nodes load the full model (Dell: CUDA, +RAM offload if model >8GB; M1: MLX/MPS, 16GB).
2. Dell runs prefill → KV cache for all layers ≈ `N_tokens × n_layers × 2 × hidden × dtype`.
3. Dell serializes the KV → streams over a socket to the M1 (one-way; ~1GB for a 1k-token prompt on
   an 8B → ~0.6s at cable speed = a real cable burst).
4. M1 seeds its decode loop with the KV → generates tokens → returns text.

## Build phases
- **Phase 0 — KV handoff proof:** Dell computes a KV cache for a short prompt, serializes it,
  streams to the M1, M1 resumes decode from it, output **matches a single-node run.** No collective
  — just a socket + a correct tensor layout.
- **Phase 1 — Layout match:** the Dell's KV memory layout must equal the M1's expected layout (same
  weights, dtype, RoPE, head ordering). This is the actual hard part.
- **Phase 2 — Long-prompt run + measure:** big prompt; record prefill time (Dell), transfer time +
  **cable utilization**, decode time (M1), total vs a single-node baseline.
- **Phase 3 (optional) — Pipeline requests:** while M1 decodes A, Dell prefills B → throughput win.

## Honest risks
1. **KV layout must match exactly across two different engines** (Dell tinygrad/PyTorch vs M1 MLX).
   Mismatched KV format → garbage out. Same model, dtype, RoPE. This is the real work, not the wire.
2. **Dell prefill needs the model to fit its 8GB GPU** for the speed win — else prefill offloads to
   RAM and crawls, negating the benefit. → small/quantized model, or accept slower prefill.
3. **Single-request win is conditional on LONG prompts.** Short prompt = tiny prefill = transfer
   overhead not worth it. Shines for R1-style long reasoning, not chat.
4. **Both nodes hold the full model** (memory cost on each).

## vs the TP build — pick by what you optimize
- **TP (Gloo):** lower single-request latency via parallel compute every layer. RISK = the
  cross-platform collective may not exist over TB. Fills cable on prefill.
- **Disaggregated (this):** SAFE — one-way stream, no collective. Fills the cable with the KV burst.
  Single-request win only on long prompts; also unlocks throughput pipelining.

## Recommendation
**Build disaggregated first** — it's the lower-risk cable-filler (no Gloo gamble), and it's the
fallback the TP scope already names. Run the TP **Gloo probe in parallel** as a cheap go/no-go; if
Gloo spans the two, TP stacks on top as the latency play. **Disaggregated is the floor; TP is the
stretch.**

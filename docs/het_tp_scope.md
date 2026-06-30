# Heterogeneous Tensor-Parallel Inference — Dell (CUDA) ↔ M1 (Metal)
### Scope · long-context reasoning · cable-filling · NVIDIA + Apple
*Dell-CC, 2026-06-29*

## Goal
Run an LLM (DeepSeek-R1 distill, long-context) **tensor-parallel** across the Dell RTX 4070
(CUDA) and the M1 (Apple MPS), all-reducing over the Thunderbolt cable via **PyTorch + Gloo**.
Targets: (a) **fill the cable during prefill**, (b) **run long-context reasoning faster than
pipeline**. Genuinely novel — there is no known NVIDIA↔Apple tensor-parallel implementation.

## Why TP (vs the ring/pipeline exo gave us)
- **Pipeline** (what we ran): each node owns *different layers*; runs sequentially; only a tiny
  activation crosses the cable per token → cable idle, no parallel compute.
- **Tensor-parallel**: both nodes run *half of every layer at once* (parallel compute) and
  **all-reduce** the partial results each layer. The all-reduce moves big tensors → fills the
  cable on prefill, and the compute genuinely runs in parallel.

## Architecture
1. PyTorch on both nodes: Dell = CUDA, M1 = MPS.
2. `torch.distributed`, **Gloo backend** (cross-platform — NCCL is NVIDIA-only and can't span
   Apple), peers over the TB-IP link (10.55.0.1 / 10.55.0.2).
3. Linear layers sharded: **column-parallel** (q/k/v, gate, up) + **row-parallel** (o_proj, down).
   Each node holds its half of every layer's weights — weights stay local, never streamed.
4. Per layer: each node computes its shard → **Gloo all-reduce over the cable** → next layer.
5. **Asymmetric split (critical):** Dell ≈ 58 TFLOPS, M1 ≈ 4.6 → ~12:1. Each layer is split
   ~proportional to speed so both finish together; a 50/50 split makes the Dell idle waiting
   for the M1 every layer. The ratio is tuned empirically.

## Build phases
- **Phase 0 — Feasibility gate (make-or-break):** PyTorch on both + a `torch.distributed` Gloo
  **all-reduce of one tensor, Dell↔M1, over the cable.** If Gloo won't span NVIDIA↔Apple over
  TB-IP, the whole approach is dead. *Test this before anything else.* (~1–2 hrs)
- **Phase 1 — One TP linear:** shard a single matmul, all-reduce, verify bit-correctness vs a
  single-node run.
- **Phase 2 — One TP transformer block:** column+row-parallel attention + MLP, correctness check.
- **Phase 3 — Full model sharded:** DeepSeek-R1-distill (start at 8B fp16, ~16GB → ~Dell 8GB +
  M1 8GB), asymmetric split tuned.
- **Phase 4 — Long-context run + measure:** tok/s, **cable utilization during prefill**, all vs
  the pipeline baseline (21 t/s / 2.86% cable).

## Honest risks — this is research-grade, NOT a guaranteed win
1. **Gloo over TB-IP between NVIDIA and Apple is untested.** Phase 0 settles it; it may simply
   not connect. This is the single biggest unknown.
2. **MPS gaps:** PyTorch's Apple-Metal backend has missing/buggy ops; the M1's shard may hit one.
3. **Heterogeneous sync penalty:** TP synchronizes *every layer*. The 12:1 speed gap means even
   with a tuned split, the slower node + the all-reduce latency cap throughput. **TP may end up
   SLOWER than pipeline for short prompts.**
4. **Cable latency:** per-layer all-reduce over ~13 Gbps IP-over-TB. Helps prefill (big tensors),
   hurts decode (small + very frequent syncs).
5. **Net:** the cable will fill on prefill; whether it's *faster* is conditional on long context +
   good tuning. Treat this as an experiment with a real chance of "fills the cable, doesn't beat
   pipeline."

## Success criteria
- **Correct:** TP output == single-node output.
- **Cable:** prefill utilization ≫ 2.86%.
- **Speed:** long-context (big prompt) **faster than the pipeline baseline** ← the real test.

## Fallback if Phase 0 fails
If Gloo won't span the two, pivot to **disaggregated inference** (prefill on Dell → stream the
KV cache one-way → decode on M1) — the other cable-filler, EXO-proven, no cross-platform
collective required (it's a one-way byte stream, not an all-reduce).

## First action
Phase 0 — install PyTorch on both, run the Gloo all-reduce Dell↔M1 over the cable. Go / no-go in
an afternoon.

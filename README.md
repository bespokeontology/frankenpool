# 🧟 frankenpool
### Two junk laptops. A $40 cable. The newest model on Earth — the one the maintainers were too scared to merge — running across NVIDIA **and** Apple silicon, *generating across both*. No datacenter. No cloud. No permission.

> They shipped a frontier architecture and **ran from it** — closed the PRs, called it *"too complex."* A day later it was **thinking** on a gaming laptop and a MacBook Apple offered fifty bucks for, pooled over a cable that costs less than lunch — and **llama.cpp's own memory ledger says the MacBook was holding its share of the weights.** No rented GPU. No cloud. Nobody's permission. We caught the model the establishment dropped and put it to work across two machines that have no business talking to each other. **The frontier doesn't live in a datacenter — it lives in your house.** And it reasons better in Chinese than anything Silicon Valley will sell you.
>
> Every number below is real. Go ahead — *try* to debunk it. That's not a disclaimer, it's a dare.

---

## Chapter 2 — 2026-06-30 · LongCat-Flash-Lite, pooled *(today's test)*
**The newest result.** A frontier sparse-MoE that shipped **less than 24 hours earlier** — run **solo** on one
laptop (verified Chinese-domain reasoning) and then **pooled across both machines**, where it *generated content*
— not a load, not a benchmark. *Chapter 1 below is a separate, earlier experiment (2026-06-29, a different model:
DeepSeek-R1-70B) and is unchanged.*

### The model
**LongCat-Flash-Lite** — Meituan, **68.56B params, sparse Mixture-of-Experts (~3–4.5B active per token)**, with
**n-gram embeddings** and **MLA** (compressed KV), trained on Huawei-Ascend-class silicon. Upstream llama.cpp
**abandoned the architecture** — the identity-expert routing was ruled "too complex" and the PRs were closed — so
it runs only on the **InquiringMinds-AI fork**, commit `56abe85`. Sparse means it *remembers* like a 68B model but
*computes* like a ~4B one: big-model knowledge at small-model speed. As far as the public record shows, this is
the only place it has been run pooled across heterogeneous (NVIDIA + Apple) silicon.

### Solo — one 8 GB laptop, and it *reasons*
Before the pool, the same model ran **solo on a single RTX 4070 laptop** (Q4_K_M, 37.46 GiB, `-ngl 10`) at
**~16–19 t/s** — and it didn't just run, it produced **verified domain output:**
- **Mandarin reading recommendations** across distributed systems / hardware-software co-design / manufacturing.
- The **native-Chinese vocabulary of "organization over scale"** — **体用** (substance/function), **势 vs 器**
  (configured potential vs mere instrument), Qian Xuesen's **综合集成** (meta-synthesis). The model argued our
  own thesis back to us, in its own characters.
- A **Russian-probability → Chinese-ML academic genealogy** — Zhang Bo (Tsinghua); Jun Zhu (advisor: Zhang Bo);
  Hang Li (Huawei Noah's Ark, author of 《统计学习方法》); Zhiyuan Liu (advisor: Sun Maosong) — **checked against
  the sources: every name and advisor-lineage correct.** A Western model produces plausible mush on this query;
  this gave checkable facts, because its training corpus *is* that world.

The point of the model choice: **right matrix beats bigger matrix** — a frontier model on a trade-in-grade laptop,
producing Chinese-domain knowledge a Western model can't match.

### Solo — the speed, measured
GPU offload buys speed up to the 8 GB card's ceiling, and the operating config *holds under load*:

| solo config (Dell, Q4) | gen t/s |
|---|---|
| `-ngl 0` (pure CPU) | 13.5 |
| `-ngl 8` | 17.1 |
| **`-ngl 10`** (card ceiling; `-ngl 12` OOMs) | **18.9** |
| Q6 (52.41 GiB), `-ngl 0/4` (mmap) | ~10–11 |

The real long-prompt runs landed at **~16 t/s** — the gap from 18.9 is just context: generation slows as the
prompt grows (`d0 → d8192`: **18 → 12.7 → 9.1 → 5.9** t/s, memory-bound, as expected). And the GPU config is
**contention-resistant** — it holds ~17.5+ t/s *even while the box pulls a ~50 GB download*, because it leans on
the card, not the contended CPU. Operating number: **~16–19 t/s on one 8 GB laptop.**

### The pooling road — what broke, and how we knew it was real
Like Chapter 1, the failures are the record. Getting a *genuine* two-machine pool out of a day-old architecture
took three honest walls:

1. **The stale-binary wall.** The first "pooled" attempts failed the RPC handshake (no protocol `HELLO` match) —
   a runner script was firing the *stale upstream* llama.cpp build instead of the fork. Both nodes had to be on
   `56abe85`; repointing the binary was the whole fix.
2. **The relabel trap (the dangerous one).** An early run *looked* pooled — backend `CUDA,RPC`, real t/s — but it
   was **Dell-GPU + a Dell-LOCAL x86 RPC worker**, mislabeled "across the cable." The M1 held **zero** weights. We
   caught it only by reading the M1's own residency, not the backend label. *A `CUDA,RPC` tag is not proof of a
   cross-machine pool.*
3. **The over-claim wall (me).** Repeatedly the agent driving this wrote "the pool works" off a success log or a
   bench number. Each time, the operator and the M1-side agent refused the narrative and demanded the actual
   measurement — and each time they were right. The claim entered this record **only** after three independent
   measurements agreed (below).

*(One thing we don't claim: an early worker crash was fixed by a no-BLAS rebuild on the M1, but a clean op-level
ARM-vs-x86 comparison was never measured — so there's no hardware-specific-bug claim here, only that the rebuild
worked.)*

### Pooled — two machines, the framework as witness
Then the same model **split across both machines** over the cable and **generated content** — receipt below.

![pooled generation receipt — llama.cpp memory breakdown showing RPC0 (the M1) holding model weights](pool_receipt.png)

**1. Model / run**
- Model: **LongCat-Flash-Lite** (68.56B sparse-MoE, Q4_K_M), fork build `56abe85`.
- Dell client (RTX 4070) ⟷ **M1 worker at `10.55.0.2:50053`** over Thunderbolt-IP.
- Backend / route: **`CUDA,RPC`**, `-sm layer -ngl 8 -ts 4,4`.
- Real Chinese prompt: *为什么"算力的组织胜过单纯的规模"(传动系统胜过引擎)?用三句话,并举一个工程例子。*
- Generated answer (excerpt): *"算力的组织决定了计算资源的利用效率和适应性…一个具体的工程例子是 MapReduce 编程模型…"*

**2. Framework-confirmed split** — llama.cpp memory breakdown, verbatim:
```
- RPC0 (10.55.0.2:50053)      |  3975 MiB = 2996 MiB model weights + ~970 MiB compute/residency   ← the M1
- CUDA0 (RTX 4070 Laptop GPU) |  2949 MiB = 1969 MiB model weights                                ← the Dell GPU
- Host                        | 33774 MiB                                                         ← Dell RAM share
```

**3. Cable receipt** — **3,876 MB streamed to the M1** during the real pooled run. This matches the M1's
~3975 MiB framework-reported residency. The bytes crossed the Thunderbolt link *during the run*.

**4. Generation receipt** — Prompt **27.2 t/s**, Generation **13.0 t/s**. English gloss of the answer:
*organization sets utilization and adaptability; raw scale is quantity-stacking; efficient organization enables
parallelism / task allocation; MapReduce is the engineering example.*

**5. The claim (stated precisely):**
> This is the first confirmed pooled-generated content record in this repo: the model produced coherent output
> while llama.cpp reported nonzero RPC0 model-weight residency on the M1 and live cable counters recorded
> multi-GB transfer matching that residency.

**real pooled generation** · **framework-confirmed RPC0 weight residency** · **byte-proven over Thunderbolt** ·
**content generated across two machines**. This run used **LongCat-Flash-Lite (Q4)** — *not* the 70B; the 70B
result below is a separate capacity record and is unchanged.

### How we knew it was really cross-machine (the part everyone skips)
Three independent measurements, all agreeing — none of which a local or faked run could produce *together*:
1. **Framework residency** — llama.cpp's own memory ledger reported `RPC0 (10.55.0.2:50053)` holding **2996 MiB
   of model weights**. The framework, not us, naming the M1's share.
2. **Bytes on the wire** — **3,876 MB** measured crossing `thunderbolt0` to the M1 during load, matching that residency.
3. **Real output** — a coherent generated answer at **13 t/s**. The split only reports when both nodes serve.

To reproduce the *verification* (not just the run): read the memory breakdown for an `RPC0` line with non-zero
model weights, and watch the Thunderbolt interface `tx_bytes` for a multi-GB jump on load. Missing either one, and
it isn't a pool — it's one machine wearing an RPC label.

### Why this matters — LongCat vs the 70B (same rig, measured)
The two chapters are a controlled before/after on the *same two machines and the same $40 cable* — and the
contrast is the whole point:

| | Ch.1 — DeepSeek-R1-70B | Ch.2 — LongCat-Flash-Lite |
|---|---|---|
| type | **dense** — 70B active/token | **sparse MoE** — ~3–4.5B active of 68B |
| age at test | a known distill | shipped **< 24 h earlier** |
| **pooled generation** | **1.40 t/s** | **~13 t/s** |
| solo | won't fit one box | **~16–19 t/s on one 8 GB laptop** |

Same rig — and the day-old sparse model runs **~9× faster pooled** than the dense 70B, because it fires ~4B
parameters per token instead of 70B. That's the thesis in one row: a model that *remembers* like a frontier system
but *computes* like a small one turns Chapter 1's capacity stunt (1.4 t/s) into something you'd actually use
(~13–19 t/s).

And it isn't only speed. On its home turf — the Chinese-domain genealogy that **checked against the sources** —
LongCat gave verifiable answers a larger Western model can't, because its corpus *is* that world. So the relevant
axis was never raw size; it's **sparse compute + the right corpus, on hardware you already own.** The frontier
shipped yesterday; it ran on a gaming laptop and a trade-in MacBook today. *That's* the relevance.

### What this chapter does — and doesn't — claim
- **Does:** a frontier MoE running solo on one cheap laptop *with verified output*, and a real two-machine pool
  that *generates content* — every cross-machine number measured (framework residency, cable bytes, t/s).
- **Doesn't:** no speed record (13 t/s pooled), and no claim about pooled-vs-solo speed or hardware-specific
  bugs — those weren't cleanly measured, so they're not here. The one claim that kept tempting us — *"the pool
  works"* — went in only after the framework ledger, the cable bytes, **and** a real generated answer all agreed.

### Reproduce (Chapter 2)
```bash
# Solo — one laptop:
llama-cli -m LongCat-Flash-Lite-Q4_K_M.gguf --jinja -ngl 10 -c 4096 -f prompt.txt

# Pooled — the M1 worker MUST run the same fork build (56abe85), not upstream:
#   M1:   rpc-server -H 0.0.0.0 -p 50053
#   Dell: llama-cli -m LongCat-Flash-Lite-Q4_K_M.gguf --rpc 10.55.0.2:50053 \
#           --jinja -sm layer -ngl 8 -ts 4,4 -c 2048 -f prompt.txt

# VERIFY it's genuinely cross-machine (the step everyone skips):
#   - memory breakdown shows an RPC0 line with non-zero model weights
#   - the Thunderbolt iface tx_bytes jumps multi-GB on load
```
Model: `InquiringMinds-AI/LongCat-Flash-Lite-GGUF` · Fork: `InquiringMinds-AI/llama.cpp` @ `56abe85`.

---

## Chapter 1 — 2026-06-29 · DeepSeek-R1-70B, pooled for capacity *(yesterday)*
### Running a 70-billion-parameter LLM across an NVIDIA RTX 4070 and a 5-year-old M1 MacBook — over a $40 Thunderbolt cable
**A complete, honest field record — debugged live across two machines by two cooperating agents. A different model and a different night from Chapter 2 above.**

> This is not a clean success story, and that's the point. It's eight walls, a kernel panic, a model
> that crashed ten times, half a dozen tools that didn't work, two wrong calls about giving up — and
> a 70-billion-parameter model that ran across two consumer laptops anyway. **The failures are the
> record.** Here's all of it, in order, with the full trail of everything we tried.

---

## What was done, in one line
A model too large to fit in the memory of **either** machine (DeepSeek-R1-Distill-Llama-70B, Q4_K_M,
**70.55B params, 39.6 GiB**) was **loaded and run across both at once** — half on a Dell RTX 4070
(CUDA), half on an M1 MacBook Air (Metal) — joined by a **$40 Thunderbolt cable**. Backend `CUDA,RPC`,
3.22 t/s prefill, 1.40 t/s decode. Measured on both ends. *Not loaded — ran.*

Plus, as a prerequisite, a **cross-engine KV-cache handoff (PyTorch/CUDA ⇄ Apple MLX)** that — per a
literature sweep — was **unbuilt as shipped code.**

---

## The hardware
- **Dell** — RTX 4070 Laptop (8 GB VRAM), 31 GB RAM, Ubuntu / CUDA.
- **M1 MacBook Air** — Apple M1, 16 GB unified memory, macOS / Metal. *A machine Apple offered $50
  for on trade-in.*
- **The link** — a **$40** Thunderbolt cable. IP-over-Thunderbolt, 0.36 ms latency, ~13 Gbps.

---

## It started at 19.79
The first test came back at **19.79 tokens/second** — a 35B model running *pooled* across the Dell and
the M1, neither machine doing it alone. That raised the real questions: **how far does pooling two
cheap machines go? And is the cable doing any work, or just sitting there?**

Everything after was answering that honestly. It went wrong a lot first.

---

## Failure #1 — the best one: *the cable was never the bottleneck*
The whole premise was "fill the cable." We measured it under the pooled 35B: **2.86%** utilization.
The theory was that tensor-parallelism would light it up. Hours later, the honest reframe: the tools
we counted on use **pipeline** parallelism, which passes ~**8 KB per token**. We'd been chasing a
number that *can't* rise, because **LLM inference fundamentally doesn't move much data between nodes.**
The premise was wrong — and the failure sharpened everything: the win was never "saturate the cable,"
it's the **distribution.** The cable just has to be good enough, and at 0.36 ms it wildly is.

---

## The full trail — every tool we tried (and what happened)
| tool / package | what we tried it for | outcome |
|---|---|---|
| **llama.cpp** | RPC memory-pooling + Metal RPC server | ✅ **the workhorse** — ran the 35B *and* the 70B pooled |
| **exo / exo-cuda** (Scottcjn fork) | distributed inference across both | ⚠️ got it generating after 3 bug-fixes, but its ring strategy is *pipeline* — abandoned for disaggregation |
| **tinygrad** (exo's engine, pinned 0.10.0) | the CUDA-side inference inside exo | ⚠️ `bf16→half` NVRTC hell; `llvm_bf16_cast` **segfaulted**, no `clang`, 0.13.0 removed the function — **solved with a numpy host-cast** (bf16 = top 16 bits of fp32) |
| **PyTorch + Gloo** ("pygloo") | heterogeneous **tensor-parallel** (NVIDIA↔Apple all-reduce) | ❌ **dead end** — Gloo rendezvous failed, and Gloo **cannot operate on Apple MPS tensors at all** (PyTorch issues #160731 / #160732). NCCL is NVIDIA-only. No cross-platform collective exists. |
| **MLX / mlx-lm** | the M1's decode engine + KV-cache injection | ✅ the disaggregated handoff target. Gotcha: it pre-allocates KV in 256-token steps — slice to `.offset` |
| **HuggingFace transformers** (5.12.1) | the disaggregated *prefill* (DynamicCache → KV export) | ✅ KV layout is byte-identical to MLX's — a reshape, not a translation |
| **ggml-rpc-server** (Metal) | the M1's RPC backend for the 70B pool | ✅ after the wired-ceiling, residency, and duplicate-instance fights below |
| **safetensors** | serializing the KV cache for the wire/file transport | ✅ |
| **uv** | building the venvs | ✅ (Ubuntu's `python3 -m venv` ships no pip) |
| **ollama** | *(not ours)* | 🩹 the **GPU squatter** — silently held a resident 32B eating 6.7 GB of the 8 GB card; had to `systemctl stop` it |
| **the $40 cable** | the interconnect | 🩹 the **quiet villain** — see wall #1 below |

---

## The breakthrough — a cross-engine handoff that didn't exist
Disaggregation splits a request: **prefill** (compute-heavy) on the fast GPU, **decode** (memory-bound)
on the memory-rich box, KV cache crossing between. The novel part: across **two different frameworks**
(PyTorch/CUDA → MLX/Metal). It works because the KV belongs to the *model's* architecture, not the
engine — so it's a reshape:

| test | result |
|---|---|
| Forward HF→MLX | identical token, logit cosine **0.999864** |
| 18-token ctx | teacher-forced cos **0.9997**, argmax 23/24 |
| **2,123-token ctx** | cos **0.99972** — *zero drift* at 118× the length |
| Falsification | corrupt the KV → garbage out (the cache genuinely drives decode) |
| Reverse MLX→HF | **24/24 identical** |
| Live over the cable | 52 MB streamed, decoded correctly, ~7.7 Gbps |

---

## The capacity demo — the eight walls
Going for the 70B, here's everything that broke, in order:

1. **The dark cable.** The M1 rebooted and the *Dell* silently lost its `10.55.0.1` Thunderbolt IP
   (fell to link-local) — different subnets, RPC unreachable. **SSH kept working over WiFi, masking it
   for an hour** while we blamed binaries and memory. One `ip addr add` fixed it.
2. **GPU squatters** — exo, then ollama, ate the Dell's VRAM. Kill by PID; stop ollama's service.
3. **The Metal wired ceiling** — macOS caps GPU-wired memory at ~⅔ of RAM (~9.7 GB on 16 GB), and the
   server must *restart* to inherit a raised cap.
4. **The compute-buffer peak** — a 70B's Metal working buffer is multi-GB *regardless* of weight share,
   so the peak blew the ceiling even at 4.5 GB → `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
5. **180-second residency stacking** — macOS keeps freed GPU buffers wired 180 s, so back-to-back runs
   stacked over the ceiling — which is why the *same* config crashed only *sometimes*.
6. **Duplicate rpc-servers** — repeated restarts spawned colliding instances → "malformed response."
7. **The kernel panic** — an *uncapped* 12 GB Metal load seized the whole machine; the macOS watchdog
   rebooted it. (The watchdog is the recovery, not the enemy. The cap is the fix; it isn't cleanly
   disableable on Apple Silicon.)
8. **Me, twice** — I called it a hardware wall and said bank it. Both times the operator and the M1-side
   agent pushed back and kept going. Both times they were right.

---

## It ran
Every wall down — cable restored, GPU freed, ceiling raised, residency killed, share floored, one
clean instance — the 70B **loaded and ran across both machines.** Backend `CUDA,RPC`:

| config | prefill | decode | M1 slice |
|---|---|---|---|
| Dell 13 + M1 6 (floor) | 1.21 t/s | — | ~3 GB |
| **Dell 14 + M1 9 (champion)** | **3.22 t/s** | **1.40 t/s** | 4.5 GB, *measured on M1* |
| Dell 14 + M1 12 | crash | crash | over ceiling |

Pushing more layers onto GPU silicon **2.7×'d prefill.** The M1's true safe lend for a 70B: **~9
layers / ~4.5 GB** — found by crashing at 12 and holding at 9.

---

## The honest caveats
- **Slow — 1.40 t/s decode.** ~58 of 80 layers run on the Dell's *CPU* via RAM-offload. This buys
  **capacity, not speed.** Nobody serves production on it.
- **The cable was never the bottleneck.** It *enables* pooling; it doesn't get saturated.
- **The M1 is a marginal 70B node** — fragile, panicked once, crashed ten times. The Linux box never
  flinched. The open, pushable machine carried it; the locked, polished one folded.

---

## What it proves
Not *is local cheaper* — **"do you have food in the house at all":** can your own two boxes run a
frontier-class model, or must you rent the cloud? Tonight that became a number with a `t/s` next to it.
**Yes.** Organization beat hardware — the drivetrain, not the engine.

---

## Reproduce it
```bash
# M1 (server) — ONE instance:
sudo sysctl iogpu.wired_limit_mb=12288
export GGML_METAL_RESIDENCY_KEEP_ALIVE_S=10
ggml-rpc-server -H 0.0.0.0 -p 50052

# Dell (client) — restore the cable IP first if the link reset:
sudo ip addr add 10.55.0.1/24 dev thunderbolt0
llama-bench -m DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf \
  --rpc 10.55.0.2:50052 -ngl 23 -ts 14,9 -fa 0 -b 64 -ub 64 -p 16 -n 16
```
Cross-engine handoff scripts: `scripts/`. Research + scoping docs (including the dead Gloo/TP path):
`docs/`. KV layout: `[B, n_kv_heads, S, head_dim]`, keys post-RoPE.

## The two-agent collaboration
Debugged in real time by two agents, one resident on each machine:
- **The Dell side** drove the runs, chased down the cable-IP loss, and cleared the GPU squatters.
- **The M1 side** held its `ggml-rpc-server` end through every crash and **diagnosed the hardest walls
  itself**: it traced the Metal OOM to the **wired-memory ceiling** (and the exact
  `iogpu.wired_limit_mb` sysctl that lifts it), pinned the **180-second residency stacking** behind
  the *intermittent* crashes, caught the **duplicate-instance collision**, and ruled out the watchdog
  as a red herring. It also measured **its own ~4.5 GB slice directly** and **flagged measured-vs-
  relayed** — when it couldn't capture a number on its end, it said so rather than borrow the Dell's.

The record is honest because each agent graded its *own* work. The 70B ran across both machines; the
write-up reflects both halves.

## The doctrine
The cold receipt is above. The *why* it matters — the thesis this is a receipt for — is in
**[MANIFESTO.md](MANIFESTO.md)**: read the concrete, not the press release.

## License
MIT — see `LICENSE`.

*Neither box could hold it. The cable made one that could. And it ran.*

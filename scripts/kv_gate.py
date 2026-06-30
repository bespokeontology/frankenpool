"""M1 gate, rigorous. Two metrics:
 (1) free-gen argmax match (path-dependent, shows greedy divergence),
 (2) teacher-forced per-step logit cosine (feeds the SAME tokens to both caches -> isolates
     whether the injected HF KV is FAITHFUL, independent of greedy near-tie flips)."""
import json, numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import load_file

META = "/Users/bespokeontology/kv_meta.json"; BLOB = "/Users/bespokeontology/kv_blob.safetensors"
meta = json.load(open(META)); blob = load_file(BLOB)
model, tok = load(meta["repo"]); nL = meta["n_layers"]
ctx = mx.array([meta["ctx_ids"]]); first = meta["next_id"]; N = 24

def new_golden():
    c = make_prompt_cache(model); model(ctx, cache=c); return c
def new_imported(dt):
    c = make_prompt_cache(model)
    for i in range(nL):
        k = mx.array(blob[f"k_{i}"]).astype(dt)[None]; v = mx.array(blob[f"v_{i}"]).astype(dt)[None]
        try: c[i].state = (k, v)
        except Exception: c[i].keys, c[i].values = k, v
        try: c[i].offset = int(k.shape[2])
        except Exception: pass
    return c
def gen(c, first_tok, n=N):
    out, cur = [], mx.array([[first_tok]])
    for _ in range(n):
        lg = model(cur, cache=c); nt = int(mx.argmax(lg[0, -1]).item()); out.append(nt); cur = mx.array([[nt]])
    return out

dt = new_golden()[0].keys.dtype
golden = gen(new_golden(), first)
imported = gen(new_imported(dt), first)

# teacher-forced KV faithfulness: feed identical tokens to both, compare logits step-by-step
seq = [first] + golden[:-1]
gc, ic = new_golden(), new_imported(dt); coss = []; agree = 0
for t in seq:
    g = np.array(model(mx.array([[t]]), cache=gc)[0, -1].astype(mx.float32))
    i = np.array(model(mx.array([[t]]), cache=ic)[0, -1].astype(mx.float32))
    coss.append(float(g @ i / (np.linalg.norm(g) * np.linalg.norm(i)))); agree += int(np.argmax(g) == np.argmax(i))

print("ctx tokens:", meta["seq_len"], "| dtype:", dt)
print("GOLDEN  :", repr(tok.decode(golden)))
print("IMPORTED:", repr(tok.decode(imported)))
print(f"free-gen argmax match: {sum(a == b for a, b in zip(golden, imported))}/{N}  (path-dependent)")
print(f"teacher-forced logit cos: mean={np.mean(coss):.6f} min={np.min(coss):.6f} | argmax agree {agree}/{len(seq)}")
print("VERDICT:", "KV FAITHFUL — divergence is benign near-tie noise"
      if np.mean(coss) > 0.999 and agree >= len(seq) - 2 else "KV SUSPECT — real structural mismatch")

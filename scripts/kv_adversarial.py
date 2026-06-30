"""Falsification test: is MLX actually USING the injected KV, or ignoring it?
Inject (a) the real HF KV, (b) zeroed, (c) position-shuffled, (d) noise-corrupted.
If MLX ignores the cache, all four match the golden -> the handoff 'proof' is a sham.
If only the real KV reproduces the golden continuation, the cache genuinely drives decode."""
import json, numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import load_file

META = "/Users/bespokeontology/kv_meta.json"; BLOB = "/Users/bespokeontology/kv_blob.safetensors"
meta = json.load(open(META)); blob = load_file(BLOB)
model, tok = load(meta["repo"]); nL = meta["n_layers"]
ctx = mx.array([meta["ctx_ids"]]); first = meta["next_id"]; N = 20

def golden_cache():
    c = make_prompt_cache(model); model(ctx, cache=c); return c
dt = golden_cache()[0].keys.dtype

def inject(mode):
    rs = np.random.RandomState(0); c = make_prompt_cache(model)
    for i in range(nL):
        k = np.array(blob[f"k_{i}"]).astype(np.float32); v = np.array(blob[f"v_{i}"]).astype(np.float32)
        if mode == "zero":
            k[:] = 0; v[:] = 0
        elif mode == "shuffle_pos":
            p = rs.permutation(k.shape[1]); k = k[:, p]; v = v[:, p]
        elif mode == "noise":
            k = k + 2.0 * rs.standard_normal(k.shape).astype(np.float32)
            v = v + 2.0 * rs.standard_normal(v.shape).astype(np.float32)
        km = mx.array(k).astype(dt)[None]; vm = mx.array(v).astype(dt)[None]
        try: c[i].state = (km, vm)
        except Exception: c[i].keys, c[i].values = km, vm
        try: c[i].offset = int(km.shape[2])
        except Exception: pass
    return c

def gen(c, n=N):
    out, cur = [], mx.array([[first]])
    for _ in range(n):
        lg = model(cur, cache=c); t = int(mx.argmax(lg[0, -1]).item()); out.append(t); cur = mx.array([[t]])
    return tok.decode(out)

print("GOLDEN (pure MLX)   :", repr(gen(golden_cache())))
print("REAL HF KV injected :", repr(gen(inject("real"))))
print("ZEROED KV           :", repr(gen(inject("zero"))))
print("SHUFFLED-POSITION KV:", repr(gen(inject("shuffle_pos"))))
print("NOISE-CORRUPTED KV  :", repr(gen(inject("noise"))))

"""M1 side (REVERSE direction): MLX prefill -> export the KV cache + meta, to hand to the Dell's HF decode."""
import json, numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"
OUT_ST, OUT_META = "/Users/bespokeontology/kv_blob_mlx.safetensors", "/Users/bespokeontology/kv_meta_mlx.json"

model, tok = load(REPO)
ids = tok.encode(PROMPT); N = len(ids)
ctx = mx.array([ids[:N - 1]]); nxt = ids[N - 1]

c = make_prompt_cache(model); model(ctx, cache=c); nL = len(c)
S = int(c[0].offset)   # MLX pre-allocates the KV buffer in 256-token steps; slice to the REAL token count
snap = [(np.array(c[i].keys[:, :, :S, :].astype(mx.float32))[0].astype(np.float32),   # [n_kv_heads, S, head_dim]
         np.array(c[i].values[:, :, :S, :].astype(mx.float32))[0].astype(np.float32)) for i in range(nL)]
nlog = model(mx.array([[nxt]]), cache=c); mlx_pred = int(mx.argmax(nlog[0, -1]).item())

tensors = {}
for i, (k, v) in enumerate(snap):
    tensors[f"k_{i}"] = k; tensors[f"v_{i}"] = v
save_file(tensors, OUT_ST)
meta = {"repo": REPO, "prompt": PROMPT, "ctx_ids": list(ids[:N - 1]), "next_id": int(nxt),
        "n_layers": nL, "seq_len": int(snap[0][0].shape[1]),
        "mlx_next_pred": mlx_pred, "mlx_next_pred_str": tok.decode([mlx_pred])}
json.dump(meta, open(OUT_META, "w"), indent=2)
print("MLX EXPORT OK:", json.dumps({k: meta[k] for k in ["seq_len", "n_layers", "mlx_next_pred_str"]}))

"""Dell side (REVERSE): inject the M1's MLX-computed KV into HuggingFace and decode.
Golden = pure-HF prefill+decode; Imported = decode from the MLX KV. Match == reverse handoff works."""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.numpy import load_file

META, BLOB = "/home/bespoke-ontology/kv_meta_mlx.json", "/home/bespoke-ontology/kv_blob_mlx.safetensors"
meta = json.load(open(META)); blob = load_file(BLOB)
tok = AutoTokenizer.from_pretrained(meta["repo"])
model = AutoModelForCausalLM.from_pretrained(meta["repo"], dtype=torch.float32).eval()
nL, N_GEN = meta["n_layers"], 24
ctx = torch.tensor([meta["ctx_ids"]]); first = meta["next_id"]

def gen_from(pkv):
    toks, cur = [], torch.tensor([[first]])
    with torch.no_grad():
        for _ in range(N_GEN):
            o = model(cur, past_key_values=pkv, use_cache=True); pkv = o.past_key_values
            t = int(o.logits[0, -1].argmax()); toks.append(t); cur = torch.tensor([[t]])
    return toks

def build_cache():
    """Inject MLX KV into an HF cache, robust across transformers versions."""
    ks = [torch.tensor(blob[f"k_{i}"])[None].to(torch.float32) for i in range(nL)]  # [1, n_kv_heads, S, head_dim]
    vs = [torch.tensor(blob[f"v_{i}"])[None].to(torch.float32) for i in range(nL)]
    legacy = tuple((ks[i], vs[i]) for i in range(nL))
    try:
        from transformers import DynamicCache
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy)
        dc = DynamicCache()
        for i in range(nL):
            dc.update(ks[i], vs[i], i)
        return dc
    except Exception as e:
        print("cache build fallback (legacy tuple):", e)
        return legacy

with torch.no_grad():
    golden = gen_from(model(ctx, use_cache=True).past_key_values)
imported = gen_from(build_cache())

print("MLX pred (meta) :", repr(meta.get("mlx_next_pred_str")))
print("GOLDEN (pure HF):", repr(tok.decode(golden)))
print("MLX->HF imported:", repr(tok.decode(imported)))
m = sum(a == b for a, b in zip(golden, imported))
print(f"REVERSE HANDOFF Mac->Dell: {m}/{N_GEN} match -> {'PASS' if m >= N_GEN - 1 else 'CHECK'}")

"""Dell side: HF prefill -> export per-layer KV cache (fp32) + metadata.
Prefill ctx = prompt[0:N-1]; the token at N-1 is what the M1 will decode from the imported cache."""
import torch, json, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.numpy import save_file

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Here is a long list of simple facts. " + " ".join(
    f"The number that comes after {i} is {i + 1}, and {i} plus one equals {i + 1}." for i in range(1, 90)
) + " Continuing the pattern, the number that comes after 90 is"
OUT_ST, OUT_META = "/home/bespoke-ontology/kv_blob.safetensors", "/home/bespoke-ontology/kv_meta.json"

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()
cfg = model.config

ids = tok(PROMPT, return_tensors="pt").input_ids       # [1, N]
N = ids.shape[1]; assert N >= 2
ctx, nxt = ids[:, :N-1], ids[:, N-1:N]                  # prefill[0:N-1], decode token @ N-1

def extract_kv(pkv):
    """Robust across transformers cache formats -> list of (K,V) per layer, each [B, n_kv_heads, S, head_dim]."""
    if hasattr(pkv, "to_legacy_cache"):
        try:
            leg = pkv.to_legacy_cache()
            if leg and leg[0] is not None:
                return [(k, v) for (k, v) in leg]
        except Exception:
            pass
    if hasattr(pkv, "layers") and pkv.layers and hasattr(pkv.layers[0], "keys"):
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return [(k, v) for (k, v) in pkv]

with torch.no_grad():
    out = model(ctx, use_cache=True)
    pkv = out.past_key_values
    snap = [(k.clone(), v.clone()) for (k, v) in extract_kv(pkv)]  # snapshot BEFORE the next forward mutates pkv
    out2 = model(nxt, past_key_values=pkv, use_cache=True)         # HF's own next-token prediction
    hf_pred = int(out2.logits[0, -1].argmax())

tensors = {}
for i, (k, v) in enumerate(snap):                       # k,v: [1, n_kv_heads, S, head_dim]
    tensors[f"k_{i}"] = k[0].float().numpy().astype(np.float32)
    tensors[f"v_{i}"] = v[0].float().numpy().astype(np.float32)
save_file(tensors, OUT_ST)

meta = {
    "repo": REPO, "prompt": PROMPT,
    "ctx_ids": ids[0, :N-1].tolist(), "next_id": int(ids[0, N-1]),
    "n_layers": len(snap), "n_kv_heads": int(snap[0][0].shape[1]),
    "head_dim": int(snap[0][0].shape[3]), "seq_len": int(snap[0][0].shape[2]),
    "rope_theta": getattr(cfg, "rope_theta", None), "rope_scaling": getattr(cfg, "rope_scaling", None),
    "hf_next_pred": hf_pred, "hf_next_pred_str": tok.decode([hf_pred]), "kv_dtype": "float32",
}
json.dump(meta, open(OUT_META, "w"), indent=2)
print("EXPORT OK:", json.dumps(meta, indent=2))

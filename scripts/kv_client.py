"""Dell client: HF prefill -> stream the KV cache over the Thunderbolt cable to the M1 server,
measuring the link while the conditional matrix physically crosses it."""
import socket, struct, json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.numpy import save as st_save

M1, PORT, REPO, IFACE = "10.55.0.2", 51000, "Qwen/Qwen2.5-0.5B-Instruct", "thunderbolt0"
PROMPT = "Here is a long list of simple facts. " + " ".join(
    f"The number that comes after {i} is {i + 1}, and {i} plus one equals {i + 1}." for i in range(1, 90)
) + " Continuing the pattern, the number that comes after 90 is"

def tx_bytes(): return int(open(f"/sys/class/net/{IFACE}/statistics/tx_bytes").read())
def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(min(n - len(buf), 1 << 20))
        if not c: raise ConnectionError("closed")
        buf += c
    return bytes(buf)
def recv_msg(s): return recvall(s, struct.unpack(">Q", recvall(s, 8))[0])
def send_msg(s, b): s.sendall(struct.pack(">Q", len(b)) + b)
def extract_kv(pkv):
    if hasattr(pkv, "to_legacy_cache"):
        try:
            leg = pkv.to_legacy_cache()
            if leg and leg[0] is not None: return [(k, v) for (k, v) in leg]
        except Exception: pass
    if hasattr(pkv, "layers") and pkv.layers and hasattr(pkv.layers[0], "keys"):
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"): return list(zip(pkv.key_cache, pkv.value_cache))
    return [(k, v) for (k, v) in pkv]

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
ids = tok(PROMPT, return_tensors="pt").input_ids; N = ids.shape[1]
print(f"prefilling {N-1} tokens on the Dell...", flush=True)
with torch.no_grad():
    snap = [(k.clone(), v.clone()) for (k, v) in extract_kv(model(ids[:, :N - 1], use_cache=True).past_key_values)]

tensors = {f"k_{i}": k[0].float().numpy().astype(np.float32) for i, (k, v) in enumerate(snap)}
tensors.update({f"v_{i}": v[0].float().numpy().astype(np.float32) for i, (k, v) in enumerate(snap)})
kv_bytes = st_save(tensors)
meta = json.dumps({"n_layers": len(snap), "next_id": int(ids[0, N - 1]), "seq_len": int(snap[0][0].shape[2])}).encode()
print(f"KV serialized: {len(kv_bytes)/1e6:.1f} MB for {snap[0][0].shape[2]} tokens; connecting to M1 over the cable...", flush=True)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.connect((M1, PORT))
send_msg(s, meta)
b0, t0 = tx_bytes(), time.time()
send_msg(s, kv_bytes)
t_send, b_send = time.time(), tx_bytes()        # client send window (buffer-optimistic)
result = json.loads(recv_msg(s).decode())       # NOTE: this waits for the M1's decode -> NOT transfer time
wall, b1 = time.time() - t0, tx_bytes()
s.close()

wire_mb = (b1 - b0) / 1e6                        # bytes actually on thunderbolt0 (authoritative byte count)
recv_s = result.get("recv_s") or (t_send - t0)  # M1 first->last byte = authoritative transfer TIME
print("=" * 60)
print("RESULT from M1:", repr(result.get("text", result)))
print(f"KV on the wire: {wire_mb:.1f} MB | M1 received in {recv_s:.4f}s = {wire_mb*8/recv_s:.0f} Mbps ({wire_mb/recv_s:.0f} MB/s)  <- authoritative")
print(f"(wall round-trip {wall:.3f}s incl. M1 decode {result.get('decode_s')}s; ctx {result.get('ctx')} tokens)")

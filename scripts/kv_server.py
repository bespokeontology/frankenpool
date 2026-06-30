"""M1 server: hold the MLX model, listen on the Thunderbolt cable, receive a streamed KV cache,
inject it, decode, return the text. The KV crosses the wire instead of a file."""
import socket, struct, json, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import load as st_load

HOST, PORT, REPO, N_GEN = "10.55.0.2", 51000, "Qwen/Qwen2.5-0.5B-Instruct", 24

def recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(min(n - len(buf), 1 << 20))
        if not c: raise ConnectionError("peer closed")
        buf += c
    return bytes(buf)
def recv_msg(sock): return recvall(sock, struct.unpack(">Q", recvall(sock, 8))[0])
def send_msg(sock, b): sock.sendall(struct.pack(">Q", len(b)) + b)

print("loading MLX model...", flush=True)
model, tok = load(REPO)
_c = make_prompt_cache(model); model(mx.array([[0]]), cache=_c); DT = _c[0].keys.dtype   # learn compute dtype
print(f"ready (dtype={DT}); listening on {HOST}:{PORT}", flush=True)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT)); srv.listen(1)

while True:
    conn, addr = srv.accept()
    try:
        meta = json.loads(recv_msg(conn).decode())
        t0 = time.time(); kv_bytes = recv_msg(conn); t_recv = time.time() - t0
        blob = st_load(kv_bytes); nL = meta["n_layers"]; first = meta["next_id"]
        cache = make_prompt_cache(model)
        for i in range(nL):
            k = mx.array(blob[f"k_{i}"]).astype(DT)[None]; v = mx.array(blob[f"v_{i}"]).astype(DT)[None]
            try: cache[i].state = (k, v)
            except Exception: cache[i].keys, cache[i].values = k, v
            try: cache[i].offset = int(k.shape[2])
            except Exception: pass
        t1 = time.time(); out, cur = [], mx.array([[first]])
        for _ in range(N_GEN):
            lg = model(cur, cache=cache); t = int(mx.argmax(lg[0, -1]).item()); out.append(t); cur = mx.array([[t]])
        result = {"text": tok.decode(out), "recv_bytes": len(kv_bytes), "recv_s": round(t_recv, 4),
                  "decode_s": round(time.time() - t1, 4), "ctx": meta["seq_len"], "n_layers": nL}
        send_msg(conn, json.dumps(result).encode())
        print(f"served: ctx={meta['seq_len']} got {len(kv_bytes)/1e6:.1f}MB in {t_recv:.3f}s -> {result['text'][:50]!r}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        try: send_msg(conn, json.dumps({"error": str(e)}).encode())
        except Exception: pass
    finally:
        conn.close()

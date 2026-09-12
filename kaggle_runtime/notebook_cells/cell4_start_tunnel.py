import os, sys, socket

sys.path.insert(0, "/kaggle/working")

# NOT 8000 — something in the Kaggle container already listens there, and
# uvicorn dies with "address already in use" while ngrok happily tunnels to
# whatever else is on that port.
PORT = 8765

os.environ["MEDGEMMA_MODEL"] = "google/medgemma-1.5-4b-it"
# "bf16" | "fp32" | "8bit" | "4bit". bf16 is ~8.6GB and fits a 16GB card.
# NOT "fp16": this model's activations overflow float16's exponent range and
# it generates empty strings with no error. Use "fp32" (~17GB, needs T4 x2)
# if bf16 misbehaves, or "8bit"/"4bit" if the load OOMs.
os.environ["MEDGEMMA_PRECISION"] = "bf16"
# Paste the raw token. No angle brackets — they are placeholder markers here,
# and ngrok rejects them with ERR_NGROK_105.
os.environ["NGROK_AUTHTOKEN"] = "PASTE_YOUR_NGROK_TOKEN_HERE"

token = os.environ["NGROK_AUTHTOKEN"]
if token.startswith("<") or "PASTE_YOUR" in token:
    raise SystemExit("Set NGROK_AUTHTOKEN to your real token, without < >.")

with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", PORT)) == 0:
        raise SystemExit(f"Port {PORT} is in use. Pick another PORT and re-run.")

from pyngrok import ngrok

ngrok.kill()  # clear any half-open session left by a failed earlier run
ngrok.set_auth_token(token)

from medgemma_server import start_in_background

# Loads the checkpoint FIRST, then opens the tunnel, so the URL only appears
# once the model can actually serve. Several minutes on a cold cache;
# near-instant if this kernel already loaded it.
ngrok_url = start_in_background(port=PORT)
print("MEDGEMMA_TUNNEL_URL =", ngrok_url)
print("precision =", os.environ["MEDGEMMA_PRECISION"])

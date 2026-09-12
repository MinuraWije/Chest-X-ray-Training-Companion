import os

from huggingface_hub import login, whoami

# Paste the raw token from https://huggingface.co/settings/tokens (read access
# is enough). The interactive login() widget is unreliable inside Kaggle, and a
# blank token surfaces later as the very unhelpful "Illegal header value
# b'Bearer '" when the model download tries to authenticate.
HF_TOKEN = "PASTE_YOUR_HF_TOKEN_HERE"

if not HF_TOKEN.strip() or "PASTE_YOUR" in HF_TOKEN:
    raise SystemExit("Set HF_TOKEN to a real Hugging Face token.")

login(token=HF_TOKEN)
# Belt and braces: some libraries read the env var rather than the stored login.
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

print("Logged in as:", whoami()["name"])

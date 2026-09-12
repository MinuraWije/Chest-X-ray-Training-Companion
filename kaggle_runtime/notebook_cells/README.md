# One file per Kaggle notebook cell

Each `.py` here is the **entire contents of one notebook cell** — copy the whole
file, paste it into an empty cell, run it. Nothing to trim, no prose to strip.

| File | Cell | Notes |
|---|---|---|
| `cell1_install.py` | 1 | Restart the kernel afterwards. |
| `cell2_write_server.py` | 2 | `%%writefile` must stay the first line. |
| `cell3_hf_login.py` | 3 | Needs a HF token with access to the gated model. |
| `cell4_start_tunnel.py` | 4 | Put your ngrok authtoken in. Prints the URL. |
| `cell5_keep_alive.py` | 5 | Leave running; the tunnel dies with the process. |

The prose explanation of each step lives in `../NOTEBOOK_CELLS.md`. These files
are generated from it, so edit that document rather than these.

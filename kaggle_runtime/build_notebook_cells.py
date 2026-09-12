"""Regenerate the paste-ready notebook cells from their real sources.

Cell 2 is `%%writefile` followed by the whole of `medgemma_server.py`, and
`NOTEBOOK_CELLS.md` embeds every cell again inside fenced blocks. That is three
copies of the same code, and a stale copy is invisible until the container is
running the wrong thing — the failure that `SERVER_VERSION` exists to detect.

So the copies are generated, never hand-edited. Edit `medgemma_server.py` or a
`notebook_cells/cellN_*.py`, then run:

    python kaggle_runtime/build_notebook_cells.py

`--check` exits non-zero if anything is out of date, for use in CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE / "medgemma_server.py"
CELLS_DIR = HERE / "notebook_cells"
DOC = HERE / "NOTEBOOK_CELLS.md"

WRITEFILE_HEADER = "%%writefile /kaggle/working/medgemma_server.py\n"

# Heading text -> the cell file whose contents belong in that section's fence.
CELL_FILES: dict[str, str] = {
    "## Cell 1 — dependencies": "cell1_install.py",
    "## Cell 2 — write the server into the container": "cell2_write_server.py",
    "## Cell 3 — Hugging Face login": "cell3_hf_login.py",
    "## Cell 4 — start the server and open the tunnel": "cell4_start_tunnel.py",
    "## Cell 5 — keep the session alive": "cell5_keep_alive.py",
}


def render_cell2() -> str:
    return WRITEFILE_HEADER + SERVER.read_text(encoding="utf-8")


def render_doc(cells: dict[str, str]) -> str:
    """Replace the first fenced python block after each cell heading."""
    lines = DOC.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        out.append(line)
        heading = line.rstrip("\n")
        if heading not in CELL_FILES:
            index += 1
            continue

        # Copy through to the opening fence, then swap the body wholesale.
        index += 1
        while index < len(lines) and not lines[index].startswith("```python"):
            out.append(lines[index])
            index += 1
        if index >= len(lines):
            raise SystemExit(f"No ```python fence found under {heading!r}.")
        out.append(lines[index])
        index += 1
        while index < len(lines) and not lines[index].startswith("```"):
            index += 1
        body = cells[CELL_FILES[heading]]
        out.append(body if body.endswith("\n") else body + "\n")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero instead of rewriting",
    )
    args = parser.parse_args()

    wanted: dict[str, str] = {"cell2_write_server.py": render_cell2()}
    cells = {
        name: wanted.get(name) or (CELLS_DIR / name).read_text(encoding="utf-8")
        for name in CELL_FILES.values()
    }
    wanted[str(DOC)] = render_doc(cells)

    stale: list[str] = []
    for name, content in wanted.items():
        path = Path(name) if name.startswith(str(HERE)) else CELLS_DIR / name
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        stale.append(str(path.relative_to(HERE.parent)))
        if not args.check:
            path.write_text(content, encoding="utf-8")

    if args.check:
        if stale:
            print("Out of date (run build_notebook_cells.py):")
            for name in stale:
                print(f"  {name}")
            return 1
        print("Notebook cells are up to date.")
        return 0

    print("Regenerated:" if stale else "Already up to date.")
    for name in stale:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

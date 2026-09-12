"""Cross-platform dev runner (Windows + macOS + Linux).

Replaces the Makefile for users who don't have `make` installed (most
Windows boxes). Targets are the same as the Makefile; pass them as
subcommands:

    python scripts/dev.py install-dev
    python scripts/dev.py data
    python scripts/dev.py run
    python scripts/dev.py test
    python scripts/dev.py lint
    python scripts/dev.py format
    python scripts/dev.py seed
    python scripts/dev.py eval
    python scripts/dev.py clean
    python scripts/dev.py all         # install-dev + data + test + lint
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV = REPO_ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_python() -> str:
    if VENV_PY.exists():
        return str(VENV_PY)
    if shutil.which("python"):
        return "python"
    return sys.executable


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else str(REPO_ROOT))
    return proc.returncode


def _pip_install(args: list[str]) -> int:
    py = _venv_python()
    return _run([py, "-m", "pip", "install", "--upgrade", "pip", *args])


def _venv_exists() -> bool:
    return VENV_PY.exists()


def _ensure_venv() -> int:
    if _venv_exists():
        return 0
    py = sys.executable
    return _run([py, "-m", "venv", str(VENV)])


def cmd_install(_: argparse.Namespace) -> int:
    rc = _ensure_venv()
    if rc != 0:
        return rc
    return _pip_install(["-r", "requirements.txt", "-e", "."])


def cmd_install_dev(_: argparse.Namespace) -> int:
    rc = _ensure_venv()
    if rc != 0:
        return rc
    return _pip_install(["-r", "requirements-dev.txt", "-e", "."])


def cmd_test(_: argparse.Namespace) -> int:
    return _run([_venv_python(), "-m", "pytest", "-q"])


def cmd_lint(_: argparse.Namespace) -> int:
    py = _venv_python()
    rc = _run([py, "-m", "ruff", "check", "src", "tests", "scripts"])
    if rc != 0:
        return rc
    return _run([py, "-m", "mypy", "src", "scripts"])


def cmd_format(_: argparse.Namespace) -> int:
    py = _venv_python()
    rc = _run([py, "-m", "ruff", "check", "--fix", "src", "tests", "scripts"])
    if rc != 0:
        return rc
    return _run([py, "-m", "black", "src", "tests", "scripts"])


def cmd_run(args: argparse.Namespace) -> int:
    host = args.host
    port = args.port
    processed_dir = REPO_ROOT.parent / "data" / "processed"
    return _run(
        [
            _venv_python(),
            "-m",
            "uvicorn",
            "cxt.interfaces.api.app:app",
            "--host",
            host,
            "--port",
            str(port),
            "--reload",
            "--reload-dir",
            str(REPO_ROOT / "src"),
            "--reload-dir",
            str(processed_dir),
            "--reload-include",
            "*.csv",
        ]
    )


def cmd_data(args: argparse.Namespace) -> int:
    py = _venv_python()
    download_cmd: list[str] = [
        py,
        "-m",
        "scripts.download_nih_subset",
        "--source",
        args.source,
    ]
    if args.source == "stub":
        download_cmd += ["--n-per-finding", str(args.n_per_finding)]
    elif args.source == "local":
        if args.local_dir is None:
            print("ERROR: --local-dir is required when --source=local", file=sys.stderr)
            return 2
        download_cmd += [
            "--local-dir",
            str(args.local_dir),
            "--n-per-finding",
            str(args.n_per_finding),
        ]
        if args.metadata_csv is not None:
            download_cmd += ["--metadata-csv", str(args.metadata_csv)]
    else:
        print(f"ERROR: unsupported --source={args.source}", file=sys.stderr)
        return 2

    rc = _run(download_cmd)
    if rc != 0:
        return rc
    return _run(
        [
            py,
            "-m",
            "scripts.build_manifest",
            "--n-per-finding",
            str(args.n_per_finding),
            "--n-negatives",
            str(args.n_per_finding),
        ]
    )


# Where we look for the bundled NIH subset by default (sibling of the
# repo root, named `Dataset/`). The "real-data" target is what the
# supervisor-review workflow uses — a single command that wipes
# data/raw + data/processed and rebuilds the 300-case real-data manifest
# from this folder.
DEFAULT_REAL_DATA_DIR: Final[Path] = REPO_ROOT.parent / "Dataset" / "images"
DEFAULT_REAL_DATA_CSV: Final[Path] = REPO_ROOT.parent / "Dataset" / "Data_Entry_2017_v2020.csv"


def cmd_real_data(args: argparse.Namespace) -> int:
    """One-shot real-data setup.

    Wipes ``data/raw`` and ``data/processed``, runs
    ``download_nih_subset --source local`` against the bundled NIH
    subset, then ``build_manifest``. Default paths are
    ``..\\Dataset\\images`` and ``..\\Dataset\\Data_Entry_2017_v2020.csv``
    (override with ``--local-dir`` / ``--metadata-csv``).
    """
    local_dir = args.local_dir or DEFAULT_REAL_DATA_DIR
    metadata_csv = args.metadata_csv or DEFAULT_REAL_DATA_CSV

    missing: list[Path] = []
    if not local_dir.is_dir():
        missing.append(local_dir)
    if not metadata_csv.is_file():
        missing.append(metadata_csv)
    if missing:
        print(
            "ERROR: real-data target needs the bundled NIH subset:",
            file=sys.stderr,
        )
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nExpected layout (relative to the repo root):",
            file=sys.stderr,
        )
        print("  Dataset/images/00000001_000.png  ...", file=sys.stderr)
        print("  Dataset/Data_Entry_2017_v2020.csv", file=sys.stderr)
        print(
            "\nOverride with --local-dir <path> --metadata-csv <path> if you "
            "unzipped the NIH box download to a different location.",
            file=sys.stderr,
        )
        return 2

    raw_dir = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"
    for target in (raw_dir, processed_dir):
        if target.exists():
            print(f"Removing {target}")
            shutil.rmtree(target, ignore_errors=True)

    print(
        f"\nBuilding real-data manifest from {local_dir} + {metadata_csv} "
        f"({args.n_per_finding} per finding + {args.n_per_finding} negatives)."
    )
    args.local_dir = local_dir
    args.metadata_csv = metadata_csv
    args.source = "local"
    rc = cmd_data(args)
    if rc == 0:
        print(
            "\nDone. The /images static mount will pick up the new files "
            "automatically (uvicorn --reload watches data/processed). If you "
            "started the backend without --reload, restart it with "
            "python scripts\\dev.py run."
        )
    return rc


def cmd_seed(_: argparse.Namespace) -> int:
    return _run([_venv_python(), "scripts/seed_admin.py"])


def cmd_eval(args: argparse.Namespace) -> int:
    """Forward everything after `eval` to run_eval.py.

    Without the passthrough there is no way to reach `--runtime kaggle_tunnel`
    through this script.
    """
    extra = list(getattr(args, "eval_args", []) or [])
    return _run([_venv_python(), "scripts/run_eval.py", *extra])


def cmd_corpus(args: argparse.Namespace) -> int:
    """Build and validate the Phase 2 retrieval corpus."""
    extra = list(getattr(args, "corpus_args", []) or [])
    return _run([_venv_python(), "scripts/build_corpus.py", *extra])


def cmd_feedback_eval(args: argparse.Namespace) -> int:
    """Run the Phase 2 feedback evaluation across the five arms."""
    extra = list(getattr(args, "feedback_eval_args", []) or [])
    return _run([_venv_python(), "scripts/run_feedback_eval.py", *extra])


def cmd_labels(args: argparse.Namespace) -> int:
    """Export relevance-labelling pairs, or score the proxy against them."""
    extra = list(getattr(args, "labels_args", []) or [])
    return _run([_venv_python(), "scripts/label_relevance.py", *extra])


def cmd_clean(_: argparse.Namespace) -> int:
    targets = [".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage", "htmlcov", "cxt.db", "data"]
    for name in targets:
        path = REPO_ROOT / name
        if path.exists():
            print(f"Removing {path}")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
    for cache in REPO_ROOT.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    # cmd_data reads args.source and friends, so it needs a fully-populated
    # namespace rather than a bare one.
    data_args = argparse.Namespace(
        source=getattr(args, "source", "stub"),
        n_per_finding=getattr(args, "n_per_finding", 5),
        local_dir=getattr(args, "local_dir", None),
        metadata_csv=getattr(args, "metadata_csv", None),
    )
    steps: list[tuple[Callable[[argparse.Namespace], int], argparse.Namespace]] = [
        (cmd_install_dev, argparse.Namespace()),
        (cmd_data, data_args),
        (cmd_seed, argparse.Namespace()),
        (cmd_test, argparse.Namespace()),
        (cmd_lint, argparse.Namespace()),
    ]
    for step, step_args in steps:
        rc = step(step_args)
        if rc != 0:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_text in (
        ("install", cmd_install, "Create .venv and install runtime deps"),
        ("install-dev", cmd_install_dev, "Create .venv, install dev deps, and pip install -e ."),
        ("test", cmd_test, "Run pytest"),
        ("lint", cmd_lint, "Run ruff + mypy"),
        ("format", cmd_format, "Run ruff --fix + black"),
        ("seed", cmd_seed, "Seed the first admin user (scripts/seed_admin.py)"),
        ("clean", cmd_clean, "Remove caches and the local data dir"),
        ("all", cmd_all, "install-dev + data + test + lint"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.set_defaults(func=fn)

    # `eval` takes passthrough args so `--runtime kaggle_tunnel` is reachable.
    eval_parser = sub.add_parser(
        "eval",
        help="Run scripts/run_eval.py (extra args are forwarded, e.g. --runtime kaggle_tunnel)",
    )
    eval_parser.set_defaults(func=cmd_eval)

    # Phase 2. All three take passthrough args for the same reason `eval` does.
    corpus_parser = sub.add_parser(
        "corpus",
        help="Run scripts/build_corpus.py (forwards --check / --stats)",
    )
    corpus_parser.set_defaults(func=cmd_corpus)

    feedback_parser = sub.add_parser(
        "feedback-eval",
        help="Run scripts/run_feedback_eval.py (forwards --max-cases / --arms / --runtime)",
    )
    feedback_parser.set_defaults(func=cmd_feedback_eval)

    labels_parser = sub.add_parser(
        "labels",
        help="Run scripts/label_relevance.py (needs `export` or `validate`)",
    )
    labels_parser.set_defaults(func=cmd_labels)

    real_data = sub.add_parser(
        "real-data",
        help=(
            "One-shot: build the real NIH data manifest "
            "(default: ..\\Dataset\\images + ..\\Dataset\\Data_Entry_2017_v2020.csv)."
        ),
    )
    real_data.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=f"Override the NIH image directory (default: {DEFAULT_REAL_DATA_DIR}).",
    )
    real_data.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=f"Override the metadata CSV (default: {DEFAULT_REAL_DATA_CSV}).",
    )
    real_data.add_argument(
        "--n-per-finding",
        type=int,
        default=50,
        help="Cases per positive finding (and number of negative cases). Default 50 = 300-case manifest.",
    )
    real_data.set_defaults(func=cmd_real_data)

    run = sub.add_parser("run", help="Run uvicorn (with --host/--port)")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.set_defaults(func=cmd_run)

    data = sub.add_parser(
        "data",
        help="Build a curated case manifest (stub|local) into data/processed/.",
    )
    data.add_argument(
        "--source",
        choices=("stub", "local"),
        required=True,
        help=(
            "REQUIRED. 'stub' = synthetic 1024x1024 placeholders (first-time "
            "setup only). 'local' = use a pre-downloaded NIH copy (--local-dir "
            "+ --metadata-csv). Re-running with the wrong --source silently "
            "overwrites data/processed/manifest.csv and data/processed/images/, "
            "so pick the right one each time."
        ),
    )
    data.add_argument(
        "--n-per-finding",
        type=int,
        default=5,
        help="Cases per positive finding (and number of negative cases).",
    )
    data.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Path to a directory of pre-downloaded NIH images (named <Image Index>.png).",
    )
    data.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Path to Data_Entry_2017.csv (NIH label metadata); used with --source=local.",
    )
    data.set_defaults(func=cmd_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = argv if argv is not None else sys.argv[1:]
    # These subcommands forward their flags to a script, so unknown args are
    # expected for them and only for them — everywhere else a typo should still
    # be an error rather than being silently passed along.
    passthrough = {
        "eval": "eval_args",
        "corpus": "corpus_args",
        "feedback-eval": "feedback_eval_args",
        "labels": "labels_args",
    }
    if raw and raw[0] in passthrough:
        args, extra = parser.parse_known_args(raw)
        setattr(args, passthrough[raw[0]], extra)
    else:
        args = parser.parse_args(raw)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

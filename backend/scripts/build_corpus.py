"""Build and validate the retrieval corpus.

    python scripts/build_corpus.py            # parse, validate, write index.json
    python scripts/build_corpus.py --check    # validate only; non-zero on failure
    python scripts/build_corpus.py --stats    # coverage per finding and aspect

The build is offline and the runtime reads only `index.json`, so PyYAML lives
in `requirements-dev.txt` and the 13 runtime dependencies are unchanged.

`--check` is the CI gate. It **fails** rather than warning when a chunk carries
an unapproved licence, is missing front matter, exceeds the length cap, or tags
a finding that is not one of ours. This repository is public, so an unlicensed
extract is published the moment it is committed; and a mistagged chunk is
invisible to filtered retrieval and shows up in the eval as a corpus coverage
gap rather than as the bug it is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.domain.entities.finding import FindingCategory  # noqa: E402
from cxt.infrastructure.config.settings import get_settings  # noqa: E402
from cxt.infrastructure.retrieval.corpus import (  # noqa: E402
    ALLOWED_LICENCES,
    CORPUS_INDEX_NAME,
    MAX_CHUNK_CHARS,
    Corpus,
    build_corpus,
)
from cxt.shared.errors import ValidationError  # noqa: E402

#: Findings the curated 300-case corpus actually contains. There are no
#: fracture cases at all, so fracture coverage is not a gap worth reporting.
COVERED_FINDINGS: tuple[str, ...] = (
    FindingCategory.PNEUMOTHORAX.value,
    FindingCategory.PLEURAL_EFFUSION.value,
    FindingCategory.CARDIOMEGALY.value,
    FindingCategory.CONSOLIDATION.value,
    FindingCategory.NODULE_OR_MASS.value,
)

#: Below this, a category's retrieval results are drawn from too few chunks for
#: precision@3 to mean anything — every query returns the same passages.
MIN_CHUNKS_PER_FINDING = 3


def _source_hashes(corpus_dir: Path) -> dict[str, str]:
    """SHA-256 per source file, mirroring data/raw/PROVENANCE.json."""
    hashes: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.upper().startswith("LICENCE") or path.name.upper() == "README.MD":
            continue
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _print_stats(corpus: Corpus) -> None:
    print(f"\nCorpus version: {corpus.version}")
    print(f"Chunks: {len(corpus)}")

    aspects = Counter(c.aspect for c in corpus.chunks)
    print("\nBy aspect:")
    for aspect, count in sorted(aspects.items()):
        note = "  (excluded from relevance)" if aspect == "management" else ""
        print(f"  {aspect:20} {count:3}{note}")

    print("\nBy finding:")
    for finding in COVERED_FINDINGS:
        chunks = corpus.for_finding(finding)
        # `management` chunks are on-topic but useless to someone reading a
        # film, so they do not count toward usable coverage.
        usable = [c for c in chunks if c.aspect != "management"]
        flag = "" if len(usable) >= MIN_CHUNKS_PER_FINDING else "  << THIN"
        print(f"  {finding:20} {len(chunks):3} ({len(usable)} usable){flag}")

    sources = Counter(c.source for c in corpus.chunks)
    print("\nBy source:")
    for source, count in sorted(sources.items()):
        print(f"  {count:3}  {source}")

    licences = Counter(c.licence for c in corpus.chunks)
    print("\nBy licence:")
    for licence, count in sorted(licences.items()):
        print(f"  {count:3}  {licence}")

    longest = max(corpus.chunks, key=lambda c: len(c.text), default=None)
    if longest is not None:
        print(
            f"\nLongest chunk: {len(longest.text)}/{MAX_CHUNK_CHARS} chars "
            f"({longest.id})"
        )


def _check_coverage(corpus: Corpus) -> list[str]:
    """Coverage warnings. Thin coverage is a schedule signal, not a build error."""
    warnings: list[str] = []
    for finding in COVERED_FINDINGS:
        usable = [c for c in corpus.for_finding(finding) if c.aspect != "management"]
        if len(usable) < MIN_CHUNKS_PER_FINDING:
            warnings.append(
                f"{finding}: only {len(usable)} usable chunk(s); "
                f"precision@3 is not meaningful below {MIN_CHUNKS_PER_FINDING}."
            )
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate the RAG corpus.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="corpus directory (default: settings.corpus_dir)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; do not write index.json. Non-zero exit on failure.",
    )
    parser.add_argument("--stats", action="store_true", help="print coverage breakdown")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    corpus_dir = args.corpus_dir or get_settings().corpus_dir
    print(f"Corpus directory: {corpus_dir}")
    print(f"Approved licences: {', '.join(sorted(ALLOWED_LICENCES))}")

    try:
        corpus = build_corpus(corpus_dir)
    except ValidationError as exc:
        # Refused, not warned about. See the module docstring.
        print(f"\nCORPUS REJECTED: {exc}", file=sys.stderr)
        return 2

    if not corpus.chunks:
        print("\nCORPUS REJECTED: no chunks found.", file=sys.stderr)
        return 2

    print(f"OK: {len(corpus)} chunks validated.")

    warnings = _check_coverage(corpus)
    if warnings:
        print("\nCoverage warnings:")
        for warning in warnings:
            print(f"  ! {warning}")

    if args.stats:
        _print_stats(corpus)

    if args.check:
        print("\n--check: validation only, index.json not written.")
        return 0

    index = corpus.to_index()
    index["source_sha256"] = _source_hashes(corpus_dir)
    out_path = corpus_dir / CORPUS_INDEX_NAME
    out_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {out_path} ({len(corpus)} chunks, version {corpus.version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

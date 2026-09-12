"""The teaching corpus: chunk schema, licence enforcement, and loading.

A chunk is an authored `##` section of a Markdown file under `data/corpus/`.
There is no chunking algorithm. Splitting on a fixed token window would add a
free parameter that has to be defended in the write-up and would cut sentences
mid-sign; a human-authored section boundary is both better and cheaper to
justify.

**Licence is enforced, not documented.** `build_corpus.py --check` refuses a
chunk whose `licence` is not on `ALLOWED_LICENCES` rather than warning about
it, because a warning in a build log is not a control — it is a thing nobody
reads until an examiner asks where the text came from. The repository is
public, so an unlicensed extract is published the moment it is committed.

The project's sources are CC BY-NC-SA 3.0, which carries **ShareAlike**: the
corpus files are themselves offered under that licence, and `data/corpus/
LICENCE.md` says so. That obligation is why `licence` is per-chunk metadata
rather than a README sentence — if a chunk from a differently-licensed source
is ever added, the mismatch has to surface at build time.

Runtime reads only `index.json`, so PyYAML is a dev dependency and the 13
runtime dependencies do not change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cxt.domain.entities.finding import FindingCategory
from cxt.shared.errors import ValidationError

__all__ = [
    "CorpusChunk",
    "Corpus",
    "ALLOWED_LICENCES",
    "ASPECTS",
    "MAX_CHUNK_CHARS",
    "CORPUS_INDEX_NAME",
    "load_corpus",
    "parse_corpus_file",
]

#: Licences whose terms permit redistributing an adapted extract in a public
#: repository. Anything else is refused at build time. CC BY-NC-ND is
#: deliberately absent: NoDerivatives does not permit a chunked adaptation,
#: which is what this corpus is.
ALLOWED_LICENCES: Final[frozenset[str]] = frozenset(
    {
        "CC BY-NC-SA 3.0",
        "CC BY-NC-SA 4.0",
        "CC BY-SA 4.0",
        "CC BY 4.0",
        "CC0 1.0",
        "Public Domain",
    }
)

#: What a chunk is *pedagogically about*, which is not the same as what it is
#: topically about. A chunk tagged `pleural_effusion` describing ultrasound-
#: guided drainage is on-topic and useless to a student reading a film.
#: Retrieval relevance is reported restricted to `aspect != management`.
ASPECTS: Final[frozenset[str]] = frozenset(
    {"radiographic_sign", "differential", "pitfall", "management", "technique"}
)

#: Three passages plus a 2000-char student answer plus the p3.0 template must
#: fit the notebook's 8000-char user turn. 700 leaves real headroom.
MAX_CHUNK_CHARS: Final[int] = 700

CORPUS_INDEX_NAME: Final[str] = "index.json"

_REQUIRED_FILE_KEYS: Final[tuple[str, ...]] = (
    "source",
    "url",
    "licence",
    "licence_url",
    "retrieved_on",
)
_REQUIRED_CHUNK_KEYS: Final[tuple[str, ...]] = ("id", "findings", "aspect")

_FRONT_MATTER_RE: Final[re.Pattern[str]] = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n", re.DOTALL
)
_SECTION_RE: Final[re.Pattern[str]] = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_CHUNK_META_RE: Final[re.Pattern[str]] = re.compile(
    r"\A\s*```ya?ml\s*\n(?P<yaml>.*?)\n```\s*\n", re.DOTALL
)
_DATE_RE: Final[re.Pattern[str]] = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z")

_VALID_CATEGORIES: Final[frozenset[str]] = frozenset(c.value for c in FindingCategory)


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """One retrievable passage, with everything needed to cite it."""

    id: str
    title: str
    text: str
    source: str
    url: str
    licence: str
    licence_url: str
    retrieved_on: str
    findings: tuple[str, ...]
    aspect: str
    confusions: tuple[str, ...] = field(default_factory=tuple)
    # Optional curated student-facing link ("Learn more"). Absent by default;
    # the frontend hides the link when empty rather than linking anywhere
    # unverified. Validated as https:// + labelled when present.
    read_more_url: str = ""
    read_more_label: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValidationError("CorpusChunk.id must be non-empty.")
        if not self.text.strip():
            raise ValidationError(f"Chunk {self.id!r} has no body text.")
        if len(self.text) > MAX_CHUNK_CHARS:
            raise ValidationError(
                f"Chunk {self.id!r} is {len(self.text)} chars; the cap is "
                f"{MAX_CHUNK_CHARS}. Three of these plus the student's answer "
                "must fit the notebook's 8000-char user turn."
            )
        if self.licence not in ALLOWED_LICENCES:
            allowed = ", ".join(sorted(ALLOWED_LICENCES))
            raise ValidationError(
                f"Chunk {self.id!r} carries licence {self.licence!r}, which is not "
                f"approved for redistribution in a public repository. Allowed: {allowed}."
            )
        if self.aspect not in ASPECTS:
            raise ValidationError(
                f"Chunk {self.id!r} has aspect {self.aspect!r}; known: {sorted(ASPECTS)}."
            )
        if not self.findings:
            raise ValidationError(f"Chunk {self.id!r} lists no findings.")
        for tag in (*self.findings, *self.confusions):
            if tag not in _VALID_CATEGORIES:
                # A typo'd tag silently removes a chunk from every filtered
                # retrieval, which reads as a corpus coverage gap in the report.
                raise ValidationError(
                    f"Chunk {self.id!r} references unknown finding {tag!r}. "
                    f"Known: {sorted(_VALID_CATEGORIES)}."
                )
        if not _DATE_RE.match(self.retrieved_on):
            raise ValidationError(
                f"Chunk {self.id!r} has retrieved_on={self.retrieved_on!r}; expected YYYY-MM-DD."
            )
        if self.read_more_url and not self.read_more_url.startswith("https://"):
            raise ValidationError(
                f"Chunk {self.id!r} has read_more_url={self.read_more_url!r}; "
                "expected an https:// URL (verified to resolve at curation time)."
            )
        if self.read_more_url and not self.read_more_label.strip():
            raise ValidationError(
                f"Chunk {self.id!r} sets read_more_url but no read_more_label."
            )
        if len(self.read_more_url) > 300:
            raise ValidationError(f"Chunk {self.id!r} has an overlong read_more_url.")

    @property
    def search_text(self) -> str:
        """What BM25 indexes: the heading counts as content.

        A section titled "Deep sulcus sign" whose body never repeats the phrase
        would otherwise be unreachable by a student who used exactly that term.
        """
        return f"{self.title}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "retrieved_on": self.retrieved_on,
            "findings": list(self.findings),
            "confusions": list(self.confusions),
            "aspect": self.aspect,
            "read_more_url": self.read_more_url,
            "read_more_label": self.read_more_label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CorpusChunk:
        return cls(
            id=str(payload["id"]),
            title=str(payload.get("title", "")),
            text=str(payload["text"]),
            source=str(payload["source"]),
            url=str(payload["url"]),
            licence=str(payload["licence"]),
            licence_url=str(payload["licence_url"]),
            retrieved_on=str(payload["retrieved_on"]),
            findings=tuple(payload.get("findings", ())),
            confusions=tuple(payload.get("confusions", ())),
            aspect=str(payload["aspect"]),
            read_more_url=str(payload.get("read_more_url", "")),
            read_more_label=str(payload.get("read_more_label", "")),
        )


@dataclass(frozen=True, slots=True)
class Corpus:
    chunks: tuple[CorpusChunk, ...]
    version: str

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def by_id(self) -> dict[str, CorpusChunk]:
        return {chunk.id: chunk for chunk in self.chunks}

    def for_finding(self, category: str) -> tuple[CorpusChunk, ...]:
        return tuple(c for c in self.chunks if category in c.findings)

    def to_index(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "chunk_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_index(cls, payload: dict[str, Any]) -> Corpus:
        return cls(
            chunks=tuple(CorpusChunk.from_dict(c) for c in payload.get("chunks", [])),
            version=str(payload.get("version", "unknown")),
        )


def _parse_yaml(text: str, *, where: str) -> dict[str, Any]:
    """Parse a YAML block. PyYAML is dev-only; the runtime reads index.json."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dev dependency
        raise ValidationError(
            "PyYAML is needed to build the corpus from Markdown. "
            "Install requirements-dev.txt, or read the prebuilt data/corpus/index.json."
        ) from exc
    try:
        loaded = yaml.safe_load(text)
    except Exception as exc:
        raise ValidationError(f"{where}: invalid YAML ({exc}).") from exc
    if not isinstance(loaded, dict):
        raise ValidationError(f"{where}: expected a YAML mapping.")
    return loaded


def parse_corpus_file(path: Path) -> list[CorpusChunk]:
    """Parse one `data/corpus/*.md` source file into its chunks.

    File-level front matter carries provenance shared by every chunk (source,
    url, licence, retrieved_on). Each `##` section carries its own fenced YAML
    block with the per-chunk keys. Missing keys raise rather than defaulting:
    a chunk that quietly lost its `findings` tag is invisible to filtered
    retrieval and shows up in the report as a coverage gap instead of a bug.
    """
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER_RE.match(raw)
    if match is None:
        raise ValidationError(f"{path.name}: missing the `---` front-matter block.")
    header = _parse_yaml(match.group("yaml"), where=path.name)
    for key in _REQUIRED_FILE_KEYS:
        if not str(header.get(key, "")).strip():
            raise ValidationError(f"{path.name}: front matter is missing {key!r}.")

    body = raw[match.end() :]
    sections = list(_SECTION_RE.finditer(body))
    if not sections:
        raise ValidationError(f"{path.name}: no `## ` sections, so no chunks.")

    chunks: list[CorpusChunk] = []
    for index, section in enumerate(sections):
        start = section.end()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(body)
        block = body[start:end].lstrip("\n")
        meta_match = _CHUNK_META_RE.match(block)
        title = section.group("title")
        if meta_match is None:
            raise ValidationError(
                f"{path.name}: section {title!r} has no ```yaml metadata block."
            )
        meta = _parse_yaml(meta_match.group("yaml"), where=f"{path.name} / {title}")
        for key in _REQUIRED_CHUNK_KEYS:
            if key not in meta:
                raise ValidationError(f"{path.name} / {title}: missing {key!r}.")
        chunks.append(
            CorpusChunk(
                id=str(meta["id"]),
                title=title,
                text=block[meta_match.end() :].strip(),
                source=str(header["source"]),
                url=str(header["url"]),
                licence=str(header["licence"]),
                licence_url=str(header["licence_url"]),
                retrieved_on=str(header["retrieved_on"]),
                findings=tuple(str(f) for f in meta.get("findings", ())),
                confusions=tuple(str(f) for f in meta.get("confusions", ())),
                aspect=str(meta["aspect"]),
                read_more_url=str(meta.get("read_more_url", "")),
                read_more_label=str(meta.get("read_more_label", "")),
            )
        )
    return chunks


def load_corpus(corpus_dir: Path) -> Corpus:
    """Load the prebuilt `index.json`, falling back to parsing the Markdown.

    The prebuilt index is preferred so the API host needs no YAML parser and no
    Markdown walk on startup. Falling back rather than failing means a developer
    who edits a chunk and forgets to rebuild still sees their edit, instead of
    debugging why it had no effect.
    """
    index_path = corpus_dir / CORPUS_INDEX_NAME
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValidationError(f"{index_path}: expected a JSON object.")
        return Corpus.from_index(payload)
    return build_corpus(corpus_dir)


def build_corpus(corpus_dir: Path) -> Corpus:
    """Parse every `*.md` under `corpus_dir`, validating as we go."""
    if not corpus_dir.exists():
        raise ValidationError(
            f"Corpus directory not found: {corpus_dir}. "
            "Set CORPUS_DIR, or run scripts/build_corpus.py."
        )
    chunks: list[CorpusChunk] = []
    seen: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.upper().startswith("LICENCE") or path.name.upper() == "README.MD":
            continue
        for chunk in parse_corpus_file(path):
            if chunk.id in seen:
                # Duplicate ids would make citation enforcement ambiguous and
                # silently halve corpus-coverage counts.
                raise ValidationError(
                    f"Duplicate chunk id {chunk.id!r} in {path.name}; "
                    f"already defined in {seen[chunk.id]}."
                )
            seen[chunk.id] = path.name
            chunks.append(chunk)
    return Corpus(chunks=tuple(chunks), version=corpus_version(chunks))


def corpus_version(chunks: list[CorpusChunk]) -> str:
    """A content hash over every chunk, so a report names the exact corpus.

    Recorded on each `Feedback` alongside `prompt_version`. Two runs that
    disagree are only comparable if this string matches.
    """
    import hashlib

    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.id):
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return f"c1.{len(chunks)}.{digest.hexdigest()[:12]}"

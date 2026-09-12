"""Corpus parsing and, above all, licence refusal.

The licence checks are the reason this file matters. The repository is public,
so an extract with the wrong licence is published the moment it is committed,
and by then the fix is a history rewrite rather than an edit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cxt.infrastructure.retrieval.corpus import (
    ALLOWED_LICENCES,
    MAX_CHUNK_CHARS,
    Corpus,
    CorpusChunk,
    build_corpus,
    load_corpus,
    parse_corpus_file,
)
from cxt.shared.errors import ValidationError

REAL_CORPUS_DIR = Path(__file__).resolve().parents[5].parent / "data" / "corpus"

HEADER = """---
source: "Test source"
url: "https://example.org/doc"
licence: "CC BY-NC-SA 3.0"
licence_url: "https://creativecommons.org/licenses/by-nc-sa/3.0/"
retrieved_on: "2026-09-12"
---
"""

SECTION = """
## A section title

```yaml
id: test-chunk-one
findings: [pleural_effusion]
confusions: [consolidation]
aspect: radiographic_sign
```

The costophrenic angle is blunted.
"""


def _write(tmp_path: Path, body: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestParsing:
    def test_parses_a_well_formed_file(self, tmp_path: Path) -> None:
        chunks = parse_corpus_file(_write(tmp_path, HEADER + SECTION))
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.id == "test-chunk-one"
        assert chunk.title == "A section title"
        assert chunk.text == "The costophrenic angle is blunted."
        assert chunk.findings == ("pleural_effusion",)
        assert chunk.confusions == ("consolidation",)

    def test_search_text_includes_the_heading(self, tmp_path: Path) -> None:
        """A section titled "Deep sulcus sign" whose body never repeats the
        phrase would otherwise be unreachable by a student who used it."""
        chunk = parse_corpus_file(_write(tmp_path, HEADER + SECTION))[0]
        assert "A section title" in chunk.search_text
        assert "costophrenic" in chunk.search_text

    def test_missing_front_matter_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="front-matter"):
            parse_corpus_file(_write(tmp_path, SECTION))

    @pytest.mark.parametrize(
        "key", ["source", "url", "licence", "licence_url", "retrieved_on"]
    )
    def test_each_required_front_matter_key_is_enforced(
        self, tmp_path: Path, key: str
    ) -> None:
        header = "\n".join(
            line for line in HEADER.splitlines() if not line.startswith(f"{key}:")
        )
        with pytest.raises(ValidationError, match=key):
            parse_corpus_file(_write(tmp_path, header + "\n" + SECTION))

    def test_section_without_metadata_block_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="metadata block"):
            parse_corpus_file(_write(tmp_path, HEADER + "\n## Bare section\n\nText.\n"))

    def test_file_with_no_sections_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="no `## ` sections"):
            parse_corpus_file(_write(tmp_path, HEADER + "\nJust prose.\n"))

    def test_duplicate_chunk_ids_across_files_are_refused(self, tmp_path: Path) -> None:
        """Duplicate ids make citation enforcement ambiguous and halve the
        corpus-coverage count without anything looking wrong."""
        _write(tmp_path, HEADER + SECTION, "a.md")
        _write(tmp_path, HEADER + SECTION, "b.md")
        with pytest.raises(ValidationError, match="Duplicate chunk id"):
            build_corpus(tmp_path)

    def test_licence_file_is_not_parsed_as_a_source(self, tmp_path: Path) -> None:
        _write(tmp_path, HEADER + SECTION, "doc.md")
        _write(tmp_path, "# Licensing\n\nProse, no front matter.\n", "LICENCE.md")
        assert len(build_corpus(tmp_path)) == 1


class TestValidation:
    def _chunk(self, **overrides: object) -> CorpusChunk:
        defaults: dict[str, object] = {
            "id": "x",
            "title": "T",
            "text": "Body text.",
            "source": "S",
            "url": "https://example.org",
            "licence": "CC BY-NC-SA 3.0",
            "licence_url": "https://example.org/l",
            "retrieved_on": "2026-09-12",
            "findings": ("pleural_effusion",),
            "aspect": "radiographic_sign",
        }
        defaults.update(overrides)
        return CorpusChunk(**defaults)  # type: ignore[arg-type]

    def test_unapproved_licence_is_refused(self) -> None:
        """CC BY-NC-ND is the live case: NoDerivatives forbids chunking."""
        with pytest.raises(ValidationError, match="not approved for redistribution"):
            self._chunk(licence="CC BY-NC-ND 4.0")

    def test_all_rights_reserved_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="not approved"):
            self._chunk(licence="All rights reserved")

    def test_nd_is_absent_from_the_allowlist(self) -> None:
        assert not any("ND" in lic for lic in ALLOWED_LICENCES)

    def test_over_length_chunk_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="the cap is"):
            self._chunk(text="w" * (MAX_CHUNK_CHARS + 1))

    def test_unknown_finding_tag_is_refused(self) -> None:
        """A typo'd tag silently vanishes from filtered retrieval and reads as
        a coverage gap in the report instead of a bug."""
        with pytest.raises(ValidationError, match="unknown finding"):
            self._chunk(findings=("pleuraleffusion",))

    def test_unknown_confusion_tag_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="unknown finding"):
            self._chunk(confusions=("nonsense",))

    def test_unknown_aspect_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="aspect"):
            self._chunk(aspect="vibes")

    def test_no_findings_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="lists no findings"):
            self._chunk(findings=())

    def test_malformed_retrieved_on_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            self._chunk(retrieved_on="Sept 2026")


class TestIndexRoundTrip:
    def test_index_round_trips(self, tmp_path: Path) -> None:
        _write(tmp_path, HEADER + SECTION)
        original = build_corpus(tmp_path)
        restored = Corpus.from_index(original.to_index())
        assert restored.version == original.version
        assert restored.chunks == original.chunks

    def test_load_prefers_the_prebuilt_index(self, tmp_path: Path) -> None:
        import json

        _write(tmp_path, HEADER + SECTION)
        built = build_corpus(tmp_path)
        (tmp_path / "index.json").write_text(
            json.dumps(built.to_index()), encoding="utf-8"
        )
        assert load_corpus(tmp_path).chunks == built.chunks

    def test_version_changes_when_text_changes(self, tmp_path: Path) -> None:
        """A report naming a corpus version must be able to rely on it."""
        _write(tmp_path, HEADER + SECTION)
        first = build_corpus(tmp_path).version
        _write(tmp_path, HEADER + SECTION.replace("blunted", "sharp"))
        assert build_corpus(tmp_path).version != first

    def test_missing_directory_raises_with_a_useful_message(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="build_corpus"):
            build_corpus(tmp_path / "nope")


class TestTheRealCorpus:
    """The committed corpus must satisfy every rule, not just the fixtures."""

    def test_loads_and_is_non_empty(self) -> None:
        corpus = load_corpus(REAL_CORPUS_DIR)
        assert len(corpus) > 0

    def test_every_chunk_carries_an_approved_licence(self) -> None:
        for chunk in load_corpus(REAL_CORPUS_DIR).chunks:
            assert chunk.licence in ALLOWED_LICENCES, chunk.id

    def test_covers_every_positive_finding_in_the_case_corpus(self) -> None:
        """There are no fracture cases in the 300, so fracture is not required."""
        corpus = load_corpus(REAL_CORPUS_DIR)
        for finding in (
            "pneumothorax",
            "pleural_effusion",
            "cardiomegaly",
            "consolidation",
            "nodule_or_mass",
        ):
            usable = [
                c for c in corpus.for_finding(finding) if c.aspect != "management"
            ]
            assert len(usable) >= 3, f"{finding} has only {len(usable)} usable chunks"

    def test_no_chunk_claims_a_source_it_does_not_have(self) -> None:
        """Phase 2's headline metric is citation fidelity. A corpus with
        fabricated attribution would make that number meaningless."""
        for chunk in load_corpus(REAL_CORPUS_DIR).chunks:
            assert chunk.source.strip()
            assert chunk.url.startswith("http")

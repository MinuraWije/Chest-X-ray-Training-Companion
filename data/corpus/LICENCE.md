# Corpus licensing and provenance

Everything under `data/corpus/` is offered under
**[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/)**
(CC BY-NC-SA 3.0).

ShareAlike is why: the project's chosen reference source, Radiopaedia, licenses
its articles under CC BY-NC-SA 3.0, and that term propagates to any adaptation.
Keeping the whole corpus directory under one licence avoids a mixed-licence
corpus where the obligations differ per file and nobody can tell which is which.
This licence applies to `data/corpus/` only; the rest of the repository is under
the project's own licence.

## Why the build refuses some sources

`backend/scripts/build_corpus.py --check` rejects any chunk whose `licence`
front-matter value is not in `ALLOWED_LICENCES`
(`backend/src/cxt/infrastructure/retrieval/corpus.py`). It refuses rather than
warns, because this repository is public: an unlicensed extract is published the
moment it is committed, and a warning in a build log is not a control.

Two sources were checked and rejected during Phase 2 planning:

| Source | Licence found | Why it is not used |
|---|---|---|
| WHO Manual of Diagnostic Imaging (Ellis & Flower, 2006) | "© World Health Organization 2006. All rights reserved." | Predates WHO's open-access policy. Reproduction requires written permission from WHO Press. The best pedagogical match, and unusable here. |
| StatPearls (NCBI Bookshelf) | CC BY-NC-**ND** 4.0 | NoDerivatives. Splitting an article into retrieval chunks is an adaptation, which that licence does not permit. |

`CC BY-NC-ND` is deliberately absent from the allowlist for this reason.

## Per-chunk provenance

Every chunk carries `source`, `url`, `licence`, `licence_url` and `retrieved_on`
in its file's front matter. `build_corpus.py` validates all five and records a
SHA-256 of each source file in `index.json`, mirroring the pattern already used
by `data/raw/PROVENANCE.json`.

## Status of the current chunks

The chunks presently committed are **authored teaching notes written for this
project**, not extracts from a third-party source. They exist so the retrieval
pipeline, the evaluation harness and the citation-enforcement path are testable
end to end.

They are marked as such in their `source` field — no chunk claims an origin it
does not have. This matters more than usual here: Phase 2's headline metric is
citation fidelity, and a corpus with fabricated attribution would make that
number meaningless.

Radiopaedia serves HTTP 403 to automated clients, so its articles must be
retrieved by hand before they can be chunked. Replacing these authored notes
with Radiopaedia-derived chunks is a content change only: the front matter,
the loader, the licence allowlist and the retrievers all stay as they are.

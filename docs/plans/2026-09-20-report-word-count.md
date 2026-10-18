# Word count and literature-review length — plan

**Date:** 2026-09-20
**Scope:** `docs/Final Project Report.docx`.
**Status:** plan only, no changes made to the report.

## Constraints given

1. **~16,000 words is an upper bound**, for the whole document as Word counts it.
2. **Chapter 2 (Literature Review) must stay at 7+ pages** and is not trimmed.
3. **Chapters 1, 3 and 4 may be trimmed** — previously off limits, now in scope.
4. **No claim may lose its supporting content or evidence.** Every trim below is
   either de-duplication, or prose converted to a table that carries the same
   facts. Nothing evidential is deleted.

---

## 1. Where the report stands

Counted from `word/document.xml`, split so it is clear which words are prose,
which are table cells, and which are auto-generated navigation lines:

| Section | Prose | Tables/captions | Nav | Total |
|---|---:|---:|---:|---:|
| Front matter | 1,047 | 140 | 370 | 1,557 |
| Ch.1 Introduction | 1,239 | 99 | 0 | 1,410 |
| Ch.2 Literature Review | 1,912 | 154 | 0 | 2,066 |
| Ch.3 Methodology | 1,331 | 2,458 | 0 | 3,789 |
| Ch.4 Design | 825 | 236 | 0 | 1,061 |
| Ch.5 Implementation | 5,701 | 585 | 0 | 6,286 |
| Ch.6 Testing *(skeleton)* | 759 | 0 | 0 | 759 |
| Ch.7 Conclusion | 2,191 | 140 | 0 | 2,331 |
| References | 755 | 0 | 0 | 755 |
| Appendices | 39 | 698 | 0 | 765 |
| **Total now** | | | | **20,779** |

**Chapter 6 is still a skeleton.** Written properly against the five-arm run it
would reach ~4,200. The finished document, untouched, therefore projects to
**~24,200 — about 8,200 over the cap.**

## 2. Two facts that shape the whole plan

**Relocating text to an appendix saves nothing.** Under a whole-document cap,
Appendix B's words count exactly as Chapter 5's do. The previous version of
this plan leaned on moving configuration detail into Appendix B; that route is
now closed. **Every word removed has to be removed by compression, not by
relocation.**

**Trimming Chapters 1, 3 and 4 cannot shrink the literature review.** Every
`Heading 1` in this document carries `pageBreakBefore`, so Chapter 2 always
starts at the top of a fresh page and Chapter 3 starts at the top of the next
one. Chapter 2's page span is determined solely by Chapter 2's own content,
which is not being touched. Its eight pages (ToC: Chapter 2 begins p.6,
Chapter 3 begins p.14) survive any amount of trimming elsewhere.

## 3. The budget

| Section | Now | Target | Change |
|---|---:|---:|---:|
| Front matter | 1,557 | 1,450 | −107 |
| Ch.1 Introduction | 1,410 | 1,150 | −260 |
| **Ch.2 Literature Review** | **2,066** | **2,066** | **0 — protected** |
| Ch.3 Methodology | 3,789 | 2,600 | −1,189 |
| Ch.4 Design | 1,061 | 900 | −161 |
| Ch.5 Implementation | 6,286 | 3,300 | −2,986 |
| Ch.6 Testing | 759 *(skeleton)* | 1,900 | +1,141 *(written to budget, not 4,200)* |
| Ch.7 Conclusion | 2,331 | 1,550 | −781 |
| **References** | **755** | **755** | **0 — protected** |
| Appendices | 765 | 650 | −115 |
| **Total** | 20,779 | **~16,300** | |

~16,300 is "16,000-ish" with the margin on the right side of being defensible:
tightening further would start costing evidence.

## 4. What gets cut, section by section

### Ch.5 Implementation: 6,286 → 3,300 (the bulk of the reduction)

| Sub-section | Now | Target | Method |
|---|---:|---:|---|
| §5.2 Technology Selection | 1,097 | 350 | 5.2.1–5.2.5 currently narrate in prose what Table 5.1 already states row by row. Keep Table 5.1 and one framing paragraph; delete the narration. **No fact lost — the table holds every version pin.** |
| §5.3.1 FR-1…FR-8 | 2,109 | 900 | Eight prose subsections become one traceability table (FR / module / mechanism / evidencing test) plus retained prose on the four that are not routine: FR-3 MedGemma, FR-4 retrieval, FR-5 structured generation, FR-6 citations. FR-1, FR-2, FR-7 and FR-8 are CRUD and role checks; the table states them completely. |
| §5.3.2 NFR-3/4/7 | 587 | 350 | The OWASP / NIST AI RMF / GDPR citations stay — NFR-3 is their only in-text home. Compression is in the surrounding explanation. |
| §5.3.3 Algorithm Implementation | 1,369 | 1,150 | **Lightest touch in the report.** This is the substantive contribution — BM25 ranking, Equations 5.1/5.2, the misconception boost and its negation guard, the ground-truth guard, citation enforcement. Prune restatement only; keep both equations and every worked detail. |
| §5.4 UI Implementation | 522 | 280 | One paragraph per figure. Figures 5.1–5.6 carry the argument once screenshots are in; the prose currently re-describes what the screenshot will show. |
| §5.1, §5.5 overview/summary | 221 | 120 | Match §4.1/§4.5's length, which they currently exceed. |
| Other prose | 381 | 150 | Linking sentences. |

### Ch.3 Methodology: 3,789 → 2,600

Chapter 3 is 65% table text, and those tables *are* the evidence, so they are
condensed rather than cut:

- **Six use-case narratives** (UC01–UC06, 911 words across six separate tables)
  → **one consolidated table**, one row per use case, columns: actor, trigger,
  main flow, postcondition. Every field is carried across; six table headers,
  six repeated preconditions and six sets of scaffolding are not. **~560 saved.**
- **Five feasibility tables** (technical, financial, operational, legal,
  schedule — 592 words) → **one table with five rows**. Same findings, one
  header. **~290 saved.**
- **Testing Approach prose** (§3.4.2 and its three sub-parts, 323 words) — this
  now duplicates Chapter 6, which states the same strategy against real
  results. Cut to a forward reference. **~200 saved.**
- **§3.3.1 Requirement Gathering** (188) and **§3.4.1 System Design** (77) →
  tightened. **~140 saved.**

Requirements traceability (231), the WBS (199) and the FR/NFR tables (195) are
untouched — they are referenced from Chapters 5 and 7.

### Ch.7 Conclusion: 2,331 → 1,550

§7.3 Problems (615), §7.4 Limitations (444) and §7.4.1 Deviations (169) state
the same three constraints — GPU quota and tunnel lifetime, NIH label noise,
corpus size — once each in *both* §7.3 and §7.4. Merge into a single pass:
each constraint stated once, as a problem encountered and the limitation it
leaves behind. **~530 saved with nothing dropped.** Future Enhancements
249→150, Concluding Remarks 215→150, §7.2.1 243→180.

### Ch.1 Introduction: 1,410 → 1,150

§1.2 Problem Background (187) and §1.3 Problem Statement (175) overlap
heavily; tighten to ~250 combined. §1.7.1 In-Scope (114) and §1.7.2 Out-of-Scope
(103) become two bulleted lists rather than prose. The aim, the five academic
questions, the six objectives and Table 1.1 are untouched — Chapter 7 answers
against them one by one and must keep something to answer against.

### Ch.4 Design: 1,061 → 900

Already terse. §4.3.6 Interface Design (217) is the only fat subsection and
partly duplicates §5.4; cut to ~120. All 17 figures and their captions stay.

### Front matter 1,557 → 1,450 and Appendices 765 → 650

Front matter is mostly navigation that regenerates on F9. The Abstract is
rewritten last with Chapter 6's headline figures and holds at ~300. Appendix B's
six configuration tables merge into three (model/inference, prompt/retrieval,
corpus/data) — same rows, half the headers.

### Ch.6 Testing: written to 1,900, not 4,200

The cheapest reduction in the report, because it is not written yet. Budget:

| 6.x | Content | Words |
|---|---|---:|
| 6.1–6.2 | Overview, objectives, testing criteria | 200 |
| 6.3 | Automated suite: 577 tests, ruff + mypy --strict, FR coverage | 250 + table |
| 6.4 | Five-arm comparison table; P@3 by archetype; citation rate and validity; overlap against its shuffled null | 700 + 3 tables |
| 6.4.3 | Proxy validation against the LLM-judge labels (80%, κ = 0.62) | 250 + table |
| 6.5–6.6 | Functional test cases — **one row per FR**, each Pass justified by a named test file and function | 200 + table |
| 6.7–6.8 | Evaluation against objectives; chapter summary | 300 |

Results tables are never compressed: they are the chapter's entire evidential
content, and they are what a marker checks.

**Amended 2026-09-20 after the re-run.** The five-arm run completed and
reproduced the 19 September run cell for cell. Three defects in the eval
harness were found in the process (the `abstained` flag conflates "this arm has
no retrieval by design" with "the model declined"; the confabulation rate is
still 100% by construction, for a different reason than the one fixed earlier;
`_feedback_prose` carries parser scaffolding and a duplicated rationale). All
three are recomputable offline from `reports/phase2_eval.json` — no further GPU
run is needed — but until they are resolved, **§6.4 reports P@3, coverage,
citation rate/validity, overlap lift and latency, and does not report an
abstention or confabulation figure.** The word budget above is unchanged; the
confabulation paragraph is replaced by a Limitations entry in Chapter 7.

## 4a. Progress, 2026-09-20

- **Chapter 6 is written**, against the corrected five-arm run: 2,667 words
  (2,475 prose across twelve sections, plus Tables 6.1–6.6). That is ~770 over
  its 1,900 budget, to be recovered in the trim.
- **Figures 5.1–5.7 are captured and inserted.** The screenshots were driven
  through a headless browser against the running application and the live
  model, so Figure 5.4 shows real MedGemma output rather than a canned
  response. The drafted Figure 5.6 covered both educator screens; it has been
  split, the rubric form is now Figure 5.7, and the List of Figures has the
  new entry.
- **The report is now 22,591 words**, 24 images, 42 tables. Front matter and
  Chapters 1–4 verified unchanged against the pre-Chapter-6 file.
- **Revised target.** With Chapter 6 at 2,667 rather than 1,900, the trim in
  §4 has to remove ~6,600 rather than ~5,800. The extra comes from Chapter 6
  itself: §6.4.2 and §6.6.1 both restate Table 6.2 in prose, and one of the
  two can go.

## 5. The no-lost-evidence rule, and how it is verified

Three trims in this plan convert prose into tables (§5.2, §5.3.1, the use-case
narratives) and one merges duplicated sections (§7.3/§7.4). The rule governing
all of them:

> A trim may remove **restatement**. It may not remove a **fact, a citation, a
> version number, a measurement, a test name, or a figure/table**.

Verification, before the document is considered done:

1. **Citation diff.** Extract every in-text citation from the current file and
   from the trimmed file. The sets must be identical — the reference list is
   unchanged, so any citation lost in trimming would orphan a reference.
2. **Figure and table diff.** 36 tables and 17 drawings today. The trimmed
   document must have all 17 drawings and its tables must account for every
   original row: consolidations are checked row-count in, row-count out.
3. **Traceability check.** Every FR-1…FR-8 and NFR-1…NFR-7 must still be
   addressed somewhere in Chapter 5, and every AQ and objective must still be
   answered in Chapter 7. The compressed §5.3.1 table is the mechanism, not an
   exception to it.
4. **Word count.** Word's own Review → Word Count, recorded.
5. **Page check.** F9, update entire table; confirm Chapter 2 still spans 7+
   pages and Chapters 1–4 read correctly with their new page numbers.

## 6. Order of work

1. **Write Chapter 6 to its 1,900-word budget**, against the five-arm run now
   executing. Doing this first sets the real total before any trimming.
2. **Record the rubric fix.** §5.3.1 and Chapter 7 currently carry the
   frontend/backend rubric mismatch as a known defect. It was fixed on
   2026-09-20 — `ReviewPage.tsx` now submits the six criteria
   `_VALID_CRITERIA` accepts, guarded by
   `tests/backend/unit/domain/test_review_form_rubric.py`. Rewrite both
   passages as "found during implementation review and corrected"; drop the
   Chapter 7 limitation. This is a trim as well as a correction.
3. **Trim Chapter 5** — the largest single reduction, and the one whose result
   determines whether the rest of the budget is comfortable.
4. **Trim Chapter 3**, then Chapters 7, 1, 4, the appendices and front matter.
5. **Rewrite the Abstract last**, with Chapter 6's headline figures.
6. **Run the five verification checks in §5.**

Chapters 1–4 are edited in place in `docs/Final Project Report.docx` only.
`docs/Draft Report.docx` remains read-only, so the original text of every
trimmed passage stays recoverable.

# Final Project Report — assemble from the Draft, add Chapters 5 and 7

## Context

`docs/Draft Report.docx` is a complete, figure-rich dissertation covering **front
matter + Chapters 1–4 + References** — 1,116 paragraphs, 24 tables, 17 embedded
images, ~10,800 words. It stops at Chapter 4 (Design). `docs/Final Project Report
Template.docx` requires seven chapters plus appendices.

The gap is **Chapter 5 (Implementation), Chapter 6 (Testing), Chapter 7
(Conclusion), and the appendices** — everything downstream of Design. The
implementation those chapters describe is finished and on disk: Phase 1 (MedGemma
over a Kaggle tunnel, 300 curated NIH cases) and Phase 2 (BM25 retrieval grounded
before generation, five evaluation arms, citation enforcement) both landed, with
574 backend tests passing and ruff + `mypy --strict` clean.

The outcome is `docs/Final Project Report.docx` — the Draft's content carried over
byte-for-byte, with Chapters 5 and 7 written, Chapter 6 scaffolded, and Appendix B
built from the code.

### Two decisions already taken

**Chapter 6 is deferred by choice.** `backend/reports/` has no `phase2_eval.md` —
the five-arm comparison (Phase 2 Step 9) is blocked on the notebook, and all 150
pairs in `data/eval/relevance_labels.json` are unlabelled. Rather than write
Chapter 6 around placeholder tables, this plan lays in the **heading skeleton
only** and leaves the chapter to be written in one pass against real numbers.

**Chapter 6's method is fixed for when it is written:** functional test cases
(TC01…TCnn) will be marked Pass on the evidence of the specific automated test
that exercises each FR, naming the test file and function in the table. No manual
result will be recorded as Pass unless it was actually run.

---

## The mechanical approach — copy the Draft, append

Do **not** generate a new `.docx` from scratch. That would discard 17 embedded
figures, 24 tables, the caption styling and the numbering definitions.

Instead: **copy `docs/Draft Report.docx` to `docs/Final Project Report.docx`, open
it with `python-docx` (1.2.0, already in `backend/.venv`), and append.** Three
facts from the Draft's XML make this clean:

1. **Headings 1–4 auto-number.** `word/styles.xml` binds every heading style to
   `numId=1` (`ilvl` 0–3). Appending a paragraph with style `Heading 1` and text
   `IMPLEMENTATION` renders as **CHAPTER 5. IMPLEMENTATION** automatically; a
   `Heading 2` under it becomes 5.1. No manual numbering anywhere.
2. **`Heading 1` carries `pageBreakBefore`.** Each new chapter starts its own page
   with no explicit break inserted.
3. **There are zero field codes in the document** (`<w:instrText>` count = 0). The
   Table of Contents, List of Tables, List of Figures and captions are all *static
   typed text*. Captions must therefore be typed literally (`Table 5.1: …`,
   `Figure 5.1: …`) using the `Caption` style.

### Front matter that must be touched

The Draft's static navigation lists cannot stay as they are once three chapters
are added. Replace the static ToC / List of Tables / List of Figures paragraphs
with **live Word fields** (`TOC \o "1-3" \h`, `TOC \h \z \c "Table"`,
`TOC \h \z \c "Figure"`), so page numbers regenerate on F9 — which is exactly what
the template's own note instructs. This is navigation, not authored content.

Four further front-matter items genuinely require a small edit, each flagged
because it touches Draft text:

| Item | Why it must change | Size of edit |
|---|---|---|
| §1.8 "Structure of the Proposal" | Lists only Chapters 2–4 and says "ends with the reference list". Also still says *Proposal*, not *Report*. | Retitle; rewrite one paragraph |
| List of Abbreviations | Ch.5 introduces BM25, CI, CSV, DTO, LoRA, SPA-side terms not in the 38-row table | Append rows |
| List of Equations | Says "Only one formal equation is used in the first four chapters" — Ch.5 adds the BM25 score and the saturating normalisation | Revise the sentence; add rows |
| Table 1.1 (AQ → objective map) | "Chapter / Section" column points only at Chapters 1–4 | Update three cells |

Everything else in Chapters 1–4 stays untouched.

---

## Chapter 5 — IMPLEMENTATION (the main writing task)

Follows the template's 5.1–5.5 exactly, in the Draft's register: third person,
present tense for the artefact, past tense for process, "the developer" never "I",
British spelling, every table introduced by a sentence naming it and followed by a
`Caption` line.

- **5.1 Chapter Overview** — one paragraph, matching the Draft's pattern.
- **5.2 Technology Selection** — 5.2.1 stack, 5.2.2 languages, 5.2.3 frameworks,
  5.2.4 libraries, 5.2.5 IDEs, 5.2.6 summary table (**Table 5.1**). Source the
  exact pins from `backend/requirements.txt`, `backend/pyproject.toml` and
  `frontend/package.json`. **Note a discrepancy to resolve here:** Chapter 3 says
  "React 18"; the artefact ships React 19. State the actual versions in 5.2 and
  add one sentence acknowledging the upgrade, rather than editing Chapter 3.
- **5.3 Implementation of the Artefact**
  - **5.3.1 Functional requirements** — one subsection per FR-1…FR-8, each naming
    the concrete module and reusing the mapping already in **Table 3.9** so
    traceability is visible. Cover: the manifest repository and 300-case pipeline;
    the attempt form and validation; `MedGemmaAnalyzer` and the Kaggle tunnel
    client (warmup, retries, base64 image transport); `PromptBuilder` and prompt
    versioning (`p2.4` → `p3.0`); `OutputParser` JSON repair + the retrying
    parser; the retrieval layer (corpus loader, BM25, the four runtimes, the
    saturating score floor, abstention); citation enforcement; the educator rubric.
  - **5.3.2 Non-functional requirements** — NFR-1…NFR-7 in order. NFR-3 is the
    place to give the orphan **OWASP / NIST AI RMF / GDPR** references an in-text
    home; NFR-4 is prompt-and-retrieval provenance in `Feedback.raw`; NFR-7 is the
    ports-and-adapters argument, citing Table 4.1.
  - This is also where the Draft's weakest section is repaired: **§4.4 Algorithm
    Design is three paragraphs and one equation with no pseudocode.** Chapter 5
    supplies the real algorithms — BM25 ranking, the `bm25/(bm25+k)` normalisation
    (**Equation 5.1 / 5.2**), the misconception boost with its negation guard, and
    the retrieve-then-generate ordering.
- **5.4 User Interface Implementation** — prose per screen, with **captioned
  placeholder figures (Figure 5.1 onward)** the user drops PNGs into later. Anchor
  the claims to the six interface-design principles already quoted in §4.3.6.
- **5.5 Chapter Summary** — two or three sentences, matching §4.5's shape.

**One consistency item to handle in prose, not by editing Chapter 3:** Table 3.11
claims 257 automated tests. The suite now collects **576 and passes 574** (2
skipped — the live-tunnel smoke tests, which need `MEDGEMMA_TUNNEL_URL`). Chapter
5 states the current figure and says plainly that the suite grew during Phase 2.

**One real defect to report honestly, not paper over.** The backend rubric
(`domain/entities/review.py`) validates six criteria — `finding_recognition`,
`retrieval_relevance`, `evidence_grounding`, `clarity`,
`usefulness_for_learning`, `safety_trustworthiness` — and rejects anything else.
The frontend (`frontend/src/pages/ReviewPage.tsx:7-14`) submits a *different* six:
`image_quality_assessment`, `anatomical_observation`, `pathology_identification`,
`differential_ranking`, `clinical_reasoning`, `report_clarity`. A review submitted
through the UI therefore fails validation with HTTP 422. This affects FR-7 and
UC06, so Chapter 5 must state it and Chapter 7 must carry it as a known defect.
Whether to fix the code before submission is the user's call — it is a small,
contained change (align the frontend list with the domain list), and fixing it is
the better outcome, but it is outside this plan's scope.

---

## Chapter 6 — TESTING (skeleton only, by decision)

Insert the template's headings 6.1–6.8 with a single bracketed note under each
saying what it will contain and what it is waiting on. Nothing is asserted.

Before it can be written the user must: run
`python scripts/dev.py feedback-eval --runtime kaggle_tunnel --max-cases 30`
for **all five arms in one notebook session** (so model drift does not confound the
comparison), and label the 150 pairs in `data/eval/relevance_labels.json` — the
latter needs no GPU.

The evidence that already exists and will fill the chapter: 574 passing tests
across domain/application/infrastructure/interfaces, ruff + `mypy --strict` clean
on 94 files, the e1–e5 Phase 1 prompt experiments (including the finding that fp16
is unusable at 0% structure validity, and that elaborating the prompt degraded
structure validity without lifting the 16.7–25.0% accuracy band), the four label
audits with their mode-collapse distributions, and the five p2.x feedback probes
at 100% structure validity over 42 generations.

---

## Chapter 7 — CONCLUSION

Full prose, 7.1–7.6 per the template.

- **7.2.1 / 7.2.2** — the aim, then O1…O6 one at a time. O1–O5 are achieved and
  evidenced. **O6 is partially achieved**: the harness, the five arms and the
  metrics all exist and are unit-tested, but the reported run has not executed.
  Say so directly.
- **7.3 Problems and Challenges** — the honest ones: MedGemma recognition sitting
  at 16–25% against a 16.7% chance floor across six prompt strategies, which
  forced the architecture from *classifier* to *tutor given the confirmed finding*;
  fp16 failing outright; Kaggle GPU quota and ngrok session lifetime; the WHO
  corpus licence check; NIH label noise.
- **7.4 Limitations** — no frontend tests at all (oxlint and `tsc -b` are the only
  frontend gates); a 30-chunk corpus against the plan's 50–80 target, all of it
  author-written teaching notes rather than the intended WHO extracts;
  single-labeller relevance judgements; NIH NLP-mined labels; the 300 images come
  from only **86 patients, one contributing 47**; no `fracture` cases exist in
  NIH ChestX-ray8, so one of the seven declared categories is untestable; lexical
  proxies standing in for misconception detection and for confabulation; the
  rubric-drift defect above.
- **7.4.1 Deviations** — fine-tuning (AQ4's third arm, WBS 6.1–6.2) was scoped
  "if time allows" and was not done; Phase 2 replaced it as the substantive
  contribution. The Abstract already positions fine-tuning as future work, so this
  is consistent.
- **7.5 Future Enhancements** — LoRA/PEFT, the MIMIC-CXR swap, corpus growth,
  multi-labeller agreement, a frontend test runner.
- **7.6 Concluding Remarks**.

Two paragraphs in 7.2.2 and 7.3 depend on Chapter 6's numbers; write them now in a
form that a single revision pass can sharpen once the eval has run.

---

## Appendix B — Technical Configuration

Built entirely from the code, no new facts needed: MedGemma model id and
generation parameters (`max_new_tokens`, timeouts, retries, repair attempts, image
byte cap), the prompt version fingerprints, the notebook server's 4000/8000
character caps, BM25 `k1`/`b`, the normalisation constant, `top_k`, the score
floor, corpus version and chunk counts by aspect, the manifest schema, and the
train/val/test split rule.

Appendices A, C and D are not built (Table 2.1 already covers A; D is meaningless
until tests have been executed).

---

## Files

| File | Action |
|---|---|
| `docs/Draft Report.docx` | **Read only — never modified** |
| `docs/Final Project Report.docx` | Created as a copy, then appended to |
| `docs/Final Project Report Template.docx` | Reference for section order and headings |
| `backend/reports/*.md` | Evidence for Chapter 6 later; Phase 1 context for Ch.5/7 |
| `backend/src/cxt/**`, `frontend/src/**`, `kaggle_runtime/**` | Source of every Chapter 5 claim |

Work in a scratch build script under the session scratchpad; the only repository
artefact produced is the new `.docx`.

---

## Verification

1. Open `docs/Final Project Report.docx` in Word. Select all, **F9**, choose
   "Update entire table" — the ToC, List of Tables and List of Figures must
   regenerate with Chapters 5–7 and correct page numbers.
2. Confirm chapter headings render as **CHAPTER 5. IMPLEMENTATION**,
   **CHAPTER 6. TESTING**, **CHAPTER 7. CONCLUSION** with no manual numbers, and
   that each starts on a fresh page.
3. Confirm Chapters 1–4 are pixel-identical to the Draft: all 17 figures present,
   all 24 tables intact, captions unchanged. Diff the extracted text of Chapters
   1–4 between the two files — it should differ only in §1.8, Table 1.1, and the
   navigation lists.
4. Confirm table and figure numbering continues without collision (next ids are
   **Table 5.1** and **Figure 5.1**).
5. Re-run `backend\.venv\Scripts\python.exe scripts\dev.py test` and `… lint`
   immediately before writing 5.3, so the counts quoted in Chapter 5 are the
   counts on the day.
6. Word count and page count sanity check against the module's limit.

---

## Prerequisite, unrelated to the report

The ngrok authtoken exposed earlier in this project is still live and still needs
revoking at `dashboard.ngrok.com`. It also blocks the Chapter 6 evaluation run.

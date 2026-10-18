# Answer comparison: show the student where their reading matched the case

## Context

The Proposal, the Research Idea, the Professionalism Assessment and the Draft
Report all promise a feedback part that **compares the student's answer with
the reference**, with missed clues highlighted. The `Feedback` entity still
has `comparison` and `missed_clues` fields, but nothing fills them with a real
comparison, and `FeedbackPage.tsx` shows the student's text and the model's
feedback as two unrelated cards.

This plan delivers that comparison **without the model**. Every case already
has a confirmed answer (`Case.confirmed_categories`: primary first, then
secondaries), and `infrastructure/retrieval/misconception.py` already detects
which findings a student named and which they ruled out, negation included. A
comparison built from those two pieces of data is deterministic,
unit-testable, needs no GPU, and doesn't touch the prompt, so every number in
Chapter 6 stays valid.

Keeping the model out of this decision is deliberate, not a shortcut: Phase 1
measured MedGemma's recognition at 16.7–23.3% against a 14.3% chance floor.
A model judging whether the student was right would be close to random.

## Step 0: stop showing the answer before the attempt (prerequisite)

`CasePage.tsx:63` renders the heading **"Case of {groundTruthLabel}"**, and the
image `alt` text on `CasePage.tsx:75` and `CaseListPage.tsx:72,79` does the
same. A student sees the answer before writing anything, which would make any
"you got it right" highlight meaningless.

- Frontend: replace the heading and alt texts with a neutral one ("Case
  `<short id>`", alt "Chest radiograph for this case"). Remove "Case of …"
  from the case list cards too.
- Backend (recommended, same change): stop sending `groundTruthCategory` and
  `groundTruthLabel` from `GET /cases` and `GET /cases/{id}` to the `STUDENT`
  role. Keep them for `EDUCATOR`/`ADMIN`. A hidden UI field is still readable
  in the network tab. Check that the `?finding=` filter isn't used by the
  student UI, because it leaks the answer the same way. If it is, keep it for
  educators only.
- Update `frontend/src/types/api.ts` (make the two fields optional) and the
  route tests.

This also fixes a validity problem for Chapter 6 / the review study:
attempts written while the title showed the answer measure reading, not
interpretation. Record that as a limitation for any attempts made before this
fix.

## Step 1: span-aware mention detection (`misconception.py`)

`named_categories` returns categories only, and `_normalise` collapses
whitespace, so its offsets don't map back onto the student's text. Add,
alongside it:

```python
@dataclass(frozen=True, slots=True)
class Mention:
    category: str
    start: int       # offsets into the ORIGINAL text, as stored on the attempt
    end: int
    text: str        # the exact substring, for the UI and for tests
    negated: bool

def find_mentions(text: str) -> tuple[Mention, ...]: ...
```

Implementation rules:

- Build the negation mask on a copy that is **the same length** as the input.
  Lower-case it and map each whitespace character to a single space, 1:1,
  never collapsing. Then run the existing forward and postposed cue logic on
  that copy, refactored to return a boolean mask instead of a blanked string.
- Compile each alias as word-boundary-anchored, with internal spaces as `\s+`
  ("pleural\s+effusion"), case-insensitive, run on the original text.
- Overlaps: longest match wins and spans don't overlap, so "pleural effusion"
  is one mention, not two.
- `negated` = the span intersects the mask.
- **Leave `named_categories` and `negated_categories` unchanged.** The
  misconception retrieval arm and the Chapter 6 numbers depend on them.
  Instead add a **parity test**: for every text in the existing misconception
  tests and in `tests/fixtures/student_attempts.json`, the set of non-negated
  `find_mentions` categories equals `named_categories(text)`, and likewise for
  negated ones. If parity holds, a later refactor can make `named_categories`
  call `find_mentions`. That isn't part of this plan.

## Step 2: the comparison service

New module `cxt/application/services/answer_comparison.py` (application
layer; it already imports from `infrastructure.retrieval.misconception`,
following the existing pattern in `submit_attempt.py:35`).

```python
class Verdict(str, Enum):
    CORRECT = "correct"                    # named the primary finding
    ALSO_PRESENT = "also_present"          # named a confirmed secondary finding
    NOT_IN_RECORD = "not_in_record"        # named something the case does not confirm
    MISSED = "missed"                      # confirmed, never mentioned
    WRONGLY_EXCLUDED = "wrongly_excluded"  # confirmed, but the student said it was absent
    CORRECTLY_EXCLUDED = "correctly_excluded"  # ruled out, and not confirmed

@dataclass(frozen=True, slots=True)
class ComparisonItem:
    category: str
    verdict: Verdict
    spans: tuple[tuple[int, int], ...]     # empty for MISSED

@dataclass(frozen=True, slots=True)
class AnswerComparison:
    items: tuple[ComparisonItem, ...]
    matched_anything: bool                 # False -> the UI shows the honest fallback
    version: str                           # "ac1", bumped whenever aliases/rules change

def compare_answer(free_text: str, case: Case) -> AnswerComparison: ...
```

Rules, each one a test:

- A category named both positively and negatively ("no effusion on the left,
  small effusion on the right") counts as **named**. Positive mentions win, and
  both spans are returned.
- `no_finding` case + student names an abnormality → `NOT_IN_RECORD`.
  Abnormal case + student writes "unremarkable" → `no_finding` is
  `NOT_IN_RECORD` and the primary is `MISSED`.
- Nothing recognised at all → `matched_anything=False`, with **no** `MISSED`
  items. A misspelling ("pneumothrax") must not be reported as "you missed
  pneumothorax" when the student actually wrote it. Show the fallback message
  instead.
- `NOT_IN_RECORD`, not "wrong": NIH labels are NLP-mined and noisy, so the UI
  wording is "not in this case's record", which is the true statement.
- Item order: primary, then secondaries, then extras, then exclusions. That
  order is stable and matches how the UI reads.

Computed **on read, not persisted**: it's a pure function of the stored
`free_text` and the case, so old attempts get it too. `version` goes into the
DTO so a screenshot or a review can always say which rules produced it.

## Step 3: API

- `routes/attempts.py`: `POST /attempts` and `GET /attempts/{id}` add a
  `comparison` object to the response. `GET` needs `get_case_repository`
  injected. If the case no longer exists, return `comparison: null` rather
  than a 500.
- `GET /attempts` (list) stays without it, like `feedback`.
- Educator review route: include it too, so the educator marks against the
  same view the student saw (the same reason `shape_feedback` was unified).
- DTO shape (camelCase, new `schemas/comparison.py`):
  `{version, matchedAnything, items: [{category, label, verdict, spans: [{start, end}]}]}`.
  `label` is the human-readable finding name.
- Comparing an attempt after it's submitted can't leak anything, since the
  student is past the answer by then. But the comparison names the confirmed
  findings, so the existing "students only see their own attempt" check in
  `get_attempt` already covers it. Add a test that proves it.

## Step 4: frontend

New `components/AnswerComparison.tsx`, replacing the plain "Your
interpretation" card in `FeedbackPage.tsx`.

- **Highlighted text.** Render `freeText` split at the span offsets.
  Each span is a `<mark>` inside a `<button>`, so it can be reached by keyboard
  and clicked or focused to show a short explanation ("You named pleural
  effusion, which is a confirmed finding on this film"). Colour **plus** an
  icon or text, never colour alone:
  - green ✓ correct / also present
  - amber ⚠ not in record
  - red ✗ wrongly excluded
  - slate ○ correctly excluded
- **Checklist** under the text: one row per confirmed finding (✓ identified
  / ✗ missed / ✗ said absent), then extras. A missed row has a "See where to
  look" link that scrolls to the model's Key observations section.
- **Layout.** Two columns on `lg:` (your answer | model feedback) and
  stacked below that. `FeedbackPage` goes from `max-w-3xl` to `max-w-6xl`
  to fit.
- **Honesty line**, always visible in small text: "Matched by keyword against
  the case record, not by reading for meaning. Misspellings and unusual
  phrasing may not be recognised."
- **Fallback** when `matchedAnything` is false: "We couldn't match any finding
  names in your answer. Compare it with the feedback on the right." No
  missed list.
- Old attempts or a missing case (`comparison === null`): render today's
  plain card, unchanged.
- Types in `types/api.ts`; the offsets index the same `freeText` string the
  API returns. Guard in the component: if a span runs past the text length,
  drop it rather than throw.

## Tests

Backend (pytest; ruff + `mypy src scripts` must stay clean):

- `test_misconception.py`: `find_mentions` offsets with double spaces,
  newlines, mixed case; multi-word aliases across a line break; longest-match
  overlap; forward and postposed negation; the terminator ("no
  pneumothorax, but an effusion"); the parity test against `named_categories`.
- New `test_answer_comparison.py`: one test per rule in Step 2, plus
  secondary findings (a `Case` with `secondary_findings`) and an empty or
  unrecognised answer.
- `test_attempts_route.py`: `comparison` present on POST and GET; `null` when
  the case is gone; another student's attempt is still 404.
- Case route tests: students don't receive the ground-truth fields;
  educators do.

Frontend: no test runner exists (a known limitation in Chapter 7), so the
gates are `tsc -b` and `oxlint`, plus a Playwright screenshot of the feedback
page using the devDependency already added. Don't add Vitest in this change.

## Report (only if not yet submitted)

- Chapter 5, FR-3: a short subsection on the comparison: deterministic by
  design, and why (the Phase 1 recognition figure). Add one screenshot figure.
  Step 0 goes under NFR / validity.
- Chapter 7 Limitations: lexical matching (misspellings, differential
  phrasing such as "effusion versus consolidation" reads as both), no
  laterality (NIH labels carry none, so "left" vs "right" can't be checked),
  and NIH label noise behind "not in record".
- If already submitted: none of this goes into the report. It's a
  post-submission enhancement.

## Out of scope

- Model-written per-sentence assessment (layer 2): needs a new prompt
  version and a re-run of the evaluation.
- Region highlighting on the image (layer 3): depends on NIH
  `BBox_List_2017.csv` coverage of the 300 cases, which hasn't been checked.
- Laterality and severity matching: no ground truth exists to check them
  against.

## Order of work

0 → 1 → 2 → 3 → 4. Each step lands with its tests. Steps 0–3 are backend
and testable on their own; Step 4 is pure rendering on top of a finished
contract.

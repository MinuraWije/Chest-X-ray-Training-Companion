"""Prompt builder for the MedGemma feedback call.

The canonical templates live in `kaggle_runtime/prompts/` so the
notebook and the backend use identical text. The backend loads them
at import time and composes the per-case instruction via
`PromptBuilder.build(...)`.

The templates are *given* the confirmed finding rather than asking the
model for it. The case record already holds the verified answer, so
asking the model to rediscover it adds nothing a student benefits from
and adds a way to be wrong: the eval harness measured 16-25% recognition
across six prompt strategies against a 16.7% chance floor, which means
feedback built on the model's own guess would misjudge most students.
What the product needs is the explaining, not the diagnosing.

Substitution uses `string.Template`, not `str.format`: the student's
free text is untrusted input and a literal `{` in it would make
`str.format` raise before the model was ever called. `safe_substitute`
also leaves an unknown `$placeholder` intact rather than raising.

`PROMPT_VERSION` plus a content hash of the loaded templates is recorded
on every `Feedback` (NFR-4), so a stored feedback row can always be
traced back to the exact prompt that produced it.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Final

from cxt.domain.entities.feedback import RetrievedPassage
from cxt.domain.entities.finding import Finding
from cxt.shared.errors import ValidationError

_PROMPTS_DIR = Path(__file__).resolve().parents[5] / "kaggle_runtime" / "prompts"

# p2.0: the confirmed finding is supplied to the model instead of being
# asked for. A contract change, not a wording tweak, hence the major bump.
# p2.1: differential `confidence` restated as student temptation rather than
# belief (a p2.0 probe returned 0.95 no_finding alongside pneumonia at 0.80),
# and the cardiothoracic threshold made explicit (a heart described as enlarged
# at a ratio of 0.50, which is normal).
# p2.2: tried and REVERTED. p2.1's "list at most one differential" clause was
# suppressing a genuinely present second finding, so p2.2 replaced it with an
# instruction to report any visible second finding. That recovered the true
# finding on one case but invented a false one on another — a confirmed
# pneumothorax was also reported as a right-sided pleural effusion, which is
# its physical opposite — and confidence-ordering violations went 2/6 to 4/6.
# Told to look harder for a second finding, the model finds one that is not
# there. The text below is p2.1's verbatim, so the version stays p2.1 and the
# p2.1 probe report remains directly comparable.
#
# The confidence ordering the prompt kept failing to enforce is now clamped in
# `OutputParser` instead: three prompt versions moved the violation count
# 3 -> 2 -> 4 with no monotone improvement, which is a wording lottery rather
# than a fix. Deterministic post-processing is the right tool for an invariant.
#
# p2.3: every category the case's labels confirm is now supplied, not just the
# primary `find_primary_category` reduced them to. A third of the test split
# carries more than one, so the tutor was previously marking against an answer
# key missing a third of its answers. This also removes the need for p2.2's
# "go and find a second finding" instruction, which invented one.
#
# p2.4: "credit the student for any of them they identified" told a student who
# wrote "the lungs look mostly clear" that they had "correctly identified the
# presence of a pleural effusion". Replaced with a prohibition on attributing
# anything the student's words do not contain — false praise is the one error
# a learner cannot detect from the feedback itself.
# p3.0: the JSON schema block moves from the system turn to the user turn, and
# the user turn gains an `$evidence_block` carrying retrieved passages.
#
# The move is forced, not stylistic. `$clinical_context` was never populated —
# `ManifestRepository._to_case` stored age/sex/view and never rendered them, so
# every production prompt said "(no clinical context provided)" while the row
# held `56,M,AP`. Filling it in (which is what stops the tutor quoting a
# cardiothoracic ratio off a projection that magnifies the heart) pushes the
# worst-case rendered system prompt to 4123 chars against the notebook's 4000
# cap — a 422, not a regression you would notice in review. The schema block is
# ~1100 chars and belongs with the answer anyway.
#
# p2.4's text is kept verbatim under `prompts/p2.4/` so the frozen Phase 1 arm
# stays byte-comparable. Its *rendered* output does change, because
# clinical_context is now real; the Phase 2 report says so rather than
# pretending the baseline is untouched.
PROMPT_VERSION: Final[str] = "p3.0"

#: The frozen Phase 1 baseline, kept loadable so Phase 2 can run it as an arm.
BASELINE_PROMPT_VERSION: Final[str] = "p2.4"

KNOWN_PROMPT_VERSIONS: Final[tuple[str, ...]] = ("p2.4", "p3.0")

_NO_CONTEXT: Final[str] = "(no clinical context provided)"

_NO_EVIDENCE: Final[str] = (
    "\nNo reference material was supplied for this case.\n"
)


@dataclass(frozen=True, slots=True)
class PromptSet:
    system: str
    user: str
    # Which template pair this came from. Defaulted so a hand-built PromptSet
    # in a test still constructs, and so a *rendered* set (the output of
    # `build`) inherits the version of the templates that produced it.
    version: str = PROMPT_VERSION

    @property
    def fingerprint(self) -> str:
        """Short content hash — identifies the exact template text used."""
        digest = hashlib.sha256(f"{self.system}\x00{self.user}".encode())
        return digest.hexdigest()[:12]


def load_default_prompts(version: str | None = None) -> PromptSet:
    """Load a versioned template pair from `kaggle_runtime/prompts/<version>/`.

    An unknown version raises rather than falling back: the whole point of
    versioning is that a report names the prompt that produced it, and a silent
    default would let a mistyped arm name be reported as a real comparison.
    """
    key = version or PROMPT_VERSION
    if key not in KNOWN_PROMPT_VERSIONS:
        known = ", ".join(KNOWN_PROMPT_VERSIONS)
        raise ValidationError(f"Unknown prompt version {key!r}. Known: {known}.")
    directory = _PROMPTS_DIR / key
    system = (directory / "system.txt").read_text(encoding="utf-8")
    user = (directory / "user.txt").read_text(encoding="utf-8")
    return PromptSet(system=system, user=user, version=key)


def render_evidence_block(passages: Sequence[RetrievedPassage]) -> str:
    """Render retrieved passages as the opaque `[S1] [S2] [S3]` block.

    Labels are positional and carry no identity: the model is told what the
    passage says, not where it came from. Handing it "WHO Manual of Diagnostic
    Imaging p.42" invites it to weigh the source's authority against the image,
    and the one thing this prompt must prevent is a passage overriding what is
    actually on the film. The mapping back to real chunk ids lives in
    `Feedback.sources`, in order.
    """
    if not passages:
        return _NO_EVIDENCE
    lines = [
        "",
        "Reference passages from published teaching texts. They describe these",
        "findings in general and were not written about this patient:",
        "",
    ]
    for index, passage in enumerate(passages, start=1):
        lines.append(f"[S{index}] {passage.text.strip()}")
        lines.append("")
    labels = ", ".join(f"[S{i}]" for i in range(1, len(passages) + 1))
    lines.extend(
        [
            f"Cite a passage as {labels} only where that passage's text supports the "
            "sentence you are writing. Those are the only labels that exist; never "
            "write one you were not shown. Citing nothing is correct when none of "
            "them bear on your point — do not reach for one to look thorough.",
            "",
        ]
    )
    return "\n".join(lines)


def _secondary_line(findings: tuple[Finding, ...]) -> str:
    """Render the additional-findings line, or nothing at all.

    An empty string rather than "none" or "(no additional findings)": on the
    two thirds of cases with a single finding, a line saying so is a prompt
    token spent telling the model something the primary already implies, and a
    place for it to start reasoning about absence.
    """
    if not findings:
        return ""
    listed = ", ".join(f"{f.label.value} ({f.category.value})" for f in findings)
    return f"Also confirmed on this film: {listed}\n"


@dataclass(frozen=True, slots=True)
class PromptBuilder:
    templates: PromptSet

    def build(
        self,
        *,
        clinical_context: str,
        student_free_text: str,
        finding: Finding,
        secondary_findings: tuple[Finding, ...] = (),
        passages: Sequence[RetrievedPassage] = (),
    ) -> PromptSet:
        """Compose the per-case prompts.

        `finding` is the case's verified ground truth, handed to the model
        rather than withheld. It is a `Finding` and not a pair of strings so
        the label and category cannot drift apart at a call site.

        `secondary_findings` are the other categories the case's labels confirm.
        They default to empty so a single-finding case renders exactly the text
        it did before.

        `passages` are the retrieved evidence. They default to empty, which
        renders the explicit "no reference material was supplied" line rather
        than a gap — the p3.0-empty control arm is exactly this call, and a
        control that differs from the treatment by an accidental blank line is
        not a control. On p2.4, whose template has no `$evidence_block`,
        `safe_substitute` drops the value and the arm is unaffected.
        """
        safe_context = clinical_context.strip() or _NO_CONTEXT
        safe_text = student_free_text.strip()
        substitutions = {
            "clinical_context": safe_context,
            "ground_truth_label": finding.label.value,
            "ground_truth_category": finding.category.value,
            "secondary_findings_line": _secondary_line(secondary_findings),
        }
        system = Template(self.templates.system).safe_substitute(**substitutions)
        # The student's answer and the passages belong in the user turn: they
        # are content rather than instruction, and the notebook caps `system`
        # at 4000 chars but `prompt` at 8000.
        user = Template(self.templates.user).safe_substitute(
            **substitutions,
            student_free_text=safe_text,
            evidence_block=render_evidence_block(passages),
        )
        return PromptSet(system=system, user=user, version=self.templates.version)

    @property
    def version(self) -> str:
        return f"{self.templates.version}+{self.templates.fingerprint}"

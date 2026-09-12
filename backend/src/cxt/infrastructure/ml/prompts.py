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
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Final

from cxt.domain.entities.finding import Finding

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
PROMPT_VERSION: Final[str] = "p2.4"

_NO_CONTEXT: Final[str] = "(no clinical context provided)"


@dataclass(frozen=True, slots=True)
class PromptSet:
    system: str
    user: str

    @property
    def fingerprint(self) -> str:
        """Short content hash — identifies the exact template text used."""
        digest = hashlib.sha256(f"{self.system}\x00{self.user}".encode())
        return digest.hexdigest()[:12]


def load_default_prompts() -> PromptSet:
    system = (_PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")
    user = (_PROMPTS_DIR / "user.txt").read_text(encoding="utf-8")
    return PromptSet(system=system, user=user)


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
    ) -> PromptSet:
        """Compose the per-case prompts.

        `finding` is the case's verified ground truth, handed to the model
        rather than withheld. It is a `Finding` and not a pair of strings so
        the label and category cannot drift apart at a call site.

        `secondary_findings` are the other categories the case's labels confirm.
        They default to empty so a single-finding case renders exactly the text
        it did before.
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
        # The student's answer belongs in the user turn: it is user content,
        # and the notebook caps `system` at 4000 chars but `prompt` at 8000.
        user = Template(self.templates.user).safe_substitute(
            **substitutions,
            student_free_text=safe_text,
        )
        return PromptSet(system=system, user=user)

    @property
    def version(self) -> str:
        return f"{PROMPT_VERSION}+{self.templates.fingerprint}"

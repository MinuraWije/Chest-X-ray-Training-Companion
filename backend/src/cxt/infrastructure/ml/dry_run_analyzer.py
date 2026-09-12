"""DryRunMedGemmaAnalyzer — IImageAnalyzer that returns canned feedback.

Used when `MEDGEMMA_RUNTIME=dry_run` (the default in dev / tests),
and by the fast unit tests. Deterministic by `attempt_id` so repeated
calls produce stable output, but each attempt gets a slightly varied
rationale so the UI demo doesn't look static.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from cxt.domain.ports.image_analyzer import ImageAnalysis
from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef

logger = logging.getLogger(__name__)


_CANNED_PAYLOADS: tuple[dict[str, object], ...] = (
    {
        "primary_finding": {
            "category": "pneumothorax",
            "label": "Right apical pneumothorax",
            "confidence": 0.82,
            "rationale": "Absent lung markings in the right upper zone with a visible visceral pleural line. No mediastinal shift.",
        },
        "differential_findings": [
            {"label": "Apical bulla", "confidence": 0.10},
            {"label": "Skin fold", "confidence": 0.05},
        ],
        "key_observations": "A thin white visceral pleural line is visible in the right apex. Lung markings are absent peripheral to the line. The costophrenic angles are sharp.",
        "teaching_point": "When you suspect a small pneumothorax, look for the visceral pleural line in the apex on an erect PA film. Expiratory films can accentuate small PTX.",
    },
    {
        "primary_finding": {
            "category": "pleural_effusion",
            "label": "Right-sided pleural effusion",
            "confidence": 0.78,
            "rationale": "Blunting of the right costophrenic angle with a meniscus sign. Left costophrenic angle is preserved.",
        },
        "differential_findings": [
            {"label": "Hemothorax", "confidence": 0.10},
            {"label": "Empyema", "confidence": 0.05},
        ],
        "key_observations": "There is blunting of the right costophrenic angle with a curvilinear upper border (meniscus) consistent with free fluid. The left costophrenic angle is sharp.",
        "teaching_point": "A meniscus sign is the hallmark of a free-flowing pleural effusion on an erect PA film. Lateral decubitus films can confirm small effusions.",
    },
    {
        "primary_finding": {
            "category": "cardiomegaly",
            "label": "Cardiomegaly",
            "confidence": 0.74,
            "rationale": "The cardiothoracic ratio appears increased on this PA view, exceeding 0.5.",
        },
        "differential_findings": [
            {"label": "Pericardial effusion", "confidence": 0.15},
        ],
        "key_observations": "The cardiac silhouette is enlarged with a cardiothoracic ratio greater than 0.5. The lung fields are otherwise clear.",
        "teaching_point": "Cardiomegaly on PA film is defined as cardiothoracic ratio > 0.5. Always confirm the film is a true PA — AP films can falsely enlarge the heart.",
    },
    {
        "primary_finding": {
            "category": "consolidation",
            "label": "Right middle lobe consolidation",
            "confidence": 0.69,
            "rationale": "There is airspace opacification in the right middle lobe with air bronchograms. No volume loss.",
        },
        "differential_findings": [
            {"label": "Lobar pneumonia", "confidence": 0.20},
            {"label": "Atelectasis", "confidence": 0.05},
        ],
        "key_observations": "Focal airspace opacification in the right mid zone with air bronchograms. The right hemidiaphragm is preserved.",
        "teaching_point": "Consolidation represents alveolar filling (pus, blood, edema, cells). Air bronchograms are a useful confirmatory sign.",
    },
    {
        "primary_finding": {
            "category": "nodule_or_mass",
            "label": "Right upper lobe nodule",
            "confidence": 0.66,
            "rationale": "A well-circumscribed rounded opacity is seen in the right upper lobe measuring approximately 2 cm.",
        },
        "differential_findings": [
            {"label": "Granuloma", "confidence": 0.20},
            {"label": "Bronchogenic carcinoma", "confidence": 0.10},
        ],
        "key_observations": "A solitary, well-circumscribed nodule is present in the right upper lobe. No cavitation, no calcification visible.",
        "teaching_point": "Compare with priors — stability over 2 years is reassuring. Solitary pulmonary nodules need workup based on size, margins, and patient risk factors.",
    },
    {
        "primary_finding": {
            "category": "fracture",
            "label": "Left lateral rib fracture",
            "confidence": 0.71,
            "rationale": "Cortical disruption in the left 7th lateral rib. No displaced fragment identified.",
        },
        "differential_findings": [
            {"label": "Healed rib fracture", "confidence": 0.10},
        ],
        "key_observations": "A cortical break is visible in the left 7th lateral rib without significant displacement. No pneumothorax or haemothorax.",
        "teaching_point": "Always look for associated complications of rib fractures: pneumothorax, haemothorax, and underlying lung or solid organ injury.",
    },
    {
        "primary_finding": {
            "category": "no_finding",
            "label": "No acute finding",
            "confidence": 0.80,
            "rationale": "The lung fields are clear, the cardiac silhouette is normal, and there is no pleural effusion or pneumothorax.",
        },
        "differential_findings": [],
        "key_observations": "Lungs are clear bilaterally. Cardiomediastinal silhouette is normal. Costophrenic angles are sharp. Bony thorax is intact.",
        "teaching_point": "A normal CXR report should explicitly comment on the heart, lungs, pleura, mediastinum, and bones. 'Normal' is not enough.",
    },
)


@dataclass(frozen=True, slots=True)
class DryRunMedGemmaAnalyzer:
    """Returns deterministic canned feedback for dev / tests.

    The canned payload is selected by `hash(attempt_id) % len(payloads)`.
    """

    def analyze(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ImageAnalysis:
        seed = int(hashlib.sha256(image.uri.encode("utf-8")).hexdigest(), 16)
        payload = _CANNED_PAYLOADS[seed % len(_CANNED_PAYLOADS)]
        import json

        text = json.dumps(payload, indent=2)
        return ImageAnalysis(
            raw_text=text,
            model_output=ModelOutput(
                text=text,
                prompt_tokens=len(prompt.split()) + len(system.split()),
                output_tokens=len(text.split()),
                model="dry-run",
            ),
        )

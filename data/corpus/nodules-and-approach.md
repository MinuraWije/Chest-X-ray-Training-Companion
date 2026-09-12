---
source: "Chest X-ray Training Companion — authored teaching note"
url: "https://github.com/MinuraWije/Chest-X-ray-Training-Companion/blob/master/data/corpus/nodules-and-approach.md"
licence: "CC BY-NC-SA 3.0"
licence_url: "https://creativecommons.org/licenses/by-nc-sa/3.0/"
retrieved_on: "2026-09-12"
---

## Nodule, mass, and why the size threshold matters

```yaml
id: nodule-definitions
findings: [nodule_or_mass]
confusions: [consolidation]
aspect: radiographic_sign
```

A pulmonary nodule is a rounded opacity up to 30 mm in diameter surrounded by
aerated lung. Above 30 mm the same lesion is called a mass, and the distinction
is not pedantry: the larger the lesion, the higher the probability it is
malignant, and masses are investigated on that basis. Both are discrete and
have definable margins, which separates them from consolidation, whose edges
fade into surrounding lung and which follows lobar anatomy rather than forming
a ball.

## Margins and what they suggest

```yaml
id: nodule-margins
findings: [nodule_or_mass]
confusions: [nodule_or_mass]
aspect: differential
```

A smooth, well-circumscribed margin is more often benign — a granuloma, a
hamartoma, a cyst. Spiculated or irregular margins, with strands radiating into
the surrounding lung, raise suspicion of malignancy. Lobulated contours sit in
between and suggest uneven growth. Margin analysis on a plain film is a weak
discriminator on its own and is not a substitute for CT, so it should inform how
urgently a lesion is worked up rather than whether it is worked up at all.

## Calcification patterns

```yaml
id: nodule-calcification
findings: [nodule_or_mass]
confusions: [nodule_or_mass]
aspect: differential
```

Calcification within a nodule is reassuring only when its pattern is one of the
benign ones: diffuse and dense, central, laminated or concentric, or the
"popcorn" pattern of a hamartoma. Stippled, eccentric or amorphous calcification
does not exclude malignancy, since a tumour can engulf a pre-existing
granuloma. Plain radiographs detect calcification poorly, so its apparent
absence on a chest film means very little.

## Nodule mimics outside the lung

```yaml
id: nodule-mimics
findings: [nodule_or_mass, no_finding]
confusions: [nodule_or_mass]
aspect: pitfall
```

Several things project over the lungs and imitate a nodule: nipple shadows,
which are usually bilateral, symmetrical and at the same level in the lower
zones; skin lesions such as moles and seborrhoeic keratoses, which have
sharp margins because they are outlined by air; healing or old rib fractures
and bone islands; and vessels seen end-on, which can be traced into a branching
vessel on careful inspection. Before reporting a solitary nodule, check whether
the opacity is truly intrapulmonary.

## A systematic approach to the frontal chest film

```yaml
id: approach-systematic-review
findings: [no_finding, pneumothorax, pleural_effusion, cardiomegaly, consolidation, nodule_or_mass]
confusions: []
aspect: technique
```

Read the same way every time so that findings are not missed through
inattention. Confirm the patient, the date and the projection. Assess technical
adequacy: rotation, inspiration, penetration. Then work through airway,
breathing (lung fields zone by zone, comparing left with right), circulation
(heart size and contour, mediastinum, hila), diaphragm and costophrenic angles,
and everything else — bones, soft tissues, lines and tubes. Review areas that
are commonly missed last: apices, behind the heart, below the diaphragm, and the
periphery.

## Assessing technical adequacy first

```yaml
id: approach-technical-adequacy
findings: [no_finding, cardiomegaly]
confusions: [cardiomegaly, pleural_effusion]
aspect: technique
```

Rotation is judged by the medial ends of the clavicles relative to the spinous
processes; rotation distorts the mediastinal contour and can create apparent
cardiomegaly or a false hilar abnormality. Adequate inspiration puts the
diaphragm at about the level of the tenth posterior rib or the sixth anterior
rib; a shallow inspiration crowds the lung bases and widens the heart. Correct
penetration allows the lower thoracic vertebrae to be seen faintly through the
heart. Stating the projection and adequacy before describing findings prevents
most projection-related errors.

## Commonly missed review areas

```yaml
id: approach-review-areas
findings: [no_finding, pneumothorax, nodule_or_mass]
confusions: []
aspect: pitfall
```

A large proportion of missed findings on chest radiographs lie in a small number
of places: the lung apices, obscured by clavicles and ribs; the retrocardiac
region, where a left lower lobe lesion hides behind the heart; the area below
the diaphragmatic domes; the periphery just inside the chest wall, where a
small pneumothorax or a peripheral nodule sits; and the bones and soft tissues,
which are easy to skip once a lung abnormality has been found. Satisfaction of
search — stopping after the first abnormality — is the single commonest reason a
second finding is missed.

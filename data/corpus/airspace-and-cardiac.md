---
source: "Chest X-ray Training Companion — authored teaching note"
url: "https://github.com/MinuraWije/Chest-X-ray-Training-Companion/blob/master/data/corpus/airspace-and-cardiac.md"
licence: "CC BY-NC-SA 3.0"
licence_url: "https://creativecommons.org/licenses/by-nc-sa/3.0/"
retrieved_on: "2026-09-12"
---

## Air bronchograms

```yaml
id: airspace-air-bronchogram
findings: [consolidation]
confusions: [pleural_effusion, nodule_or_mass]
aspect: radiographic_sign
```

An air bronchogram is a branching, air-filled bronchus made visible because the
alveoli around it have filled with fluid, pus, blood or cells while the airway
itself stays patent. Normally bronchi are invisible beyond the hila, since air
inside an airway surrounded by air has no contrast. Seeing dark branching tubes
within an area of increased density is strong evidence that the opacity is
within the lung rather than in the pleural space, which is what separates
consolidation from a pleural effusion or a pleural mass.

## The silhouette sign

```yaml
id: airspace-silhouette-sign
findings: [consolidation]
confusions: [pleural_effusion, cardiomegaly]
aspect: radiographic_sign
```

Two structures of similar density in direct anatomical contact lose the border
between them. A consolidated right middle lobe abuts the right heart border and
erases it; a right lower lobe consolidation sits posteriorly, leaves the heart
border crisp, and obscures the hemidiaphragm instead. Lingular disease erases
the left heart border, left lower lobe disease the left hemidiaphragm. The sign
is therefore a localiser: which border has disappeared tells you which lobe is
involved, without a lateral film.

## Consolidation does not mean infection

```yaml
id: airspace-consolidation-differential
findings: [consolidation]
confusions: [nodule_or_mass, pleural_effusion]
aspect: differential
```

Consolidation is a description of alveolar filling, not a diagnosis. Pneumonia
is the commonest cause, but pulmonary oedema, haemorrhage, aspiration,
infarction, contusion and some malignancies produce the same appearance.
Distribution helps: perihilar and bilaterally symmetrical suggests oedema; a
single lobe with a fissural boundary suggests lobar pneumonia; peripheral and
wedge-shaped raises infarction. A student who writes "consolidation" and
describes its distribution is on firmer ground than one who writes "pneumonia".

## Distinguishing consolidation from collapse

```yaml
id: airspace-consolidation-vs-collapse
findings: [consolidation]
confusions: [consolidation, pneumothorax]
aspect: differential
```

Both produce increased density. Consolidation fills the lung but preserves its
volume, so the fissures stay in place and the mediastinum does not move.
Collapse loses volume, so the fissure displaces toward the affected area, the
hemidiaphragm rises, the hilum is pulled up or down, and the mediastinum shifts
toward the opacity. Volume loss is the discriminator, and the position of the
mediastinum is the easiest place to look for it.

## Measuring the cardiothoracic ratio

```yaml
id: cardiac-ctr-measurement
findings: [cardiomegaly]
confusions: [pleural_effusion, consolidation]
aspect: radiographic_sign
```

The cardiothoracic ratio is the maximum transverse cardiac diameter divided by
the maximum internal transverse diameter of the thorax, measured on the same
erect PA film at full inspiration. A ratio at or below 0.50 is normal. Above
0.50 on an adequate PA film suggests cardiomegaly, and the degree matters less
than the trend across serial films. The measurement is a ratio of two widths on
one image, so both must be taken at their widest points, not estimated by eye.

## Why AP projection invalidates the ratio

```yaml
id: cardiac-ap-projection-pitfall
findings: [cardiomegaly]
confusions: [cardiomegaly]
aspect: pitfall
```

On a PA film the heart lies close to the detector and is projected near its true
size. On an AP film — which is what portable and supine radiographs are — the
X-ray source is in front and the heart is further from the detector, so it is
magnified, typically by 15 to 20 per cent. A low lung volume from a shallow
supine inspiration widens the cardiac silhouette further. A heart that looks
enlarged on an AP film may be entirely normal, so the correct report is that
cardiac size cannot be assessed on this projection, rather than a number treated
as if it were diagnostic.

## Cardiomegaly versus an apparently large heart

```yaml
id: cardiac-mimics
findings: [cardiomegaly]
confusions: [cardiomegaly, pleural_effusion, consolidation]
aspect: differential
```

Several things widen the cardiac silhouette without the heart being enlarged: a
pericardial effusion, which gives a smooth globular "water-bottle" outline; a
large anterior mediastinal mass abutting the heart; pectus excavatum, which
compresses and displaces it; and adjacent consolidation or effusion merging with
the cardiac border through the silhouette effect. Before calling cardiomegaly,
check the projection, the inspiration, and whether the border you are measuring
is actually heart.

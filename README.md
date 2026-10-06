# Évaluer le codage d'images 360° sur la sphère

Ce projet réimplémente le cadre d'évaluation proposé par **Yu, Lakshman et Girod** dans
*A Framework to Evaluate Omnidirectional Video Coding Schemes* (IEEE ISMAR 2015, Stanford),
et l'applique au codec **HEVC** (x265, codage intra).

## Le problème

Une image 360° est une sphère, mais les codecs ne savent coder que des rectangles. On la
projette donc sur un plan, le plus souvent en **équirectangulaire (ERP)**, qui suréchantillonne
fortement les pôles (facteur 1/cos φ). Deux conséquences :

1. le **PSNR calculé sur le plan** surpondère les pôles, que l'utilisateur regarde peu ;
2. le codec dépense des bits sur des zones sur-représentées.

L'article propose de mesurer la qualité **sur ce que voit l'utilisateur** (PSNR de viewport,
à partir des mouvements de tête) et montre qu'un **S-PSNR pondéré** par des statistiques de
mouvements de tête l'approche sans connaître la trajectoire exacte. Il compare ensuite
plusieurs projections avec ces métriques.

## Ce qui est implémenté

| Étape | Contenu | Fichiers |
|---|---|---|
| 1. Projections | ERP, equal-area (EAP), dyadique, cubemap (CMP) ; génération par la sphère (Fig. 2), interpolation bicubique | `omni360/projections.py`, `omni360/sampling.py` |
| 2. Métriques | PSNR de viewport (Éq. 1 à 3), S-PSNR, S-PSNR pondérés (WeightSph, LatSph), Quad, BD-rate | `omni360/viewport.py`, `omni360/metrics.py`, `omni360/bdrate.py` |
| 3. HEVC | x265 intra aux QP 22, 27, 32, 37 sur chaque projection (Tableau 1, Fig. 6) | `scripts/run_hevc.py`, `omni360/hevc.py` |
| Analyse | tableaux BD-rate (Tableaux 1 et 2), courbes débit/distorsion (Fig. 6 et 7) | `scripts/report.py` |

## Installation (Windows)

```powershell
uv venv --python 3.12 .venv        # ou : py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                              # 32 tests
```

FFmpeg avec x265 est fourni par `imageio-ffmpeg`, rien d'autre à installer.

## Données

Placez des panoramas **équirectangulaires 2:1** (4K ou plus) dans `data/test/`. Les HDRI de
[Poly Haven](https://polyhaven.com/hdris) (licence CC0) existent en JPG tonemappé et
conviennent bien. Une dizaine d'images suffit.

Sans données, le mot `synthetic` remplace un fichier : il génère une mire de test.

## Utilisation

```powershell
# 1. Projections, viewports, Fig. 1 et Fig. 5  ->  results/demo/
python scripts/demo_projections.py data/test/mon_panorama.jpg

# 3. HEVC sur les 4 projections  ->  results/hevc.csv
python scripts/run_hevc.py data/test

# Tableaux BD-rate et courbes  ->  results/report.md, results/rd_*.png
python scripts/report.py results/hevc.csv --reference hevc:erp
```

## Choix et limites

* **Images fixes** (codage intra) plutôt que vidéos, pour garder des temps de calcul
  raisonnables ; le cadre d'évaluation est le même.
* **Mouvements de tête synthétiques** (lacet uniforme, tangage gaussien autour de l'horizon).
  L'article utilise de vraies trajectoires (10 utilisateurs, Oculus Rift DK2). Des données
  réelles comme Salient360! peuvent être passées sous forme de tableau (lacet, tangage, roulis).
* **Résolutions réduites** par défaut : vérité terrain 3072×1536 et images codées 2048×1024
  (l'article : 6K×3K et 4K×2K). Le rapport 1,5 entre les deux est conservé.
* **Projection dyadique** : les deux bandes polaires (à demi-résolution horizontale) sont
  rangées côte à côte au-dessus de la bande équatoriale, soit 82 % des pixels de l'ERP.
* **Cubemap** en 3×2, avec un budget de pixels égal à celui de l'ERP.
* Le **débit** est exprimé en bits par pixel de la grille ERP codée, identique pour toutes
  les projections afin de pouvoir les comparer.

## Résultats

*À compléter après les expériences sur de vraies images* (tableaux de `results/report.md` et
courbes `results/rd_*.png`).

## Références

* M. Yu, H. Lakshman, B. Girod, *A Framework to Evaluate Omnidirectional Video Coding Schemes*, IEEE ISMAR 2015.
* G. Bjøntegaard, *Calculation of average PSNR differences between RD-curves*, ITU-T VCEG-M33, 2001.

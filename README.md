# INF8770 – TP3 : Recherche d'images par contenu pictural

**Polytechnique Montréal | Automne 2026**

## Description

Ce dépôt contient le travail pratique 3 du cours **INF8770 – Technologies multimédias**.

L'objectif du projet est de concevoir et d'évaluer un système de recherche d'images par le contenu pictural, en comparant deux approches de description visuelle :

- **Description globale :** histogrammes de couleurs RGB et histogrammes de chrominance Cb-Cr (YCbCr).
- **Description locale :** détection et mise en correspondance de points d'intérêt SIFT (*Scale-Invariant Feature Transform*).

Le système compare quatre images requêtes à une galerie de 32 images candidates afin d'identifier les images les plus pertinentes malgré différentes transformations visuelles.

## Objectifs

- Implémenter des histogrammes de couleurs RGB et Cb-Cr.
- Détecter et comparer des points d'intérêt SIFT.
- Étudier l'influence de différents seuils de correspondance SIFT.
- Classer les images candidates selon leur similarité avec une requête.
- Évaluer la précision et le rappel aux rangs Top-5 et Top-10.
- Mesurer et comparer les temps d'exécution.
- Analyser les performances et les limites des différentes approches.
- Proposer une méthode hybride combinant rapidité et précision.

## Données

Le jeu de données fourni comprend :

- **4 images requêtes** présentant différentes perturbations visuelles.
- **32 images candidates** dans la galerie.
- **5 images pertinentes par requête.**
- **12 images distractrices.**
- Un fichier `verite_terrain.csv` permettant l'évaluation quantitative des résultats.

Les requêtes portent sur quatre situations :

| Requête | Description |
|---|---|
| A | Scène de motocross avec zooms et rotations |
| B | Chalet en montagne avec variations d'exposition |
| C | Monoplace Mercedes F1 avec changements d'arrière-plan |
| D | Composition lumineuse floue (bokeh) avec variations spatiales |

## Méthodologie

### 1. Histogrammes de couleurs

Extraction, quantification et normalisation des histogrammes RGB et Cb-Cr, suivies d'une comparaison des images à l'aide d'une mesure de distance.

### 2. Points d'intérêt SIFT

Extraction des caractéristiques locales et mise en correspondance des descripteurs par distance euclidienne (L2), avec validation croisée.

Trois seuils de distance seront étudiés : strict, modéré et permissif.

### 3. Évaluation expérimentale

Les méthodes seront comparées à partir des métriques suivantes :

- Précision@5 et Précision@10
- Rappel@5 et Rappel@10
- Temps moyen d'exécution par requête
- Nombre de points d'intérêt SIFT détectés

## Organisation du projet

```text
INF8770-TP3/
├── donnees_tp3/
│   ├── requetes/
│   ├── galerie/
│   └── verite_terrain.csv
├── rapport/
│   └── figures/
├── src/
├── README.md
└── .gitignore
```

La structure pourra évoluer pendant le développement. Le code final sera regroupé dans un seul fichier Python ou notebook conformément aux consignes de remise.

## Technologies

- Python 3
- OpenCV
- NumPy
- Matplotlib

## Organisation du travail

Le projet est réalisé en équipe avec Git et GitHub.

- `main` : branche principale contenant les versions validées.
- `feature/...` : branches dédiées au développement.
- Pull Requests : intégration et révision des modifications.

## Livrables

**Date limite : 30 novembre 2026 à 17 h.**

- **Gradescope :** rapport PDF de 10 pages maximum, annexes incluses.
- **Moodle :** archive contenant le rapport PDF et l'ensemble du code source dans un seul fichier `.py` ou `.ipynb`.

Les outils, bibliothèques et éventuelles utilisations d'IA générative seront déclarés et référencés dans le rapport conformément aux consignes du cours.

---

**INF8770 – Technologies multimédias**

Département de génie informatique et génie logiciel

Polytechnique Montréal
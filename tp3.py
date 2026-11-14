"""
INF8770 - TP3: Image retrieval by visual content
Global description (RGB / Cb-Cr histograms) vs local description (SIFT).

Usage:
    python tp3.py
The script reads donnees_tp3/ (next to this file) and writes all tables
and figures to resultats/.

Dependencies: Python 3, OpenCV (>= 4.4, cv2.SIFT_create), NumPy, Matplotlib.
"""

import csv
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Experiment parameters (identical for the four queries)
# --------------------------------------------------------------------------
ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DONNEES = os.path.join(ICI, "donnees_tp3")
DOSSIER_REQUETES = os.path.join(DOSSIER_DONNEES, "requetes")
DOSSIER_GALERIE = os.path.join(DOSSIER_DONNEES, "galerie")
FICHIER_VERITE = os.path.join(DOSSIER_DONNEES, "verite_terrain.csv")
DOSSIER_RESULTATS = os.path.join(ICI, "resultats")

NB_NIVEAUX = 16                      # quantization intervals per channel
SEUILS_SIFT = {"strict": 100.0,      # maximum L2 distance for a match
               "modéré": 200.0,
               "permissif": 300.0}
RANGS_K = (5, 10)
NB_REPETITIONS_TEMPS = 5             # average timing over multiple runs

METHODES = ["Histo RGB", "Histo Cb-Cr"] + [f"SIFT T={int(t)} ({n})"
                                            for n, t in SEUILS_SIFT.items()]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def lister_png(dossier):
    return sorted(f for f in os.listdir(dossier) if f.lower().endswith(".png"))


def charger_images(dossier):
    """Return {name: BGR image} (cv2.imread loads images in BGR, not RGB)."""
    images = {}
    for nom in lister_png(dossier):
        img = cv2.imread(os.path.join(dossier, nom), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(os.path.join(dossier, nom))
        images[nom] = img
    return images


def charger_verite_terrain(chemin):
    """Used only for evaluation, never for retrieval."""
    verite = {}
    with open(chemin, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            verite[ligne["requete"]] = set(ligne["images_pertinentes"].split(";"))
    return verite


# --------------------------------------------------------------------------
# Global approach: color histograms
# --------------------------------------------------------------------------
def histogramme_canal(canal, nb_niveaux):
    """Histogram of an 8-bit channel quantized into nb_niveaux equal intervals,
    normalized (sum = 1) to be independent of the number of pixels."""
    h = cv2.calcHist([canal], [0], None, [nb_niveaux], [0, 256]).ravel()
    return h / h.sum()


def descripteur_rgb(img_bgr, nb_niveaux=NB_NIVEAUX):
    """Concatenate R | G | B (restore the channels to RGB order)."""
    b, g, r = cv2.split(img_bgr)
    return np.concatenate([histogramme_canal(c, nb_niveaux) for c in (r, g, b)])


def descripteur_cbcr(img_bgr, nb_niveaux=NB_NIVEAUX):
    """Chrominance only. COLOR_BGR2YCrCb -> channel 0 = Y, 1 = Cr, 2 = Cb.
    Discard Y and concatenate Cb | Cr."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return np.concatenate([histogramme_canal(c, nb_niveaux) for c in (cb, cr)])


def distance_l2(h1, h2):
    return float(np.linalg.norm(h1 - h2))


def classer_histogrammes(desc_requete, desc_galerie):
    """Return [(name, distance)] sorted by increasing distance
    (smaller distance = more similar)."""
    scores = [(nom, distance_l2(desc_requete, d)) for nom, d in desc_galerie.items()]
    return sorted(scores, key=lambda x: (x[1], x[0]))


# --------------------------------------------------------------------------
# Local approach: SIFT keypoints
# --------------------------------------------------------------------------
SIFT = cv2.SIFT_create()
APPARIEUR = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


def extraire_sift(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    points, descripteurs = SIFT.detectAndCompute(gris, None)
    return points, descripteurs


def apparier(desc_q, desc_g):
    """L2 matches with cross-validation (crossCheck=True)."""
    if desc_q is None or desc_g is None or len(desc_q) == 0 or len(desc_g) == 0:
        return []
    return APPARIEUR.match(desc_q, desc_g)


def classer_sift(desc_requete, desc_galerie, seuil):
    """Score = number of matches with distance < threshold.
    Sort by decreasing score (more matches = more similar)."""
    scores = []
    for nom, d in desc_galerie.items():
        corr = apparier(desc_requete, d)
        scores.append((nom, sum(1 for m in corr if m.distance < seuil)))
    return sorted(scores, key=lambda x: (-x[1], x[0]))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def precision_a_k(classement, pertinents, k):
    top = [nom for nom, _ in classement[:k]]
    return sum(1 for n in top if n in pertinents) / k


def rappel_a_k(classement, pertinents, k):
    top = [nom for nom, _ in classement[:k]]
    return sum(1 for n in top if n in pertinents) / len(pertinents)


# --------------------------------------------------------------------------
# Experiment execution
# --------------------------------------------------------------------------
def executer_requete(methode, img_q, index):
    """Compute the query descriptor and then rank the gallery.
    The index (gallery descriptors) is precomputed offline."""
    if methode == "Histo RGB":
        return classer_histogrammes(descripteur_rgb(img_q), index["rgb"])
    if methode == "Histo Cb-Cr":
        return classer_histogrammes(descripteur_cbcr(img_q), index["cbcr"])
    seuil = float(methode.split("T=")[1].split(" ")[0])
    _, desc_q = extraire_sift(img_q)
    return classer_sift(desc_q, index["sift"], seuil)


def construire_index(galerie):
    t0 = time.perf_counter()
    rgb = {n: descripteur_rgb(im) for n, im in galerie.items()}
    t_rgb = time.perf_counter() - t0
    t0 = time.perf_counter()
    cbcr = {n: descripteur_cbcr(im) for n, im in galerie.items()}
    t_cbcr = time.perf_counter() - t0
    t0 = time.perf_counter()
    sift_kp, sift = {}, {}
    for n, im in galerie.items():
        sift_kp[n], sift[n] = extraire_sift(im)
    t_sift = time.perf_counter() - t0
    temps = {"Histo RGB": t_rgb, "Histo Cb-Cr": t_cbcr, "SIFT": t_sift}
    return {"rgb": rgb, "cbcr": cbcr, "sift": sift, "sift_kp": sift_kp}, temps


def experience(requetes, index, verite):
    resultats = []      # one row per (query, method)
    classements = {}    # (query, method) -> complete ranking
    for nom_q, img_q in requetes.items():
        pertinents = verite[nom_q]
        for methode in METHODES:
            executer_requete(methode, img_q, index)  # warm-up (not timed)
            durees = []
            for _ in range(NB_REPETITIONS_TEMPS):
                t0 = time.perf_counter()
                classement = executer_requete(methode, img_q, index)
                durees.append(time.perf_counter() - t0)
            classements[(nom_q, methode)] = classement
            resultats.append({
                "requete": nom_q,
                "methode": methode,
                "P@5": precision_a_k(classement, pertinents, 5),
                "P@10": precision_a_k(classement, pertinents, 10),
                "R@5": rappel_a_k(classement, pertinents, 5),
                "R@10": rappel_a_k(classement, pertinents, 10),
                "temps_ms": 1000 * float(np.mean(durees)),
                "temps_std_ms": 1000 * float(np.std(durees)),
                "top10": " ".join(n[4:6] for n, _ in classement[:10]),
            })
    return resultats, classements


def distribution_distances_sift(requetes, index):
    """Distribution of the L2 distances of all crossCheck matches between each
    query and the whole gallery. Used to choose the three thresholds T
    (strict / moderate / permissive) WITHOUT the ground truth."""
    lignes = []
    for nom_q, img_q in requetes.items():
        _, desc_q = extraire_sift(img_q)
        d = np.array([m.distance for desc_g in index["sift"].values()
                      for m in apparier(desc_q, desc_g)])
        ligne = {"requete": nom_q, "nb_correspondances": len(d)}
        if len(d):
            ligne.update({"centile_5": float(np.percentile(d, 5)),
                          "mediane": float(np.median(d)),
                          **{f"part_sous_T{int(t)}": float(np.mean(d < t))
                             for t in SEUILS_SIFT.values()}})
        lignes.append(ligne)
    return lignes


# --------------------------------------------------------------------------
# Outputs: CSV files, tables, and figures
# --------------------------------------------------------------------------
def ecrire_csv(chemin, lignes):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)


def figure_tableau(resultats, chemin):
    entetes = ["Requête", "Méthode", "P@5", "P@10", "R@5", "R@10", "Temps (ms)"]
    cellules = [[r["requete"][8], r["methode"], f"{r['P@5']:.2f}", f"{r['P@10']:.2f}",
                 f"{r['R@5']:.2f}", f"{r['R@10']:.2f}", f"{r['temps_ms']:.1f}"]
                for r in resultats]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(cellules) + 0.6))
    ax.axis("off")
    t = ax.table(cellText=cellules, colLabels=entetes, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.15)
    for i, r in enumerate(resultats, start=1):
        if ord(r["requete"][8]) % 2 == 0:
            for j in range(len(entetes)):
                t[i, j].set_facecolor("#eef2f7")
    fig.savefig(chemin, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_seuils_sift(resultats, chemin):
    """Impact of the three thresholds T on P@5, P@10, and R@10, by query."""
    noms_q = sorted({r["requete"] for r in resultats})
    meth_sift = [m for m in METHODES if m.startswith("SIFT")]
    seuils = list(SEUILS_SIFT.values())
    styles = [("o", "-", 9), ("s", "--", 7), ("^", ":", 6), ("D", "-.", 5)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, metrique in zip(axes, ["P@5", "P@10", "R@10"]):
        for q, (mk, ls, ms) in zip(noms_q, styles):
            ys = [next(r[metrique] for r in resultats
                       if r["requete"] == q and r["methode"] == m) for m in meth_sift]
            ax.plot(seuils, ys, marker=mk, linestyle=ls, markersize=ms, linewidth=2,
                    alpha=0.85, label=f"Requête {q[8]}")
        moy = [np.mean([r[metrique] for r in resultats if r["methode"] == m]) for m in meth_sift]
        ax.plot(seuils, moy, color="k", linestyle="--", marker="X", markersize=8, label="Moyenne")
        ax.set_xticks(seuils)
        ax.set_xticklabels([f"{int(t)}\n({n})" for n, t in SEUILS_SIFT.items()])
        ax.set_title(metrique)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Valeur de la métrique (0 à 1)")
    fig.supxlabel("Seuil de distance L2 maximale T (unités du descripteur SIFT, 128 dimensions)",
                  fontsize=10)
    poignees, etiquettes = axes[0].get_legend_handles_labels()
    fig.legend(poignees, etiquettes, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(chemin, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_comparaison_methodes(resultats, chemin, metrique="P@5"):
    noms_q = sorted({r["requete"] for r in resultats})
    x = np.arange(len(noms_q))
    largeur = 0.8 / len(METHODES)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, m in enumerate(METHODES):
        ys = [next(r[metrique] for r in resultats if r["requete"] == q and r["methode"] == m)
              for q in noms_q]
        ax.bar(x + (i - (len(METHODES) - 1) / 2) * largeur, ys, largeur, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Requête {q[8]}" for q in noms_q])
    ax.set_ylabel(f"{metrique} (proportion, 0 à 1)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=200)
    plt.close(fig)


def figure_temps(resultats, temps_index, chemin):
    moy = [np.mean([r["temps_ms"] for r in resultats if r["methode"] == m]) for m in METHODES]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    barres = ax.bar(range(len(METHODES)), moy, color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xticks(range(len(METHODES)))
    ax.set_xticklabels(METHODES, rotation=15, fontsize=8)
    ax.set_ylabel("Temps moyen par requête (ms, échelle log)")
    for b, v in zip(barres, moy):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(chemin, dpi=200)
    plt.close(fig)


def figure_top5(nom_q, img_q, classement, galerie, pertinents, titre, chemin, unite):
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.6))
    axes[0].imshow(cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Requête {nom_q[8]}", fontsize=10, fontweight="bold")
    for ax, (nom, score) in zip(axes[1:], classement[:5]):
        ax.imshow(cv2.cvtColor(galerie[nom], cv2.COLOR_BGR2RGB))
        ok = nom in pertinents
        couleur = "green" if ok else "red"
        for s in ax.spines.values():
            s.set_edgecolor(couleur)
            s.set_linewidth(4)
        val = f"{score:.3f}" if isinstance(score, float) else str(score)
        ax.set_title(f"{nom[:-4]}  {unite}={val}\n{'pertinente' if ok else 'non pertinente'}",
                     fontsize=8, color=couleur)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_correspondances(img_q, img_g, seuil, titre, chemin, max_traits=80):
    kp_q, d_q = extraire_sift(img_q)
    kp_g, d_g = extraire_sift(img_g)
    corr = sorted([m for m in apparier(d_q, d_g) if m.distance < seuil],
                  key=lambda m: m.distance)
    vis = cv2.drawMatches(img_q, kp_q, img_g, kp_g, corr[:max_traits], None,
                          matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{titre}\n{len(corr)} correspondances < T={int(seuil)} "
                 f"({min(len(corr), max_traits)} tracées)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_points_interet(requetes, chemin):
    fig, axes = plt.subplots(1, len(requetes), figsize=(16, 3.2))
    for ax, (nom, img) in zip(axes, requetes.items()):
        kp, _ = extraire_sift(img)
        vis = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Requête {nom[8]} : {len(kp)} points SIFT", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def figure_histogrammes(nom_q, img_q, noms_g, galerie, chemin):
    """RGB and Cb-Cr histograms for a query and a few gallery images."""
    images = [(f"Requête {nom_q[8]}", img_q)] + [(n[:-4], galerie[n]) for n in noms_g]
    fig, axes = plt.subplots(2, len(images), figsize=(3.2 * len(images), 4.4))
    for j, (titre, img) in enumerate(images):
        for i, (fct, etiquettes, couleurs) in enumerate((
                (descripteur_rgb, ["R", "G", "B"], ["r", "g", "b"]),
                (descripteur_cbcr, ["Cb", "Cr"], ["tab:blue", "tab:red"]))):
            h = fct(img).reshape(len(etiquettes), NB_NIVEAUX)
            for k, (lab, c) in enumerate(zip(etiquettes, couleurs)):
                axes[i, j].plot(np.arange(NB_NIVEAUX), h[k], color=c, label=lab)
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_xlabel("Intervalle de quantification", fontsize=7)
            if j == 0:
                axes[i, j].set_ylabel("Proportion de pixels", fontsize=7)
                axes[i, j].legend(fontsize=7)
        axes[0, j].set_title(titre, fontsize=9)
    fig.tight_layout()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main program
# --------------------------------------------------------------------------
def main():
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    print(f"OpenCV {cv2.__version__}, NumPy {np.__version__}")

    requetes = charger_images(DOSSIER_REQUETES)
    galerie = charger_images(DOSSIER_GALERIE)
    verite = charger_verite_terrain(FICHIER_VERITE)
    assert len(galerie) == 32 and len(requetes) == 4

    index, temps_index = construire_index(galerie)
    print("Temps d'indexation de la galerie (s) :",
          {k: round(v, 3) for k, v in temps_index.items()})

    # Number of SIFT keypoints
    kp_lignes = [{"image": n, "type": "requete", "nb_points_sift": len(extraire_sift(im)[0])}
                 for n, im in requetes.items()]
    kp_lignes += [{"image": n, "type": "galerie", "nb_points_sift": len(index["sift_kp"][n])}
                  for n in galerie]
    ecrire_csv(os.path.join(DOSSIER_RESULTATS, "points_sift.csv"), kp_lignes)
    for l in kp_lignes[:4]:
        print(f"  {l['image']} : {l['nb_points_sift']} points SIFT")

    # Distribution of SIFT match distances (choice of T, without ground truth)
    dist = distribution_distances_sift(requetes, index)
    ecrire_csv(os.path.join(DOSSIER_RESULTATS, "distribution_distances_sift.csv"), dist)
    print("\nDistances des correspondances SIFT (requête vs galerie) :")
    for l in dist:
        if "mediane" in l:
            print(f"  {l['requete']} : médiane = {l['mediane']:.0f}, "
                  f"5e centile = {l['centile_5']:.0f}, "
                  f"part < 300 = {l['part_sous_T300']:.2f}")

    resultats, classements = experience(requetes, index, verite)
    ecrire_csv(os.path.join(DOSSIER_RESULTATS, "resultats.csv"), resultats)

    # Averages by method
    moyennes = []
    for m in METHODES:
        rs = [r for r in resultats if r["methode"] == m]
        moyennes.append({"methode": m, **{k: float(np.mean([r[k] for r in rs]))
                                          for k in ("P@5", "P@10", "R@5", "R@10", "temps_ms")}})
    ecrire_csv(os.path.join(DOSSIER_RESULTATS, "moyennes.csv"), moyennes)
    ecrire_csv(os.path.join(DOSSIER_RESULTATS, "temps_indexation.csv"),
               [{"methode": k, "temps_total_s": v, "temps_par_image_ms": 1000 * v / len(galerie)}
                for k, v in temps_index.items()])

    print(f"\n{'Req':<4}{'Méthode':<26}{'P@5':>6}{'P@10':>6}{'R@5':>6}{'R@10':>6}{'ms':>9}  Top-10")
    for r in resultats:
        print(f"{r['requete'][8]:<4}{r['methode']:<26}{r['P@5']:>6.2f}{r['P@10']:>6.2f}"
              f"{r['R@5']:>6.2f}{r['R@10']:>6.2f}{r['temps_ms']:>9.1f}  {r['top10']}")
    print("\nMoyennes :")
    for m in moyennes:
        print(f"    {m['methode']:<26}{m['P@5']:>6.2f}{m['P@10']:>6.2f}{m['R@5']:>6.2f}"
              f"{m['R@10']:>6.2f}{m['temps_ms']:>9.1f}")

    # Figures
    figure_tableau(resultats, os.path.join(DOSSIER_RESULTATS, "tableau_resultats.png"))
    figure_seuils_sift(resultats, os.path.join(DOSSIER_RESULTATS, "impact_seuil_sift.png"))
    figure_comparaison_methodes(resultats, os.path.join(DOSSIER_RESULTATS, "comparaison_P5.png"), "P@5")
    figure_comparaison_methodes(resultats, os.path.join(DOSSIER_RESULTATS, "comparaison_P10.png"), "P@10")
    figure_temps(resultats, temps_index, os.path.join(DOSSIER_RESULTATS, "temps_requete.png"))
    figure_points_interet(requetes, os.path.join(DOSSIER_RESULTATS, "points_sift_requetes.png"))

    dossier_top5 = os.path.join(DOSSIER_RESULTATS, "top5")
    os.makedirs(dossier_top5, exist_ok=True)
    for (nom_q, m), cl in classements.items():
        unite = "d" if m.startswith("Histo") else "n"
        fichier = f"top5_{nom_q[8]}_{m.split(' (')[0].replace(' ', '_').replace('=', '')}.png"
        figure_top5(nom_q, requetes[nom_q], cl, galerie, verite[nom_q],
                    f"Top-5 de la requête {nom_q[8]} — {m} "
                    f"({'d = distance L2 entre histogrammes' if unite == 'd' else 'n = nb de correspondances'})",
                    os.path.join(dossier_top5, fichier), unite)

    # Opposite failure cases, selected automatically from the results
    m_sift = [m for m in METHODES if "modéré" in m][0]
    seuil_mod = SEUILS_SIFT["modéré"]
    p5 = {(r["requete"], r["methode"]): r["P@5"] for r in resultats}

    def ecart(q):  # SIFT advantage over the best histogram
        return p5[(q, m_sift)] - max(p5[(q, "Histo RGB")], p5[(q, "Histo Cb-Cr")])

    q_sift = max(requetes, key=ecart)    # histogram fails, SIFT succeeds
    q_histo = min(requetes, key=ecart)   # SIFT fails, histogram succeeds
    print(f"\nCas histogramme < SIFT : requête {q_sift[8]} ; cas SIFT < histogramme : requête {q_histo[8]}")

    meilleur_pert = next(n for n, _ in classements[(q_sift, m_sift)] if n in verite[q_sift])
    figure_correspondances(requetes[q_sift], galerie[meilleur_pert], seuil_mod,
                           f"Requête {q_sift[8]} vs {meilleur_pert[:-4]} — SIFT, T={int(seuil_mod)}",
                           os.path.join(DOSSIER_RESULTATS, f"correspondances_{q_sift[8]}_{meilleur_pert[:-4]}.png"))
    # Matches at the three thresholds for the same pair (illustrating the role of T)
    for nom_s, t in SEUILS_SIFT.items():
        figure_correspondances(requetes[q_sift], galerie[meilleur_pert], t,
                               f"Requête {q_sift[8]} vs {meilleur_pert[:-4]} — seuil {nom_s}",
                               os.path.join(DOSSIER_RESULTATS,
                                            f"correspondances_{q_sift[8]}_T{int(t)}.png"))
    # Query-to-worst-distractor matches (false positives with permissive T)
    t_perm = SEUILS_SIFT["permissif"]
    m_perm = [m for m in METHODES if "permissif" in m][0]
    faux = next((n for n, _ in classements[(q_sift, m_perm)] if n not in verite[q_sift]), None)
    if faux:
        figure_correspondances(requetes[q_sift], galerie[faux], t_perm,
                               f"Requête {q_sift[8]} vs {faux[:-4]} (non pertinente) — T={int(t_perm)}",
                               os.path.join(DOSSIER_RESULTATS, f"correspondances_{q_sift[8]}_{faux[:-4]}_faux.png"))

    histo_best = max(("Histo RGB", "Histo Cb-Cr"), key=lambda m: p5[(q_histo, m)])
    figure_top5(q_histo, requetes[q_histo], classements[(q_histo, m_sift)], galerie, verite[q_histo],
                f"Requête {q_histo[8]} — SIFT T={int(seuil_mod)} (n = nb de correspondances)",
                os.path.join(DOSSIER_RESULTATS, f"echec_sift_{q_histo[8]}.png"), "n")
    figure_top5(q_histo, requetes[q_histo], classements[(q_histo, histo_best)], galerie, verite[q_histo],
                f"Requête {q_histo[8]} — {histo_best} (d = distance L2)",
                os.path.join(DOSSIER_RESULTATS, f"succes_histo_{q_histo[8]}.png"), "d")
    figure_top5(q_sift, requetes[q_sift], classements[(q_sift, "Histo RGB")], galerie, verite[q_sift],
                f"Requête {q_sift[8]} — Histo RGB (d = distance L2)",
                os.path.join(DOSSIER_RESULTATS, f"echec_histo_{q_sift[8]}.png"), "d")

    # RGB and Cb-Cr histograms: query + its first 3 relevant images (by file name)
    for q in requetes:
        pert = sorted(verite[q])
        figure_histogrammes(q, requetes[q], pert[:3], galerie,
                            os.path.join(DOSSIER_RESULTATS, f"histogrammes_{q[8]}.png"))

    print(f"\nRésultats écrits dans {DOSSIER_RESULTATS}")


if __name__ == "__main__":
    main()

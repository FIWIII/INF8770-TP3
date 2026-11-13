================================================================================
INF8770 - Technologies Multimédias - Automne 2026
Archive de Données pour le Travail Pratique #3 :
Recherche d'images par le contenu pictural (Histogrammes de couleurs vs SIFT)
================================================================================

Cette archive contient le jeu de données d'images nécessaire à la réalisation de
vos expérimentations d'indexation et de recherche par le contenu pour le TP3.
Toutes les images sont fournies au format PNG (24 bits couleur), à une résolution
uniforme de 512 x 384 pixels, afin de garantir des comparaisons rigoureuses et un
temps d'exécution rapide sur CPU.

Principe d'utilisation : pour chacune des 4 images du dossier requetes/, votre
système calcule sa distance (ou similarité) avec chacune des 32 images du dossier
galerie/, trie ces 32 images de la plus similaire à la moins similaire, puis
retourne les meilleures correspondances (Top-5 et Top-10).

--------------------------------------------------------------------------------
STRUCTURE DE L'ARCHIVE :
--------------------------------------------------------------------------------

1. Dossier requetes/ (4 images requêtes aux caractéristiques contrastées) :
   - requete_A.png (512 x 384) : Scène de motocross richement texturée (détails
     mécaniques, lettrages, rayons, casques), présente dans la galerie sous de
     forts zooms localisés, rotations et changements d'angle de vue.
   - requete_B.png (512 x 384) : Scène extérieure d'un chalet peint en montagne,
     présente dans la galerie sous de fortes variations d'exposition lumineuse
     (sous-exposition crépusculaire, surexposition, éclairage directionnel).
   - requete_C.png (512 x 384) : Monoplace Mercedes F1 de Kimi Antonelli sur
     circuit, dont la voiture réapparaît dans la galerie devant cinq arrière-plans
     complètement différents.
   - requete_D.png (512 x 384) : Composition lumineuse lisse hors-focus (bokeh)
     sans coins ni textures haute fréquence, mais dotée d'une palette chromatique
     distinctive dont la disposition spatiale varie dans la galerie.

2. Dossier galerie/ (32 images candidates numérotées de img_01.png à img_32.png) :
   - Base d'images dans laquelle s'effectue la recherche pour chaque requête.
   - Contient exactement 5 images pertinentes pour chacune des 4 requêtes
     (soit 20 images cibles au total) ainsi que 12 images distractrices variées,
     mélangées aléatoirement.

3. Fichier verite_terrain.csv (UTF-8) :
   - Associe à chaque fichier requête ses 5 images pertinentes exactes dans la
     galerie (séparées par un point-virgule ';').
   - Ce fichier sert exclusivement au calcul objectif de vos métriques de
     performance (Précision@5, Rappel@10) et ne doit évidemment jamais être
     utilisé par l'algorithme de recherche lui-même.

--------------------------------------------------------------------------------
RAPPELS TECHNIQUES IMPORTANTS :
--------------------------------------------------------------------------------
- Attention à l'ordre des canaux lors du chargement et des conversions sous
  OpenCV : cv2.imread charge les images en BGR (et non en RGB), et la conversion
  cv2.COLOR_BGR2YCrCb place la luminance Y dans le canal 0, Cr dans le canal 1
  et Cb dans le canal 2.
- Appliquez exactement les mêmes paramètres (nombre de niveaux de quantification
  des histogrammes, mesure de distance, et les trois seuils de distance maximale
  sur les correspondances SIFT) aux quatre requêtes afin que vos comparaisons
  expérimentales restent valides.

Bon succès dans votre TP3 !
L'équipe enseignante d'INF8770

# Détection d'Anomalies par Auto-encodeur (Réseau Encodeur-Décodeur)

Ce projet implémente un réseau de neurones de type auto-encodeur dédié à la détection d'anomalies sur le dataset *Shuttle*. Le modèle apprend à reconstruire exclusivement les données normales afin d'identifier les déviations lors de la phase de test.

## 1. Choix des Hyperparamètres

La sélection des paramètres repose sur une analyse statistique des données pour garantir une compression optimale cite: 5, 32753e.

* **Espace Latent (k = 6) :** Basé sur l'analyse des valeurs propres suivantes : $7.9204 \times 10^{-4}, 1.2390 \times 10^{-4}, 2.3841 \times 10^{-3}, 6.2564 \times 10^{-1}, 9.8503 \times 10^{-1}, 1.0004, 1.0245, 1.7436, 3.6172$ cite: 6, 7, 8.
* **Justification :** On observe 6 valeurs propres significatives, justifiant le choix de k=6 pour l'espace latent.
* **Taux d’apprentissage :** Fixé à **0,0015** pour assurer une stabilisation fluide de la perte cite: 10, 11.
* **Optimiseur :** **Adam**, utilisé pour sa convergence efficace cite: 12, 13.
* **Epochs :** L'entraînement est réalisé sur **80 époques**. La courbe des pertes montre que le plateau de convergence est atteint vers la 76ème époque cite: 14, 15.

![Loss Curve](./images/Epoch_vs_MSE_loss.png)
*Figure 1 : Évolution de la perte MSE (K=6) montrant la stabilisation du modèle.*

## 2. Architecture du Réseau

Le réseau adopte une structure symétrique en "sablier" pour extraire les caractéristiques essentielles des signaux cite: 16, 30a669.

![Architecture](./images/architecture.png)
*Figure 2 : Schéma de l'architecture du réseau Encodeur-Décodeur.*

* **Dimensions des entrées/sorties :** Entrées de $1 \times 9$ et sorties de $1 \times 9$ cite: 20, 21.
* **Encodeur :** Trois couches linéaires progressives (9 → 8 → 7 → 6) cite: 17, 3216a9.
* **Activations :** Utilisation de couches **ReLU** après chaque couche linéaire pour introduire de la non-linéarité cite: 18.
* **Espace Latent :** Dimension compressée de **6** cite: 19.
* **Décodeur :** Structure miroir de l'encodeur (6 → 7 → 8 → 9) cite: 30a669.
* **Sortie finale :** Aucune fonction d'activation n'est appliquée après la dernière couche linéaire pour permettre une reconstruction fidèle des valeurs réelles cite: 22.

## 3. Stratégie d’Entraînement

L'entraînement est optimisé pour traiter les données normales tout en ignorant les anomalies cite: 23.

* **Entraînement par lots (Batch Size) :** Utilisation de lots de **256**. Ce choix empirique maximise la vitesse de calcul tout en évitant une généralisation excessive cite: 24, 25.
* **Prétraitement :** Élimination des données invalides et séparation des étiquettes pour ne garder que les 9 dimensions des capteurs cite: 26, 27.
* **Normalisation :**
    * Calculée uniquement sur les données d'entraînement valides (classe 1) cite: 28, 29.
    * Les données de test sont normalisées avec les moyennes et déviations standard du set d'entraînement pour éviter que les anomalies ne biaisent l'échelle cite: 30, 32753e.

![Strategie Entrainement](./images/image_32753e.png)
*Figure 3 : Détail du script de normalisation et de filtrage des données.*

## 4. Détermination du Seuil Optimal

La classification finale repose sur le calcul de l'erreur de reconstruction (MSE). Un seuil est déterminé pour séparer le "sain" de l'"anormal" cite: 9, 32.

* **Méthode :** Recherche du compromis idéal minimisant les Faux Négatifs (FN) et les Faux Positifs (FP) cite: 33, 35.
* **Analyse visuelle :** Utilisation d'histogrammes de pertes pour identifier la séparation entre les deux distributions cite: 35.

![Histogramme des pertes](./images/histogramme_pertes.png)
*Figure 4 : Distribution des erreurs de reconstruction pour les données normales et anormales.*

* **Évaluation :** Le seuil est validé par la courbe de la F-mesure (F1-score) en fonction du seuil appliqué cite: 31.

![F-Mesure](./images/FMesure.png)
*Figure 5 : Évolution de la F-mesure et de l'Accuracy selon le seuil sélectionné.*

## 5. Résultats Obtenus

Les performances finales démontrent la haute précision du modèle pour ce jeu de données cite: 36 :
* **F-score :** 0,9912
* **Accuracy :** 98,62 %
* **Test Loss finale :** 2,5372
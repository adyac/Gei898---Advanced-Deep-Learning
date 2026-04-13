# Loss Functions - Équations Détaillées

## 1. HINGE LOSS - Discriminateur

$$L_D = \mathbb{E}[\max(0, 1 - D(x, y))] + \mathbb{E}[\max(0, 1 + D(x, \tilde{y}))]$$

**Où:**
- $D(x, y)$ = sortie discriminateur pour paire réelle (image, masque réel)
- $D(x, \tilde{y})$ = sortie discriminateur pour paire fausse (image, masque généré)
- $\mathbb{E}[\cdot]$ = espérance (moyenne sur le batch)
- $\max(0, \cdot)$ = ReLU

**Interprétation:**
- Premier terme: Pénalise si $D(real) < 1$ (mauvaise discrimination des réels)
- Deuxième terme: Pénalise si $D(fake) > -1$ (mauvaise discrimination des faux)

---

## 2. HINGE LOSS - Générateur

$$L_G^{adv} = \mathbb{E}[\max(0, 1 - D(x, \tilde{y}))]$$

**Où:**
- $\tilde{y} = G(x)$ = masque généré par le générateur

**Interprétation:**
- Force le générateur à produire des masques avec $D(x, \tilde{y}) > 1$ (tromper le discriminateur)

---

## 3. BINARY CROSS-ENTROPY LOSS

$$L_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\tilde{y}_i) + (1-y_i) \log(1-\tilde{y}_i)]$$

**Où:**
- $y_i \in \{0, 1\}$ = valeur ground truth du pixel $i$
- $\tilde{y}_i \in [0, 1]$ = valeur prédite (après Sigmoid)
- $N$ = nombre total de pixels

**Interprétation:**
- Pénalise l'erreur de classification par pixel
- Utilise la probabilité logarithmique (log-likelihood)
- Pousse le générateur à bien classifier chaque pixel

---

## 4. L1 LOSS (Mean Absolute Error)

$$L_{L1} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \tilde{y}_i|$$

**Où:**
- $y_i$ = valeur ground truth du pixel $i$
- $\tilde{y}_i$ = valeur prédite du pixel $i$
- $N$ = nombre total de pixels (H × W × C)

**Interprétation:**
- Pénalité linéaire (contrairement à L2 qui est quadratique)
- Moins sensible aux outliers que L2
- Ajoute la régularisation spatiale (smoothness)

---

## 5. LOSS COMBINÉE DU GÉNÉRATEUR

$$L_G = L_G^{adv} + \lambda_{L1} \cdot (L_{BCE} + 0.5 \cdot L_{L1})$$

**Développée complètement:**

$$L_G = \mathbb{E}[\max(0, 1 - D(x, G(x)))] + \lambda_{L1} \left( -\frac{1}{N}\sum_{i}[y_i \log(\tilde{y}_i) + (1-y_i)\log(1-\tilde{y}_i)] + 0.5 \cdot \frac{1}{N}\sum_{i}|y_i - \tilde{y}_i| \right)$$

**Hyperparamètres:**
- $\lambda_{L1} = 2.0$ (poids de la reconstruction L1)
- Le $0.5$ réduit l'importance de L1 (BCE est plus important pour masques binaires)

**Intuition:**
```
L_G = (Tromper D) + 2.0 × (Classification binaire + Smoothness)
      └──Adversarial──┘   └─────────Reconstruction─────────┘
```

---

## 6. LOSS COMPLÈTE DU DISCRIMINATEUR

$$L_D = \mathbb{E}[\max(0, 1 - D(x, y))] + \mathbb{E}[\max(0, 1 + D(x, G(x)))]$$

**Alternative avec Label Smoothing:**

$$L_D = \mathbb{E}[\max(0, 1 - (1-\alpha) \cdot D(x, y))] + \mathbb{E}[\max(0, 1 + D(x, G(x)))]$$

**Où:**
- $\alpha = \text{label\_smooth} = 0.2$ (smoothing factor)
- Real labels deviennent $0.8$ au lieu de $1.0$

**Effet du smoothing:**
- Rend D moins confiant et plus régularisé
- Évite l'overfitting du discriminateur
- Maintient un gradient stable pour G

---

## 7. COMPARAISON: HINGE vs BCE LOSS

### Binary Cross-Entropy (Traditionnel)

$$L_{BCE}^{disc} = -\mathbb{E}[\log D(x,y)] - \mathbb{E}[\log(1-D(x,G(x)))]$$

**Problème:** Quand D converge, gradient → 0 (saturation sigmoïde)

### Hinge Loss (Notre implémentation)

$$L_{Hinge}^{disc} = \mathbb{E}[\max(0, 1 - D(x,y))] + \mathbb{E}[\max(0, 1 + D(x,G(x)))]$$

**Avantage:** Gradient reste constant même si D converge

---

## 8. COMPOSITION FINALE - BOUCLE D'ENTRAÎNEMENT

**Étape 1 - Entraîner Discriminateur:**
$$\theta_D \leftarrow \theta_D - \nabla_{\theta_D} L_D$$

**Étape 2 - Entraîner Générateur:**
$$\theta_G \leftarrow \theta_G - \nabla_{\theta_G} L_G$$

**Où:**
- $\theta_D, \theta_G$ = paramètres du discriminateur et générateur
- $\nabla$ = gradient

---

## Résumé des Coefficients

| Composant | Coefficient | Raison |
|-----------|------------|--------|
| Hinge Loss | 1.0 | Signal adversarial principal |
| BCE Loss | λ_L1 = 2.0 | Classification binaire importante |
| L1 Loss | 0.5 × λ_L1 = 1.0 | Smoothness secondaire |
| Label Smooth | 0.2 | Régularisation D |

---

## Calculs du Gradient (Simplifié)

### Gradient Hinge Loss

Pour $\max(0, 1 - D(x)):$

$$\frac{\partial}{\partial D} \max(0, 1 - D) = \begin{cases} -1 & \text{si } D < 1 \\ 0 & \text{si } D \geq 1 \end{cases}$$

**Avantage:** Le gradient est constant (-1), jamais 0!

### Gradient BCE Loss

Pour $-[y\log(p) + (1-y)\log(1-p)]:$

$$\frac{\partial}{\partial p} BCE = \frac{p - y}{p(1-p)}$$

**Problème:** Quand $p \to 0$ ou $p \to 1$, gradient devient très petit

---

## Notes Numériques

**Plages de valeurs en entraînement:**
- Hinge Loss: Typiquement 0.5-2.0
- BCE Loss: Typiquement 0.5-1.0
- L1 Loss: Typiquement 0.5-0.8
- GenLoss total: Typiquement 2.0-5.0
- DiscLoss: Descend de ~1.0 vers ~0.0 (normal)

**Interprétation:**
- GenLoss > 3.0 = Discriminateur fort (bon!)
- L1Loss < 0.7 = Reconstruction bonne (bon!)
- DiscLoss > 0.1 = Apprentissage équilibré (bon!)

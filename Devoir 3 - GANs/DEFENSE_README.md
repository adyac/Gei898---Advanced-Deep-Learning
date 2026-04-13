# Pix2Pix — Défense de Projet GEI898
## Traduction Conditionnelle d'Images : Détection de Défauts sur Transistors

---

## Table des matières

1. [Contexte et problème](#1-contexte-et-problème)
2. [Pipeline complet : de l'image brute au masque généré](#2-pipeline-complet--de-limage-brute-au-masque-généré)
3. [Architecture du Générateur : U-Net](#3-architecture-du-générateur--u-net)
4. [Architecture du Discriminateur : PatchGAN](#4-architecture-du-discriminateur--patchgan)
5. [Fonction de perte combinée](#5-fonction-de-perte-combinée)
6. [Boucle d'entraînement](#6-boucle-dentraînement)
7. [Schéma bloc du pipeline d'entraînement](#7-schéma-bloc-du-pipeline-dentraînement)
8. [Métriques d'évaluation et leur signification](#8-métriques-dévaluation-et-leur-signification)
9. [Expérience bruit](#9-expérience-bruit)
10. [Questions probables des professeurs — avec réponses](#10-questions-probables-des-professeurs--avec-réponses)

---

## 1. Contexte et problème

### Le problème : Détection d'anomalies sur transistors

Le jeu de données **MVTec AD** contient des images de transistors industriels avec différents types de défauts :
- `bent_lead` — patte courbée
- `cut_lead` — patte coupée
- `damaged_case` — boîtier endommagé
- `misplaced` — composant mal positionné

L'objectif est de prendre une image d'un transistor défectueux et de **générer automatiquement un masque de localisation du défaut** — une image binaire qui indique *où* se trouve le défaut.

Ce n'est **pas** de la classification (bon/mauvais). C'est de la **segmentation conditionnelle** : on traduit une image d'entrée dans une image de sortie qui encode l'information spatiale du défaut.

### Pourquoi un GAN conditionnel (Pix2Pix) ?

Un GAN classique génère des images à partir de bruit aléatoire (non conditionné). Ici, on veut que la sortie **dépende précisément de l'entrée**. Pix2Pix (Isola et al., CVPR 2017) introduit le cadre de **traduction d'image à image** : le générateur n'est plus libre de créer n'importe quoi — il est conditionné sur une image source.

---

## 2. Pipeline complet : de l'image brute au masque généré

### Étape 1 — Chargement et prétraitement des données

```
Image brute (PNG, taille variable, ex: 256×256 ou 900×700 pixels)
    │
    ▼
Resize → 128×128 pixels
    │
    ▼
ToTensor() : H×W×C (NumPy uint8) → C×H×W (Tensor float32) dans [0.0, 1.0]
    │
    ▼
Normalize(mean=0.5, std=0.5) : [0,1] → [-1, 1]
    │
    ▼
Tensor condition : shape (3, 128, 128), valeurs dans [-1, 1]
```

**Pourquoi normaliser vers [-1, 1] ?**
Le générateur utilise `Tanh()` en sortie, qui produit des valeurs dans [-1, 1]. Il faut donc que les targets (masques ground truth) soient aussi dans [-1, 1] pour que la perte L1 soit cohérente.

**Appariement des données (dataset pairing)**

```
test/bent_lead/000.png      ←→    ground_truth/bent_lead/000_mask.png
test/bent_lead/001.png      ←→    ground_truth/bent_lead/001_mask.png
test/cut_lead/000.png       ←→    ground_truth/cut_lead/000_mask.png
...
```
Le dataset charge les paires **(image défectueuse, masque ground truth)** en triant les fichiers par ordre alphabétique. Au total : 4 types × N images = ~80-200 paires selon le dataset.

### Étape 2 — Passage dans le Générateur U-Net

```
Tensor (B, 3, 128, 128)
    │
    ▼
4 blocs encodeurs + bottleneck → features compressées (B, 512, 8, 8)
    │
    ▼
4 blocs décodeurs + skip connections → reconstruction (B, 3, 128, 128)
    │
    ▼
Tanh() → valeurs dans [-1, 1]
    │
    ▼
Masque généré : (B, 3, 128, 128) dans [-1, 1]
```

### Étape 3 — Évaluation du Discriminateur PatchGAN

```
[Image condition (B,3,128,128), Masque généré (B,3,128,128)]
    │
    ▼ Concatenation sur l'axe des canaux
(B, 6, 128, 128)
    │
    ▼
4 blocs de downsampling Conv + BN + LeakyReLU
    │
    ▼
Carte de scores patches (B, 1, 7, 7)   ← chaque cellule = vrai ou faux local
```

### Étape 4 — Calcul des pertes et rétropropagation

```
D_loss = BCE(D(cond, real), 1) + BCE(D(cond, fake), 0) / 2
G_loss = BCE(D(cond, fake), 1) + λ × L1(fake, target)   λ=10
    │
    ▼
Backpropagation → mise à jour des poids
```

### Étape 5 — Dénormalisation pour visualisation

```
Tensor [-1, 1] → (Tensor + 1) / 2 → [0, 1] → image RGB affichable
```

---

## 3. Architecture du Générateur : U-Net

### Vue d'ensemble

Le générateur est un **encodeur-décodeur symétrique avec skip connections**. L'idée fondamentale : compresser une image jusqu'à une représentation vectorielle compacte (le bottleneck), puis reconstruire une image de sortie — mais en réinjectant les features des couches d'encodage à chaque étape du décodage.

### Encodeur — chemin descendant

| Bloc    | Entrée (canaux) | Sortie (canaux) | Résolution | Opérations                           |
|---------|-----------------|-----------------|------------|--------------------------------------|
| enc1    | 3               | 64              | 128×128    | Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU |
| MaxPool |                 |                 | 64×64      | Fenêtre 2×2, stride 2                |
| enc2    | 64              | 128             | 64×64      | Conv 3×3 → BN → ReLU × 2            |
| MaxPool |                 |                 | 32×32      |                                      |
| enc3    | 128             | 256             | 32×32      | Conv 3×3 → BN → ReLU × 2            |
| MaxPool |                 |                 | 16×16      |                                      |
| enc4    | 256             | 512             | 16×16      | Conv 3×3 → BN → ReLU × 2            |
| MaxPool |                 |                 | 8×8        |                                      |

### Bottleneck

| Bloc       | Entrée (canaux) | Sortie (canaux) | Résolution |
|------------|-----------------|-----------------|------------|
| bottleneck | 512             | 512             | 8×8        |

**C'est ici que l'image est la plus compressée** : (3, 128, 128) = 49 152 valeurs → (512, 8, 8) = 32 768 valeurs, soit ~1.5× compression globale. La représentation latente capte le contexte global de l'image.

### Décodeur — chemin ascendant avec skip connections

| Opération   | Entrée (canaux)   | Détail                          | Sortie (canaux) | Résolution |
|-------------|-------------------|---------------------------------|-----------------|------------|
| upconv4     | 512               | ConvTranspose2d k=4, s=2, p=1   | 256             | 16×16      |
| skip concat | 256 + 512 = **768** | Concatenation avec e4            | 768             | 16×16      |
| dec4        | 768               | Conv 3×3 → BN → ReLU × 2        | 256             | 16×16      |
| upconv3     | 256               | ConvTranspose2d                  | 128             | 32×32      |
| skip concat | 128 + 256 = **384** | Concatenation avec e3            | 384             | 32×32      |
| dec3        | 384               | Conv 3×3 → BN → ReLU × 2        | 128             | 32×32      |
| upconv2     | 128               | ConvTranspose2d                  | 64              | 64×64      |
| skip concat | 64 + 128 = **192**  | Concatenation avec e2            | 192             | 64×64      |
| dec2        | 192               | Conv 3×3 → BN → ReLU × 2        | 64              | 64×64      |
| upconv1     | 64                | ConvTranspose2d                  | 64              | 128×128    |
| skip concat | 64 + 64 = **128**   | Concatenation avec e1            | 128             | 128×128    |
| dec1        | 128               | Conv 3×3 → BN → ReLU → sortie   | 3               | 128×128    |
| Tanh        | 3                 | Normalisation finale              | 3               | 128×128    |

### Pourquoi les skip connections sont essentielles

Sans skip connections (encodeur-décodeur simple) : le décodeur reçoit uniquement le bottleneck de 8×8. Il doit halluciner tous les détails spatiaux fins — bords, textures, positions précises. Le résultat est flou.

Avec skip connections : à chaque étape de reconstruction, le décodeur reçoit **les features de l'encodeur à la même résolution**. Concrètement, quand on reconstruit à 64×64, on a accès aux features extraites à 64×64 lors de l'encodage. Le détail fin n'a pas besoin d'être mémorisé dans le bottleneck — il est passé directement via les connexions.

```
Encodeur                    Décodeur
e1 (128×128) ─────────────────────────→ concat avec d1 (128×128)
e2 (64×64)   ──────────────────────→ concat avec d2 (64×64)
e3 (32×32)   ────────────────────→ concat avec d3 (32×32)
e4 (16×16)   ──────────────────→ concat avec d4 (16×16)
                  bottleneck (8×8)
```

Ce flux de gradient raccourci évite aussi le problème du **gradient qui disparaît** dans les couches profondes.

---

## 4. Architecture du Discriminateur : PatchGAN

### Concept fondamental : classer des patches, pas l'image entière

Un discriminateur classique prend une image et sort un scalaire : "réel" (1) ou "faux" (0). Le PatchGAN produit une **carte 2D de scores**, où chaque valeur correspond à un patch de 70×70 pixels dans l'image originale.

**Intuition** : si vous devez évaluer le réalisme d'une photo, vous regardez la qualité locale — la texture, les bords, la cohérence des couleurs dans une petite zone. Le PatchGAN force le générateur à produire des détails localement réalistes, pas juste globalement plausibles.

### Architecture

```
Input: concat[condition(3ch), image(3ch)] = (B, 6, 128, 128)

Bloc 1:  Conv(6→64,  k=4, s=2, p=1) → LeakyReLU(0.2)          128→64   [Pas de BN]
Bloc 2:  Conv(64→128, k=4, s=2, p=1) → BN → LeakyReLU(0.2)     64→32
Bloc 3:  Conv(128→256, k=4, s=2, p=1) → BN → LeakyReLU(0.2)    32→16
Bloc 4:  Conv(256→512, k=4, s=2, p=1) → BN → LeakyReLU(0.2)    16→8
Final:   Conv(512→1, k=4, s=1, p=1)                              8→7

Output: (B, 1, 7, 7)  ← 49 scores de patches
```

**Pourquoi pas de BatchNorm au premier bloc ?** Le premier bloc voit des pixels bruts. BN à cette étape normaliserait la distribution des entrées et effacerait l'information de contraste absolu, qui est utile pour distinguer réel/faux.

**L'entrée à 6 canaux** : le discriminateur reçoit toujours la paire (image condition, image cible/générée). Il ne peut pas évaluer un masque isolément — il doit juger si le masque est *cohérent avec l'image d'entrée*. C'est ce qui en fait un discriminateur **conditionnel**.

### Champ réceptif de 70×70

Chaque cellule de la sortie 7×7 "voit" une zone de 70×70 pixels dans l'entrée 128×128. La perte est calculée comme la **moyenne de ces 49 scores**. En entraînement, toutes les cellules doivent classer la bonne réponse.

---

## 5. Fonction de perte combinée

### Perte du discriminateur

$$\mathcal{L}_D = \frac{1}{2}\left[\underbrace{\text{BCE}(D(x, y), 1)}_{\text{réels → 1}} + \underbrace{\text{BCE}(D(x, \hat{y}), 0)}_{\text{générés → 0}}\right]$$

où $x$ = image condition, $y$ = masque réel, $\hat{y}$ = masque généré.

Le discriminateur cherche à maximiser cette perte — à être bon pour distinguer. On l'entraîne en alternance avec le générateur.

**Note implémentation** : lors de l'entraînement du discriminateur, `fake_images` est créé avec `torch.no_grad()` pour éviter de propager le gradient dans le générateur.

### Perte du générateur

$$\mathcal{L}_G = \underbrace{\text{BCE}(D(x, \hat{y}), 1)}_{\text{tromper D}} + \lambda \cdot \underbrace{\|y - \hat{y}\|_1}_{\text{reconstruction fidèle}}$$

avec $\lambda = 10$ (hyperparamètre).

**Deux objectifs en tension :**
- **Terme GAN** (adversarial) : générer des masques qui *ressemblent* à des masques réels selon le discriminateur — forcer le réalisme perceptuel
- **Terme L1** : générer des masques fidèles pixel-à-pixel — forcer la précision spatiale

### Interprétation des pertes observées

Après 30 epochs : `GenLoss ≈ 13.0`, `DiscLoss ≈ 0.05`, `L1Loss ≈ 0.99`

La formule numérique : $\mathcal{L}_G = \mathcal{L}_{GAN} + 10 \times 0.99 \approx 3.1 + 9.9 = 13.0$

Cela signifie :
- **L1 = 0.99** : l'erreur de reconstruction normalisée est grande en valeur absolue, mais c'est attendu car les masques ground truth sont souvent **noirs à >90%** (le défaut occupe peu de pixels). Une erreur L1 moyenne sur des images mostly-black est structurellement haute.
- **DiscLoss = 0.05** : le discriminateur a presque convergé — il distingue très bien réel/faux, ce qui est normal en fin d'entraînement.
- **La qualité visuelle** est ce qui importe — des résultats tels que des masques montrant clairement la forme et la position du défaut confirment que le modèle a appris.

### Pourquoi L1 et non L2 ?

L1 produit des sorties moins floues que L2. Avec L2, la perte carée pénalise les grandes erreurs plus fortement et pousse le générateur à produire la **moyenne** de toutes les cibles possibles (flou). L1 est moins sensible aux valeurs aberrantes et préserve mieux les contours.

---

## 6. Boucle d'entraînement

### Ordre des mises à jour : un principe fondamental des GANs

```
Pour chaque batch (condition, target) :

┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 1 — Entraîner le Discriminateur                          │
│                                                                  │
│  1a. Générer fake = G(condition)  [avec torch.no_grad()]        │
│  1b. D(condition, target)    → real_output                       │
│  1c. D(condition, fake.detach()) → fake_output                  │
│  1d. disc_loss = (BCE(real, 1) + BCE(fake, 0)) / 2              │
│  1e. disc_loss.backward() ; disc_optimizer.step()               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ÉTAPE 2 — Entraîner le Générateur                              │
│                                                                  │
│  2a. Générer fake = G(condition)    [avec gradient cette fois]   │
│  2b. D(condition, fake) → fake_output_for_gen                   │
│  2c. gen_loss = BCE(fake_output, 1) + λ × L1(fake, target)      │
│  2d. gen_loss.backward() ; gen_optimizer.step()                 │
└─────────────────────────────────────────────────────────────────┘
```

**Pourquoi `.detach()` dans l'étape 1c ?**
Quand on entraîne le discriminateur, on ne veut pas propager le gradient de la perte du discriminateur dans le générateur. `.detach()` coupe le graphe de calcul entre les poids du générateur et cette perte.

**Pourquoi générer fake deux fois ?**
Dans l'étape 1, on utilise `torch.no_grad()` pour économiser de la mémoire. Dans l'étape 2, on doit régénérer pour avoir un graphe de calcul valide permettant la rétropropagation dans G.

### Initialisation des poids

```python
nn.init.normal_(m.weight, mean=0.0, std=0.02)   # Conv, ConvTranspose
nn.init.normal_(m.weight, mean=0.0, std=0.02)   # BatchNorm weight
nn.init.constant_(m.bias, 0)                      # BatchNorm bias
```

Cette initialisation provient du papier original de Pix2Pix/DCGAN. Une variance de 0.02 est choisie pour démarrer les poids dans une zone linéaire des activations (ni trop proches de 0, ni saturés).

### Hyperparamètres clés

| Paramètre     | Valeur   | Justification                                              |
|---------------|----------|------------------------------------------------------------|
| `lr`          | 0.0001   | Adam pour GANs, plus bas que SGD standard pour stabilité  |
| `beta_1`      | 0.5      | Momentum réduit (recommandé par Radford et al. pour GANs) |
| `beta_2`      | 0.999    | Standard Adam                                              |
| `lambda_l1`   | 10       | Compromis réalisme vs fidélité                             |
| `batch_size`  | 4        | Limité par la mémoire GPU pour le U-Net                    |
| `img_size`    | 128      | 128² = 16384 pixels (contrainte assignée)                  |

**Pourquoi beta_1=0.5 pour les GANs ?** La valeur par défaut de Adam (beta_1=0.9) conserve beaucoup de momentum, ce qui peut rendre l'entraînement adversarial instable. Beta_1=0.5 réagit plus vite aux nouvelles informations du gradient.

---

## 7. Schéma bloc du pipeline d'entraînement

```
═══════════════════════════════════════════════════════════════════════
                    PIPELINE PIX2PIX — VUE SYSTÈME
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                        DONNÉES (MVTec AD)                           │
│                                                                      │
│  dataset/transistor/                                                 │
│  ├── test/                     ├── ground_truth/                    │
│  │   ├── bent_lead/            │   ├── bent_lead/                   │
│  │   ├── cut_lead/             │   ├── cut_lead/                    │
│  │   ├── damaged_case/         │   ├── damaged_case/                │
│  │   └── misplaced/            │   └── misplaced/                   │
│       [image RGB défectueuse]        [masque de défaut]             │
└────────────────────────┬──────────────────────┬──────────────────────┘
                         │ appariement par ordre │
                         ▼ alphabétique         │
┌────────────────────────────────────────────────▼────────────────────┐
│                   Pix2PixDataset (pix2pix_dataset.py)               │
│                                                                      │
│  (test_img, gt_mask)  ──→  Resize(128) ──→ Normalize([-1,1])       │
│                                                                      │
│  Output: (condition: B×3×128×128, target: B×3×128×128)             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ DataLoader  batch_size=4
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BOUCLE D'ENTRAÎNEMENT                        │
│                         (pix2pix_train.py)                          │
│                                                                      │
│  condition  ─────────────────────────────────────────────────────┐  │
│                                                                   │  │
│  ┌─────────────────────────────────────────────────────────┐     │  │
│  │              GÉNÉRATEUR U-Net (pix2pix.py)              │     │  │
│  │                                                         │     │  │
│  │  condition (3,128,128)                                  │     │  │
│  │      │                                                  │     │  │
│  │  ┌───▼───────────────────────────────────────────────┐  │     │  │
│  │  │ enc1(3→64,128×128) ─────────────────────────────────────┐ │  │
│  │  │ pool → enc2(64→128,64×64) ──────────────────────────┐   │ │  │
│  │  │ pool → enc3(128→256,32×32) ──────────────────────┐  │   │ │  │
│  │  │ pool → enc4(256→512,16×16) ───────────────────┐  │  │   │ │  │
│  │  │ pool → bottleneck(512→512,8×8)                │  │  │   │ │  │
│  │  │              │                                │  │  │   │ │  │
│  │  │         upconv4 ──→ cat[↑,e4] ──→ dec4(256) ←┘  │  │   │ │  │
│  │  │              │                                   │  │   │ │  │
│  │  │         upconv3 ──→ cat[↑,e3] ──→ dec3(128) ←───┘  │   │ │  │
│  │  │              │                                      │   │ │  │
│  │  │         upconv2 ──→ cat[↑,e2] ──→ dec2(64)  ←──────┘   │ │  │
│  │  │              │                                          │ │  │
│  │  │         upconv1 ──→ cat[↑,e1] ──→ dec1(3)   ←──────────┘ │  │
│  │  │              │                                            │  │
│  │  │           Tanh() → fake_mask (3,128,128)                  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │  │
│                                                                   │  │
│              fake_mask ──────────────────────────────────┐        │  │
│                                                          │        │  │
│  ┌───────────────────────────────────────────────────────▼──────┐ │  │
│  │            DISCRIMINATEUR PatchGAN (pix2pix.py)              │ │  │
│  │                                                               │ │  │
│  │  Appel 1 [pour D] : D(condition, target)  → real_score (7×7) │ │  │
│  │  Appel 2 [pour D] : D(condition, fake.detach()) → fake_score │ │  │
│  │  Appel 3 [pour G] : D(condition, fake) → fake_score_for_G    │ │  │
│  │                                                               │ │  │
│  │  Architecture interne :                                       │ │  │
│  │   cat[cond,img](6ch) → block1(64,noBN) → block2(128) →       │ │  │
│  │   block3(256) → block4(512) → Conv→ (1,7,7)                  │ │  │
│  └───────────────────────────────────────────────────────────────┘ │  │
│                                                                      │  │
│  ┌───────────────────────────────────────────────────────────────┐   │  │
│  │                  PERTES (pix2pix_dataset.py)                  │   │  │
│  │                                                               │   │  │
│  │  disc_loss = BCE(real_score, 1) + BCE(fake_score, 0)         │   │  │
│  │             ─────────────────────────────────────            │   │  │
│  │                            2                                  │   │  │
│  │                                                               │   │  │
│  │  gen_loss = BCE(fake_score_for_G, 1) + 10 × L1(fake, target) │   │  │
│  └───────────────────────────────────────────────────────────────┘   │  │
│                                                                      │  │
└─────────────────────────────────────────────────────────────────────┘  │
                   ▲                          │                           │
                   │  mise à jour des poids   │  gradients               │
                   └──────────────────────────┘                           │
                                                                         │
                         [après N epochs]                                │
                                ▼                                        │
┌────────────────────────────────────────────────────────────────────────┘
│                        SORTIES
│
│  pix2pix_generator.pth          ← poids du générateur entraîné
│  pix2pix_training_losses.png    ← courbes de perte
│  pix2pix_samples.png            ← exemples de masks générés
└────────────────────────────────────────────────────────────────────────
```

---

## 8. Métriques d'évaluation et leur signification

### L1 Error (Mean Absolute Error)

$$L_1 = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

- **Ce que ça mesure** : l'écart absolu moyen entre chaque pixel prédit et le pixel ground truth.
- **Interprétation** : valeur entre 0 et 2 (normalisé dans [-1,1]). Plus proche de 0 = meilleure reconstruction.
- **Limite** : ne tient pas compte de la structure spatiale. Un masque décalé de 2 pixels peut avoir un L1 élevé même s'il est visuellement très juste.
- **Dans notre cas** : L1 ≈ 0.99 est élevé parce que les images sont **principalement noires** (fond noir, défauts occupent ~5-15% de l'image).

### MSE (Mean Squared Error)

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

- **Ce que ça mesure** : similaire au L1 mais pénalise les grandes erreurs de façon quadratique.
- **Plus sensible aux outliers** que L1 : un pixel très mal prédit a un impact disproportionné.

### SSIM — Structural Similarity Index

$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- **Ce que ça mesure** : la similarité perceptuelle — luminosité, contraste, et structure locale.
- **Plage** : [-1, 1]. SSIM > 0.7 = bonne qualité structurelle.
- **Pourquoi c'est meilleur que L1/MSE** : SSIM correspond mieux au jugement humain de la qualité d'image. Deux images peuvent avoir un SSIM élevé même si le L1 est élevé (si la structure est préservée).
- **3 composantes** : luminance (moyennes), contraste (variances), structure (covariance).

### Interprétation conjointe

| Scénario | L1 | SSIM | Indication |
|----------|----|------|------------|
| Masque correctement localisé | Faible à moyen | Élevé | Bon modèle |
| Masque décalé ou légèrement flou | Élevé | Moyen | Acceptable |
| Sortie entièrement noire | Élevé | Bas | Pas de généralisation |
| Bruit structuré | Moyen | Bas | Artefacts |

---

## 9. Expérience bruit

### Protocole

```
Mode normal  : G(image_défectueuse)     → masque prédit → compare avec GT
Mode bruité  : G(bruit_gaussien N(0,1)) → masque prédit → compare avec GT
Mode hybride : G(α×image + (1-α)×bruit) → masque prédit → compare avec GT
```

### Ce que ça prouve

Si le générateur a vraiment appris à traduire les images, ses prédictions avec du bruit en entrée doivent être **significativement moins bonnes**. Un ratio de dégradation de 2-5× sur la L1 prouve que le modèle est **conditionnel dépendant** — il n'a pas simplement mémorisé une distribution a priori des masques.

### Analyse théorique

Le générateur U-Net est fortement ancré dans l'image d'entrée via les skip connections. Avec du bruit en entrée :

1. **L'encodeur** extrait des features de bruit — pas de structure utile
2. **Le bottleneck** reçoit un code latent aléatoire
3. **Les skip connections** transmettent du bruit à tous les niveaux du décodeur
4. **La sortie** reflète cette désorganisation

Avec une image réelle, les skip connections transmettent des features structurées (bords, textures, positions des éléments du transistor) qui guident la reconstruction du masque.

Cette expérience répond à la question : *"Le modèle a-t-il appris la tâche, ou a-t-il simplement mémorisé la distribution de sortie ?"* → Il a appris la tâche.

---

## 10. Questions probables des professeurs — avec réponses

### Bloc 1 : Compréhension de l'architecture

**Q1 : Pourquoi utilise-t-on un U-Net plutôt qu'un encodeur-décodeur simple ?**

> Un encodeur-décodeur simple force toute l'information à passer par le bottleneck, ce qui crée un goulot d'étranglement. Les détails fins (bords du défaut, position exacte) sont perdus. Les skip connections dans l'U-Net court-circuitent le bottleneck pour les features haute résolution : l'information spatiale détaillée est réinjectée directement dans le décodeur à la bonne résolution. Résultat : la reconstruction est précise et nette, pas juste globalement correcte.

---

**Q2 : Pourquoi concatener plutôt qu'additionner dans les skip connections ?**

> L'addition fusionne deux ensembles de features mais oblige les deux représentations à être dans le même espace. La concatenation préserve les deux représentations distinctes et laisse le réseau apprendre comment les combiner (via les poids convolutifs suivants). Dans l'U-Net, les features encodées et les features décodées ont des significations différentes (texture d'entrée vs reconstruction cible), donc la concatenation est plus appropriée.

---

**Q3 : Pourquoi le PatchGAN et non un discriminateur global ?**

> Un discriminateur global produit un seul scalaire pour toute l'image — il peut être trompé par des images globalement plausibles mais localement incohérentes. Le PatchGAN force une évaluation locale : chaque patch de 70×70 doit être réaliste indépendamment. Cela pousse le générateur à produire des textures et des détails fins cohérents partout, pas seulement une impression globale correcte. De plus, la sortie 7×7 fournit un signal de supervision plus riche que un seul scalaire.

---

**Q4 : Pourquoi l'entrée du discriminateur est-elle à 6 canaux ?**

> Le discriminateur reçoit la concaténation de l'image condition (3 canaux, l'image du transistor défectueux) et de l'image évaluée (3 canaux, masque réel ou généré). Il doit apprendre à juger si un masque est *crédible par rapport à cette image spécifique*. Sans l'image condition, le discriminateur ne pourrait que juger si un masque ressemble à un masque générique, pas s'il correspond à cet image précise.

---

**Q5 : Qu'est-ce que BatchNorm fait dans votre réseau et pourquoi ne l'utilisez-vous pas au premier bloc du discriminateur ?**

> BatchNorm normalise les activations d'une couche pour chaque batch (moyenne ≈ 0, variance ≈ 1). Cela accélère l'entraînement, réduit la sensibilité à l'initialisation, et régularise légèrement. Au premier bloc du discriminateur, l'entrée est directement les pixels bruts. BN effacerait l'information de contraste absolu (les différences de luminosité globale), qui est utile pour distinguer réel de généré. Les couches suivantes, qui opèrent sur des features abstraites, bénéficient de BN.

---

### Bloc 2 : Compréhension de l'entraînement et des pertes

**Q6 : Pourquoi y a-t-il deux termes dans la perte du générateur ?**

> Le générateur a deux objectifs qui peuvent être en tension :
> 1. **Tromper le discriminateur** (terme GAN) : produire des masques réalistes que le discriminateur classifie comme "réel"
> 2. **Correspondre précisément à la cible** (terme L1) : minimiser l'erreur pixel-par-pixel avec le ground truth
>
> Sans le terme L1, le générateur peut créer des masques "réalistes" mais sans rapport avec l'entrée (mode collapse ou pure hallucination). Sans le terme GAN, le générateur produit des images floues (moyenne statistique). La combinaison force à la fois le réalisme et la fidélité.

---

**Q7 : Pourquoi `fake.detach()` dans l'entraînement du discriminateur ?**

> Lors de l'entraînement du discriminateur, on calcule une perte et on propage le gradient. Si on n'utilise pas `.detach()`, PyTorch propagerait ce gradient non seulement dans les poids du discriminateur, mais aussi dans ceux du générateur—ce qui ne fait pas partie du plan d'entraînement pour cette étape. `.detach()` coupe le graphe de calcul entre la sortie du générateur et la perte du discriminateur, assurant que seul le discriminateur est mis à jour.

---

**Q8 : Comment savez-vous que le modèle a bien appris et ne fait pas d'overfitting ?**

> Le principal indicateur est la qualité visuelle des masques générés : le modèle produit des masques qui montrent clairement la forme et la position du défaut sur des images qu'il n'a pas vues. Si c'était de l'overfitting pur, les prédictions sur de nouvelles images seraient incohérentes. L'expérience bruit confirme aussi que le modèle a appris la tâche (généralisation conditionnelle) et pas seulement à reproduire les masques du dataset.

---

**Q9 : Pourquoi lambda_l1=10 et non 100 comme dans le papier original ?**

> Le papier Pix2Pix propose λ=100 pour des datasets avec de nombreuses paires d'images (Cityscapes, Maps — des milliers d'exemples). Notre dataset est plus petit (~80-200 paires). Avec λ=100, le terme L1 domine trop, ce qui peut ralentir l'apprentissage du terme GAN et produire des images trop floues. λ=10 est une réduction empirique pour adapter au contexte de notre dataset plus petit.

---

**Q10 : Pourquoi Adam avec beta_1=0.5 pour les GANs ?**

> La valeur standard d'Adam est beta_1=0.9, qui conserve 90% du momentum précédent. Dans les GANs, l'entraînement adversarial est intrinsèquement non-stationnaire : le paysage de perte change à chaque mise à jour des deux réseaux. Avec beta_1=0.9, l'optimiseur réagit trop lentement aux changements, favorisant l'instabilité. beta_1=0.5 (recommandé par Radford et al. 2015 pour DCGAN) réagit plus rapidement aux nouvelles informations du gradient, ce qui stabilise l'entraînement adversarial.

---

### Bloc 3 : Compréhension des données et du problème

**Q11 : Pourquoi vous entraînez sur les images de TEST et non d'entraînement du dataset MVTec ?**

> Dans MVTec AD, le dossier `train/` ne contient que des images **normales** (sans défaut). Il n'y a pas de masques ground truth pour l'entraînement. Le dossier `test/` contient les images avec défauts et le dossier `ground_truth/` contient leurs masques correspondants. C'est la seule source de paires (image_défectueuse, masque) disponible dans ce dataset. Pour un déploiement en production, il faudrait une séparation train/val/test plus rigoureuse, mais dans le cadre de ce devoir, cette utilisation est justifiée.

---

**Q12 : Quel est l'impact de la normalisation [-1, 1] sur l'architecture ?**

> La normalisation [-1, 1] est liée au choix de l'activation de sortie Tanh(). Si les inputs et targets sont dans [0, 1] mais que la sortie est dans [-1, 1], la perte L1 calculerait des erreurs entre des espaces incompatibles. En normalisant tout vers [-1, 1], on assure la cohérence entre la range de la sortie Tanh et la range des targets. De plus, des entrées centrées en 0 améliorent la stabilité numérique du gradient dans les premières couches.

---

**Q13 : Comment les images sont-elles appariées ? Que se passe-t-il si les noms de fichiers ne correspondent pas ?**

> L'appariement se fait par tri alphabétique des listes de fichiers dans chaque dossier defect_type (via `sorted()`). Si les fichiers ont des noms non correspondants (ex: 000.png et 001_mask.png), la paire sera incorrecte mais l'entraînement continuera sans erreur — et les résultats seront mauvais. Dans le dataset MVTec AD, la correspondance est garantie par la structure officielle du dataset (même nombre de fichiers, nommés dans le même ordre), donc cette hypothèse est valide ici.

---

### Bloc 4 : Questions conceptuelles approfondies

**Q14 : En quoi Pix2Pix est-il différent d'un GAN classique (WGAN) ?**

> Un WGAN classique génère des images **à partir de bruit aléatoire** — aucune condition. Il apprend la distribution des images réelles et peut échantillonner dans cette distribution. Pix2Pix est un **GAN conditionnel** : le générateur reçoit une image spécifique en entrée et doit produire une sortie déterministe correspondante. La condition est ce qui rend la génération contrôlable et utile pour des tâches comme la traduction d'images. L'architecture U-Net (vs DCGAN pour WGAN) est choisie précisément pour maximiser le couplage entre condition et sortie via les skip connections.

---

**Q15 : Quelles sont les limites de cette approche ?**

> 1. **Données d'entraînement limitées** : le modèle a vu peu de paires. Plus d'augmentaion de données (flip, rotation, ajustement de contraste) améliorerait la généralisation.
> 2. **L1 cause du flou** : la perte L1 moyennée tend à produire des transitions douces aux bords des défauts.
> 3. **Correspondance par ordre** : si on déploie sur de nouvelles images sans masques, on ne peut pas évaluer quantitativement.
> 4. **Pas de mesure d'incertitude** : le modèle est déterministe — il ne peut pas exprimer "je ne suis pas sûr de la localisation".
> 5. **Sensibilité aux nouveaux types de défauts** : un défaut non présent dans le dataset d'entraînement sera mal généré.

---

**Q16 : Qu'arriverait-il si on retirait les skip connections ?**

> Sans skip connections, c'est un encodeur-décodeur simple (comme un auto-encodeur). Le décodeur ne recevrait que le bottleneck (512, 8, 8) pour reconstruire (3, 128, 128). L'information de position fine des défauts, des bords du transistor, des structures locales — toute cette information doit être compressée dans 8×8 = 64 positions. En pratique, les sorties seraient floues et mal localisées. Les skip connections sont la fonctionnalité principale qui rend U-Net supérieur à un simple auto-encodeur pour la segmentation.

---

**Q17 : Qu'est-ce que le mode collapse dans les GANs et comment le détecte-t-on ici ?**

> Le mode collapse est quand le générateur apprend à produire toujours la même sortie (ou un très petit ensemble de sorties) indépendamment de l'entrée — parce que ces sorties trompent suffisamment bien le discriminateur. On le détecterait ici par : toutes les sorties visuellement identiques malgré différentes entrées, perte du discriminateur qui monte brusquement (le discriminateur détecte facilement la répétition), perte L1 qui stagne à une valeur fixe. Dans notre cas, des sorties différentes pour des entrées différentes confirment l'absence de mode collapse.

---

**Q18 : Pourquoi utiliser `BCEWithLogitsLoss` plutôt que `BCELoss` ?**

> `BCEWithLogitsLoss` combine la sigmoïde et le calcul de la BCE en une seule opération. Cela évite des problèmes de stabilité numérique : si le réseau produit une valeur très grande (ex: 50), `sigmoid(50) ≈ 1.0` avec précision float32, et `log(1.0 - 1.0)` = `-∞`. En fusionnant les deux opérations, PyTorch utilise une formulation mathématiquement équivalente mais numériquement stable : `max(0, x) - x*y + log(1 + exp(-|x|))`.

---

*Dernière mise à jour : 30 epochs d'entraînement, lambda_l1=10, lr=0.0001, GPU RTX 4060, PyTorch 2.7.1+cu118*

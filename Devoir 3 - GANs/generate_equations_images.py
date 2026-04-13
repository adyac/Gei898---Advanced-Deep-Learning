"""
Generate equations image for PowerPoint presentation
Uses matplotlib to create a clean equations visualization
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Set up matplotlib without LaTeX (LaTeX not installed)
rcParams['text.usetex'] = False
rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 10

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Loss Functions - Équations Complètes', fontsize=20, fontweight='bold', y=0.995)

# ============================================================================
# Slide 1: Hinge Loss (Disc + Gen)
# ============================================================================
ax1 = axes[0]
ax1.axis('off')

equations_disc = [
    'DISCRIMINATOR - HINGE LOSS',
    '',
    'L_D = E[max(0, 1 - D(x, y))] + E[max(0, 1 + D(x, y_tilde))]',
    '',
    'Premier terme: Penalise si D(reel) < 1',
    'Deuxieme terme: Penalise si D(faux) > -1',
    '',
    'GENERATOR - HINGE LOSS (Adversarial)',
    '',
    'L_G_adv = E[max(0, 1 - D(x, G(x)))]',
    '',
    'Force G a produire D(x, y_tilde) > 1 (tromper D)',
    '',
    'Avantage Hinge vs BCE:',
    'Gradient constant meme si D converge => apprentissage stable',
]

text_disc = '\n'.join(equations_disc)
ax1.text(0.05, 0.95, text_disc, transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

# ============================================================================
# Slide 2: BCE + L1 Losses
# ============================================================================
ax2 = axes[1]
ax2.axis('off')

equations_recon = [
    'BINARY CROSS-ENTROPY LOSS',
    '',
    'L_BCE = -1/N * SUM[y_i * log(y_tilde_i) + (1-y_i) * log(1-y_tilde_i)]',
    '',
    'Role: Classification pixel-by-pixel de masques binaires',
    'Effet: Pousse chaque pixel vers 0 ou 1',
    '',
    'L1 LOSS (Mean Absolute Error)',
    '',
    'L_1 = 1/N * SUM|y_i - y_tilde_i|',
    '',
    'Role: Regularisation spatiale + smoothness',
    'Effet: Moins sensible aux outliers que L2, preserve details fins',
]

text_recon = '\n'.join(equations_recon)
ax2.text(0.05, 0.95, text_recon, transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))

# ============================================================================
# Slide 3: Combined Generator Loss
# ============================================================================
ax3 = axes[2]
ax3.axis('off')

equations_combined = [
    'LOSS COMBINEE DU GENERATEUR',
    '',
    'L_G = L_G_adv + lambda_L1 * (L_BCE + 0.5 * L_1)',
    '',
    'L_G = E[max(0, 1 - D(x, G(x)))] + 2.0 * (L_BCE + 0.5 * L_1)',
    '',
    'Composants et roles:',
    '',
    '  Adversarial: E[max(0, 1 - D(...))]  => Tromper D',
    '  Classification: L_BCE              => Classification binaire',
    '  Regularisation: 0.5 * L_1           => Smoothness pixels',
    '',
    'Hyperparametres:',
    '  lambda_L1 = 2.0      (equilibre reconstruction vs adversarial)',
    '  label_smooth = 0.2   (regularise discriminateur)',
]

text_combined = '\n'.join(equations_combined)
ax3.text(0.05, 0.95, text_combined, transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

plt.tight_layout()
plt.savefig('Loss_Functions_Equations.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Équations sauvegardées: Loss_Functions_Equations.png")

# ============================================================================
# Create a second figure with comparison and gradient analysis
# ============================================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
fig2.suptitle('Analyse Détaillée: Gradient et Comparaison', fontsize=20, fontweight='bold', y=0.995)

# Comparison
ax_comp = axes2[0]
ax_comp.axis('off')

comparison = [
    'HINGE LOSS vs BCE LOSS - Pourquoi HINGE?',
    '',
    'Binary Cross-Entropy (Traditionnel):',
    'L_BCE_disc = -E[log D(x,y)] - E[log(1-D(x,G(x)))]',
    '',
    'PROBLEME: Quand D converge, gradient => 0 (saturation sigmoide)',
    'RESULTAT: G arrete d\'apprendre car ne recoit plus de signal',
    '',
    'Hinge Loss (Notre implementation):',
    'L_Hinge_disc = E[max(0, 1 - D(x,y))] + E[max(0, 1 + D(x,G(x)))]',
    '',
    'AVANTAGE: Gradient reste CONSTANT meme si D converge',
    'RESULTAT: Apprentissage stable, pas de plateau d\'apprentissage',
]

text_comp = '\n'.join(comparison)
ax_comp.text(0.05, 0.95, text_comp, transform=ax_comp.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

# Gradient behavior
ax_grad = axes2[1]
ax_grad.axis('off')

gradient_analysis = [
    r'\textbf{ANALYSE DES GRADIENTS}',
    '',
    r'\textbf{Gradient Hinge Loss:}',
    r'$\frac{\partial}{\partial D} \max(0, 1 - D) = \begin{cases} -1 & \text{si } D < 1 \\ 0 & \text{si } D \geq 1 \end{cases}$',
    '',
    r'\quad \textcolor{green}{\textbf{Propriété:}} Constant et non-zéro tant que $D < 1$ (apprentissage continu)',
    '',
    r'\textbf{Gradient BCE Loss:}',
    r'$\frac{\partial}{\partial p} BCE = \frac{p - y}{p(1-p)}$',
    '',
    r'\quad \textcolor{red}{\textbf{Problème:}} Quand $p \rightarrow 0$ ou $p \rightarrow 1$, gradient $\rightarrow 0$ (saturation)',
    '',
    r'\textbf{Résumé des valeurs typiques pendant entraînement:}',
    r'\quad GenLoss: 2.0-5.0 \quad | \quad L1Loss: 0.5-0.8 \quad | \quad DiscLoss: décroit de 1.0 à 0.0',
]

text_grad = '\n'.join(gradient_analysis)
ax_grad.text(0.05, 0.95, text_grad, transform=ax_grad.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#E0F2F1', alpha=0.8))

plt.tight_layout()
plt.savefig('Loss_Analysis_Gradients.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Analyse gradients sauvegardée: Loss_Analysis_Gradients.png")

print("\n✓ Tous les fichiers d'équations ont été générés avec succès!")
print("  - Loss_Functions_Equations.png")
print("  - Loss_Analysis_Gradients.png")
print("  - LOSS_EQUATIONS.md (document texte)")

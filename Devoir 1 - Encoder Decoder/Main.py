import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from Network import Autoencoder
import matplotlib.pyplot as plt

def prepare_data(train_path, test_path):

    column_names = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "classe"]
    train_data = pd.read_csv(train_path, sep=' ', names=column_names)
    test_data = pd.read_csv(test_path, sep=' ', names=column_names)

    X_train = train_data[train_data['classe'] == 1].drop(columns=['classe']).values
    # Normalize Test and Train values
    means = np.mean(X_train, axis=0) # Avec axis = 0, on calcule means et stds de chaque colonne s1,s2...
    stds = np.std(X_train, axis=0)
    X_train_norm = (X_train - means) / (stds + 1e-8)

    X_test = test_data.drop(columns=['classe']).values  # Retirer la dernière colonne
    X_test_norm = (X_test - means) / (stds + 1e-8)      # normaliser valeurs de test aussi, car modèle entrainé sur valeurs normalisées
                                                        # utiliser means et stds trouvés sur X_train sur X_test pour garder mm echelle de normalisation
    y_test = test_data['classe'].values   # Garder les étiquettes de classe pour calculer F-mesure plus tard

    return X_train_norm, X_test_norm, y_test, means, stds

# Load data
# Should use a data loader
X_train_norm, X_test_norm, y_test, means, stds = prepare_data('dataset/shuttle.trn', 'dataset/shuttle.tst')

# Convertir en tenseur PyTorch
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

# Initialize
model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Choisir Adam car utilise effet d'inertie, se base sur pentes précédentes
criterion = torch.nn.MSELoss()
batch_size = 256

# Train loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    #Feed batch size to model not all of them
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)  # Calcul diff entre entrée et sortie

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("Entraînement terminé. Calcul du seuil optimal...")

# Test mode
model.eval()
with torch.no_grad():
    reconstruction = model(X_test_tensor)
    L = torch.mean((X_test_tensor - reconstruction) ** 2, dim=1).numpy()

# Tracer seuil vs F1
thresholds = np.linspace(np.min(L), np.max(L), 200)
f1_scores = []
y_true = (y_test != 1).astype(int)

# 3. Calculer la F1 pour chaque seuil
for T0 in thresholds:
    y_pred = (L >= T0).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    f1_scores.append(f1)

# 4. Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1-Score', color='blue', linewidth=2)
plt.axvline(thresholds[np.argmax(f1_scores)], color='red', linestyle='--',
                label=f'Best T0: {thresholds[np.argmax(f1_scores)]:.4f}')

plt.title('Évolution de la F-mesure en fonction du seuil T0')
plt.xlabel('Seuil (Erreur de reconstruction L)')
plt.ylabel('Score F1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Barre de progression (pip install tqdm)

# Import de tes autres fichiers
from v1.dataset import load_and_process_data
from v1.model import ShakespeareTransformer

# --- HYPERPARAMÈTRES (Selon PDF Page 8) ---
D_MODEL = 512  # D
DIM_FF = 512  # F (Dimension intermédiaire)
SEQ_LEN = 64  # N
NUM_HEADS = 8  # H
NUM_LAYERS = 4  # K
BATCH_SIZE = 64  # B
LEARNING_RATE = 1e-4  # Classique pour Adam
EPOCHS = 5  # À ajuster selon ta patience

# Sélection du device (GPU ou CPU)
# Remplacer la ligne device = ... par :
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")

# 1. Chargement des données
print("Chargement des données...")
train_dataset, val_dataset, test_dataset, vocab_size, w2i, i2w = load_and_process_data(
    '../tiny_shakespeare.txt',
    seq_len=SEQ_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# 2. Instanciation du Modèle
model = ShakespeareTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FF,
    max_len=SEQ_LEN
).to(device)

print(f"Modèle créé avec {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M paramètres")

# 3. Loss et Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Boucle d'entraînement
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for x_batch, y_batch in progress_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward
        # logits shape: (Batch, Seq, Vocab)
        logits = model(x_batch)

        # Calcul de la Loss
        # PyTorch CrossEntropy veut (Batch*Seq, Vocab) vs (Batch*Seq)
        # On aplatit les dimensions Batch et Seq
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            logits = model(x_val)
            loss = criterion(logits.view(-1, vocab_size), y_val.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"FIN EPOCH {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 5. Sauvegarde
torch.save(model.state_dict(), "shakespeare_model.pth")
print("Modèle sauvegardé !")
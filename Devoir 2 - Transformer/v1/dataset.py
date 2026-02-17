import torch
from torch.utils.data import Dataset, DataLoader
import re


class ShakespeareDataset(Dataset):
    def __init__(self, token_ids, seq_len=64):
        """
        Args:
            token_ids (list[int]): La liste complète du texte converti en indices.
            seq_len (int): La longueur de la séquence d'entrée (N=64).
        """
        self.seq_len = seq_len
        self.samples = []

        # --- ÉTAPE 4 : Création des séquences avec chevauchement ---
        # Consigne : Séquence de 64, chevauchement de 50% (donc saut de 32)
        # On a besoin de seq_len + 1 jetons pour avoir (Input, Target)
        # Input: mots [0..63], Target: mots [1..64]
        stride = seq_len // 2  # 32

        # On boucle sur le texte avec un pas de 32
        for i in range(0, len(token_ids) - seq_len, stride):
            # On s'assure d'avoir assez de données pour input + target (seq_len + 1)
            chunk = token_ids[i: i + seq_len + 1]

            if len(chunk) == seq_len + 1:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # On récupère le chunk de taille 65 (64 input + 1 target)
        chunk = self.samples[idx]

        # Input (x): Les 64 premiers
        x = chunk[:-1]
        # Target (y): Les 64 suivants (décalés de 1)
        y = chunk[1:]

        return x, y


def load_and_process_data(filepath, seq_len=64, train_split=0.70, val_split=0.15):
    """
    Fonction principale pour charger, nettoyer et diviser les données.
    """

    # --- ÉTAPE 1 : Download/Load text en string ---
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- Nettoyage selon le PDF (Page 3) ---
    # "éliminer la ponctuation et convertir la casse en minuscule"
    # Utilisation de l'expression régulière suggérée
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)

    print(f"Nombre total de mots (jetons) trouvés : {len(tokens)}")

    # --- Création du Dictionnaire (Page 4) ---
    # On crée le vocabulaire basé sur TOUT le texte (ou juste le train,
    # mais pour ce devoir simple, tout le texte est souvent toléré)
    vocab = sorted(list(set(tokens)))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    vocab_size = len(vocab)
    print(f"Taille du vocabulaire : {vocab_size}")

    # Conversion de tout le texte en indices numériques
    full_data_indices = [word_to_idx[w] for w in tokens]

    # --- ÉTAPE 2 : Division (Train / Valid / Test) ---
    n = len(full_data_indices)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    # Le reste va au test

    train_data = full_data_indices[:n_train]
    val_data = full_data_indices[n_train: n_train + n_val]
    test_data = full_data_indices[n_train + n_val:]

    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # --- ÉTAPE 3 : Création des Datasets ---
    train_dataset = ShakespeareDataset(train_data, seq_len)
    val_dataset = ShakespeareDataset(val_data, seq_len)
    test_dataset = ShakespeareDataset(test_data, seq_len)

    return train_dataset, val_dataset, test_dataset, vocab_size, word_to_idx, idx_to_word
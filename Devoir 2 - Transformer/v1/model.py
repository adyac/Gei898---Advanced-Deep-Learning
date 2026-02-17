import torch
import torch.nn as nn
import math


class ShakespeareTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=512, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 1. Embeddings (Jetons + Position)
        # Le PDF demande des "plongements de positions qui seront APPRIS"
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # 2. Le cœur du Transformer
        # On utilise EncoderLayer car on n'a pas de cross-attention
        # batch_first=True permet d'avoir des entrées (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,  # F=512 selon le PDF
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. La tête de sortie (Projection vers le vocabulaire)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        B, N = x.shape

        # A. Création des embeddings
        # Positions: [0, 1, 2, ..., N-1] pour chaque batch
        positions = torch.arange(0, N, device=x.device).unsqueeze(0).expand(B, N)

        # Somme des embeddings (Token + Position) * sqrt(d_model) est une astuce courante mais optionnelle
        x = self.token_embedding(x) + self.position_embedding(positions)

        # B. Création du Masque Causal (Crucial !)
        # Empêche la position t de voir t+1
        # mask shape: (N, N) avec -inf au dessus de la diagonale
        mask = nn.Transformer.generate_square_subsequent_mask(N).to(x.device)

        # C. Passage dans le Transformer
        # is_causal=True est une optimisation pour les nouvelles versions de PyTorch
        out = self.transformer_encoder(x, mask=mask, is_causal=True)

        # D. Prédiction
        logits = self.output_head(out)

        return logits
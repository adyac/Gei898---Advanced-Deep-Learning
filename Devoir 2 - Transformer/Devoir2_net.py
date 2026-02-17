import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Devoir2_Net(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_lenN: int = 64,
        embedding_dimensionD: int = 512,
        nb_headsH: int = 8,
        nb_transformerlayersK: int = 4,
        interdimF: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embedding_dimensionD % nb_headsH == 0, "embedding_dimension must be divisible by nb_heads"
        self.vocab_size = vocab_size
        self.seq_lenN = seq_lenN
        self.embed_dim = embedding_dimensionD
        self.nb_heads = nb_headsH

        self.nb_transformerLayersK = nb_transformerlayersK
        self.interdimF = interdimF
        self.dropout = dropout

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(self.seq_lenN, self.embed_dim)


        # Transformer decoder
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.nb_heads,
            dim_feedforward=self.interdimF,
            dropout=self.dropout,
            batch_first=True,  # (B, N, D)
            norm_first=False
        )
        self.transformerDecoder = nn.TransformerDecoder(self.dec_layer, num_layers=self.nb_transformerLayersK)
        self.head = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self,x):
        batch_size, seq_len = x.shape
        assert seq_len <= self.seq_lenN, f"sequence length cannot exceed the maximum sequence length allowed ({self.seq_lenN})"
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)  # shape (batch_size, seq_len)

        # transformerDecoder params
        memory = torch.zeros(batch_size, 1, self.embed_dim).to(device = x.device)
        tgt_mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        tgt_mask = torch.triu(tgt_mask, diagonal = 1)

        embedded_words = self.token_embedding(x)
        embedded_positions = self.position_embedding(positions)

        embedded_sequence = embedded_positions + embedded_words
        memory = memory.to(dtype=embedded_sequence.dtype)

        z = self.transformerDecoder(tgt=embedded_sequence, memory=memory, tgt_mask=tgt_mask)
        logits = self.head(z)
        return logits







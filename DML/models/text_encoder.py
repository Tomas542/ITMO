import torch
from abc_encoder import Encoder
from sentence_transformers import SentenceTransformer
from torch import nn


class TextEncoder(nn.Module, Encoder):
    def __init__(self, weight_path: str = "johnnyboycurtis/ModernBERT-small-v2") -> None:
        super().__init__()
        self.enc = SentenceTransformer(weight_path)

    def forward(self, text: list[str]) -> torch.Tensor:
        emb = self.enc.encode(text)
        return emb.unsqueeze(1)

    def encode(self, text: list[str]) -> torch.Tensor:
        """Encoder for text features

        Args:
            text (list[str]) - text input data

        Returns:
            torch.Tensor: embedding
        """
        return self(text)

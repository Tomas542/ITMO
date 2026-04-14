from pathlib import Path

import torch
from abc_encoder import Encoder
from audio_preprocessor import AudioToMelSpectrogramPreprocessor
from conformer import ConformerEncoder
from safetensors.torch import load_model
from torch import nn


class AudioEncoder(nn.Module, Encoder):
    def __init__(self, wegith_path: Path = Path("weights/")) -> None:
        super().__init__()
        self.enc = ConformerEncoder()
        self.proc = AudioToMelSpectrogramPreprocessor()
        self.load_weights(wegith_path)

    def forward(self, wavs: torch.Tensor, unpaded_length: list[int]) -> torch.Tensor:
        assert len(wavs.shape) == 3, "Audio should be [batch, channels, length] size"
        processed_features, length = self.proc.get_features(wavs, unpaded_length)
        emb, _ = self.enc(processed_features, length)
        return emb.transpose(1, 2)

    def encode(self, wavs: torch.Tensor, unpaded_length: list[int]) -> torch.Tensor:
        """Encoder for audio features

        Args:
            wavs (torch.Tensor) - wavs of shape [batch, channels, length]
            unpaded_length (list[int]) - list of audio lengths before padding

        Returns:
            torch.Tensor: embedding
        """
        return self(wavs, unpaded_length)

    def load_weights(self, path: Path) -> None:
        load_model(self.proc, path.joinpath("audio_preprocessor.safetensors"))
        load_model(self.enc, path.joinpath("conformer.safetensors"))

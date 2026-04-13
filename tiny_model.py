"""
Shared model architectures and utilities for VAD (Voice Activity Detection).
"""

import torch
import torch.nn as nn
import torchaudio


class TinyVAD(nn.Module):
    """
    Lightweight VAD model: CNN (spectral) + GRU (temporal) + DNN (classification).
    Used in both distillation (distill.py) and fine-tuning (finetune_torgo.py).
    """
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout2d(0.1)
        )
        self.gru = nn.GRU(
            input_size=160,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.output = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) mel-spectrogram batch
        Returns:
            (B, T) speech probability for each frame
        """
        B, T, F = x.shape
        x = x.unsqueeze(1)                     # (B, 1, T, F)
        x = self.cnn(x)                        # (B, 16, T, F//4)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3)             # (B, T, C, F//4)
        x = x.reshape(B, T, C * F)            # (B, T, 160)
        x, _ = self.gru(x)                    # (B, T, 16)
        x = self.output(x)                    # (B, T, 1)
        return torch.sigmoid(x).squeeze(-1)   # (B, T)


# ============================================================
# Feature Extraction Utilities
# ============================================================
def get_mel_transform(sample_rate: int, n_mels: int, hop_length: int = 160):
    """
    Create MelSpectrogram transform.
    Args:
        sample_rate: Sample rate in Hz (e.g., 16000)
        n_mels: Number of mel bins (e.g., 40)
        hop_length: Samples per frame (default 160 = 10ms at 16kHz)
    Returns:
        torchaudio.transforms.MelSpectrogram instance
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=hop_length,
        n_mels=n_mels
    )


def extract_features(wav: torch.Tensor, mel_transform) -> torch.Tensor:
    """
    Extract mel-spectrogram from waveform.
    Args:
        wav: (T,) audio waveform
        mel_transform: MelSpectrogram transform instance
    Returns:
        (time_frames, n_mels) mel-spectrogram
    """
    with torch.no_grad():
        mel = mel_transform(wav)       # (n_mels, time_frames)
        mel = (mel + 1e-6).log()
        mel = mel.T                    # (time_frames, n_mels)
    return mel


def collate_vad_batch(batch, n_mels: int):
    """Pad variable-length (mel, label) sequences to the same length."""
    mels, labels = zip(*batch)
    max_len = max(m.shape[0] for m in mels)
    batch_size = len(mels)

    mel_pad = torch.zeros(batch_size, max_len, n_mels)
    label_pad = torch.zeros(batch_size, max_len)

    for i, (m, l) in enumerate(zip(mels, labels)):
        t = m.shape[0]
        mel_pad[i, :t] = m
        label_pad[i, :t] = l

    return mel_pad, label_pad

"""
Knowledge Distillation: CRDNN (Teacher) -> TinyVAD (Student)
Training on LibriParty data first, TORGO added later
"""

from functools import partial
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from tiny_model import TinyVAD, get_mel_transform, extract_features, collate_vad_batch

# ============================================================
# Config
# ============================================================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS      = 40
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
TEMPERATURE = 3.0       # Distillation temperature (Hinton et al. 2015 recommend T in [2,5])
ALPHA       = 0.3       # 0.3 * hard + 0.7 * soft (more weight on soft labels)
DATA_DIR    = "/courses/CS6140.202630/students/cong.m/LibriParty/dataset"
SAVE_DIR    = "/courses/CS6140.202630/students/cong.m/results/TinyVAD"
JSON_DIR    = "/courses/CS6140.202630/students/cong.m/results/VAD/save"

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ============================================================
# Feature Extraction
# ============================================================
mel_transform = get_mel_transform(SAMPLE_RATE, N_MELS)


# ============================================================
# Dataset
# ============================================================
class LibriPartyDataset(Dataset):
    """
    Load LibriParty dataset using preprocessed JSON files.
    JSON structure:
      {example_id: {
          wav: {file, start, stop},
          speech: [[start_sec, end_sec], ...]
      }}
    Each example is a 5-second chunk with frame-level labels.
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        print(f"Loaded {len(self.keys)} examples from {json_path}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]

        # Load audio chunk
        wav_info = item["wav"]
        wav, sr  = torchaudio.load(
            wav_info["file"],
            frame_offset=wav_info["start"],
            num_frames=wav_info["stop"] - wav_info["start"]
        )
        wav = wav.squeeze()
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        # Extract features
        mel = extract_features(wav, mel_transform).cpu()   # (T, n_mels)
        T   = len(mel)

        # Create frame-level labels
        # speech: [[start_sec, end_sec], ...]
        labels = torch.zeros(T)
        hop    = 0.01   # 10ms per frame
        for seg in item.get("speech", []):
            start = int(seg[0] / hop)
            end   = int(seg[1] / hop)
            labels[start:min(end, T)] = 1.0

        return mel, labels


collate_fn = partial(collate_vad_batch, n_mels=N_MELS)


# ============================================================
# Distillation Loss
# ============================================================
def distillation_loss(student_out, teacher_out, hard_labels,
                      alpha=ALPHA, T=TEMPERATURE):
    """
    Loss = alpha * BCE(hard) + (1-alpha) * BCE(soft)
    soft labels are teacher outputs divided by temperature
    """
    bce = nn.BCELoss()

    # Hard loss: compare with ground truth
    hard_loss = bce(student_out, hard_labels)

    # Soft loss: compare with teacher output
    # Temperature scaling makes distribution softer
     soft_teacher = torch.sigmoid(teacher_out / T)
    soft_student  = torch.clamp(student_out, 1e-7, 1 - 1e-7)
    soft_loss = bce(soft_student, soft_teacher.detach())

    return alpha * hard_loss + (1 - alpha) * soft_loss


# ============================================================
# Get Teacher predictions
# ============================================================
def get_teacher_predictions(hparams, mel_batch):
    """
    Get teacher predictions using same pipeline as train.py compute_forward
    mel_batch: (B, T, n_mels) - already extracted mel features
    """
    with torch.no_grad():
        x    = mel_batch.to(DEVICE)
        lens = torch.ones(x.shape[0]).to(DEVICE)

        # Normalize (mean_var_norm is in modules)
        x = hparams["modules"]["mean_var_norm"](x, lens)

        # CNN
        out = hparams["modules"]["cnn"](x)

        # Reshape for RNN
        out = out.reshape(
            out.shape[0],
            out.shape[1],
            out.shape[2] * out.shape[3]
        )

        # RNN
        out, _ = hparams["modules"]["rnn"](out)

        # DNN
        out = hparams["modules"]["dnn"](out)

        return out.squeeze(-1)   # (B, T)


# ============================================================
# Training Loop
# ============================================================
def train(hparams, student, train_loader, val_loader):

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    best_f1   = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training
        student.train()
        total_loss = 0.0

        for mel, labels in train_loader:
            mel    = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            # Student forward
            student_out = student(mel)

            # Teacher forward
            teacher_out = get_teacher_predictions(hparams, mel)

            # Distillation loss
            loss = distillation_loss(student_out, teacher_out, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        student.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for mel, labels in val_loader:
                mel = mel.to(DEVICE)
                out = student(mel).cpu()
                preds = (out > 0.5).int()
                all_preds.extend(preds.flatten().tolist())
                all_labels.extend(labels.flatten().int().tolist())

        f1   = f1_score(all_labels, all_preds, zero_division=0)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)

        scheduler.step(1 - f1)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"F1: {f1:.4f} | "
              f"Prec: {prec:.4f} | "
              f"Rec: {rec:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(student.state_dict(),
                       os.path.join(SAVE_DIR, "best_tinyvad.pt"))
            print(f"  --> Best model saved! F1={best_f1:.4f}")

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    return student


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # Load datasets using preprocessed JSON
    train_set = LibriPartyDataset(os.path.join(JSON_DIR, "train.json"))
    val_set   = LibriPartyDataset(os.path.join(JSON_DIR, "valid.json"))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    # Load Teacher
    import sys
    sys.path.insert(0, "/courses/CS6140.202630/students/cong.m")
    from hyperpyyaml import load_hyperpyyaml

    with open("/courses/CS6140.202630/students/cong.m/results/VAD/hyperparams.yaml") as f:
        hparams = load_hyperpyyaml(f)

    teacher = hparams["model"]
    ckpt = torch.load(
        "/courses/CS6140.202630/students/cong.m/results/VAD/save/CKPT+epoch_72/model.ckpt",
        map_location=DEVICE,
        weights_only=False
    )
    teacher.load_state_dict(ckpt)
    teacher.eval().to(DEVICE)

    # Move all teacher modules to GPU
    for mod in hparams["modules"].values():
        mod.to(DEVICE).eval()
    print("Teacher loaded!")

    # Init Student
    student = TinyVAD().to(DEVICE)
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")

    # Train
    student = train(hparams, student, train_loader, val_loader)
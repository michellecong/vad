"""
Fine-tune TinyVAD on TORGO dysarthric speech dataset.
Goal: improve model's ability to detect voice activity in atypical (dysarthric) speech.

Evaluation protocol: Leave-One-Speaker-Out (LOSO) cross-validation.
  - For each speaker in ALL_SPEAKERS, train on all other speakers, test on that speaker.
  - This is the standard protocol for TORGO (speaker-independent evaluation).
  - Results reported per fold + macro-average across all folds.

Labeling strategy:
  - Only files WITH .phn/.PHN phoneme annotations are used.
  - 'sil', 'noi', 'sp', 'h#' -> silence (0), all other phonemes -> speech (1)
  - Annotated sessions: F01/S1, F03/S1-S3, F04/S2, M01/S1+S2_3,
                        M02/S1, M04/S1+S2, M05/S1+S2, MC01/S3
  - phn_arrayMic pairs with wav_arrayMic; phn_headMic pairs with wav_headMic.
  - Both .phn and .PHN (uppercase) extensions are supported.
  - Files without a matching annotation file are skipped entirely.

Fine-tuning strategy:
  - Load pre-trained TinyVAD weights (from distillation) fresh for each fold.
  - Freeze CNN layers; adapt only GRU + DNN to dysarthric temporal patterns.
  - Lower LR (1e-4) to avoid catastrophic forgetting on typical speech.
"""

import os
import copy
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# ============================================================
# Config
# ============================================================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS      = 40
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-5
FREEZE_CNN  = True

TORGO_DIR  = os.path.join(os.path.dirname(__file__), "TORGO")
PRETRAINED = "/courses/CS6140.202630/students/cong.m/results/TinyVAD/best_tinyvad.pt"
SAVE_DIR   = "/courses/CS6140.202630/students/cong.m/results/TinyVAD/loso_lr5e5"

CHUNK_DUR  = 5.0
HOP_LENGTH = 160      # 10ms at 16kHz

# Phoneme labels counted as silence; 'noi' used by MC01/M02 instead of 'sil'
SILENCE_PHN = {"sil", "noi", "sp", "h#", "epi", "pau"}

# All speakers that have at least one annotated session
ALL_SPEAKERS = ["F01", "F03", "F04", "M01", "M02", "M04", "M05", "MC01"]

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ============================================================
# Feature Extraction
# ============================================================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)

def extract_features(wav: torch.Tensor) -> torch.Tensor:
    """wav: (T,) -> mel: (time_frames, n_mels)"""
    with torch.no_grad():
        mel = mel_transform(wav)
        mel = (mel + 1e-6).log()
        mel = mel.T
    return mel


# ============================================================
# Label Generation
# ============================================================
def labels_from_phn(phn_path: str, n_samples: int) -> torch.Tensor:
    """
    Parse TORGO .phn/.PHN file -> sample-level speech mask (1=speech, 0=silence).
    Format: <start_sample> <end_sample> <phoneme>
    """
    mask = torch.zeros(n_samples)
    with open(phn_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start = int(parts[0])
            end   = min(int(parts[1]), n_samples)
            if parts[2].lower() not in SILENCE_PHN and start < n_samples:
                mask[start:end] = 1.0
    return mask


def sample_mask_to_frame_labels(mask: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Downsample sample-level mask to frame-level labels (majority vote per frame)."""
    labels = torch.zeros(n_frames)
    for t in range(n_frames):
        s = t * HOP_LENGTH
        chunk = mask[s:min(s + HOP_LENGTH, len(mask))]
        if len(chunk) > 0 and chunk.mean() >= 0.5:
            labels[t] = 1.0
    return labels


# ============================================================
# File Collection
# ============================================================
def find_phn(phn_dir: str, stem: str):
    """
    Find annotation file for a wav stem. Handles:
      - Exact match:            '0001' -> '0001.PHN'  (most sessions)
      - Leading-zeros stripped: '0001' -> '1.PHN'     (M04/Session1)
    Tries both .phn and .PHN extensions.
    """
    if not os.path.isdir(phn_dir):
        return None
    for s in [stem, stem.lstrip("0") or "0"]:
        for ext in (".phn", ".PHN"):
            candidate = os.path.join(phn_dir, s + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def collect_files(torgo_dir: str, speakers: list) -> list:
    """
    Collect (wav_path, phn_path) for all annotated files of the given speakers.
    Files without a matching annotation are excluded.
    Searches Session1/2/2_3/3 and both headMic/arrayMic types.
    """
    sessions  = ["Session1", "Session2", "Session2_3", "Session3"]
    mic_types = ["headMic", "arrayMic"]
    entries   = []
    seen_wavs = set()

    for spk in speakers:
        spk_dir = os.path.join(torgo_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for sess in sessions:
            for mic in mic_types:
                wav_dir = os.path.join(spk_dir, sess, f"wav_{mic}")
                phn_dir = os.path.join(spk_dir, sess, f"phn_{mic}")
                if not os.path.isdir(wav_dir):
                    continue
                for fname in sorted(os.listdir(wav_dir)):
                    if not fname.lower().endswith(".wav"):
                        continue
                    wav_path = os.path.join(wav_dir, fname)
                    if wav_path in seen_wavs:
                        continue
                    seen_wavs.add(wav_path)
                    phn_path = find_phn(phn_dir, os.path.splitext(fname)[0])
                    if phn_path is not None:
                        entries.append((wav_path, phn_path))
    return entries


# ============================================================
# Dataset
# ============================================================
class TorgoDataset(Dataset):
    """
    Loads annotated TORGO files for a list of speakers.
    Each wav is chunked into CHUNK_DUR-second mel segments with frame-level labels.
    """
    def __init__(self, speakers: list, torgo_dir: str = TORGO_DIR):
        self.chunks = []
        entries = collect_files(torgo_dir, speakers)
        print(f"  Speakers {speakers}: {len(entries)} annotated files")

        chunk_samples = int(CHUNK_DUR * SAMPLE_RATE)

        for wav_path, phn_path in entries:
            try:
                wav, sr = torchaudio.load(wav_path)
                wav = wav.squeeze(0)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            except Exception as e:
                print(f"  Warning: skipping {wav_path}: {e}")
                continue

            n_samples   = len(wav)
            speech_mask = labels_from_phn(phn_path, n_samples)

            for i in range(max(1, n_samples // chunk_samples)):
                s         = i * chunk_samples
                wav_chunk = wav[s:min(s + chunk_samples, n_samples)]
                if len(wav_chunk) < chunk_samples:
                    wav_chunk = torch.cat(
                        [wav_chunk, torch.zeros(chunk_samples - len(wav_chunk))]
                    )
                mel        = extract_features(wav_chunk).cpu()
                T          = mel.shape[0]
                mask_chunk = speech_mask[s:s + chunk_samples]
                if len(mask_chunk) < chunk_samples:
                    mask_chunk = torch.nn.functional.pad(
                        mask_chunk, (0, chunk_samples - len(mask_chunk))
                    )
                self.chunks.append((mel, sample_mask_to_frame_labels(mask_chunk, T)))

        n_speech = sum(l.sum().item() for _, l in self.chunks)
        n_total  = sum(l.numel()      for _, l in self.chunks)
        print(f"    -> {len(self.chunks)} chunks, "
              f"speech ratio: {n_speech / max(n_total, 1):.2%}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def collate_fn(batch):
    mels, labels = zip(*batch)
    max_len   = max(m.shape[0] for m in mels)
    mel_pad   = torch.zeros(len(mels), max_len, N_MELS)
    label_pad = torch.zeros(len(mels), max_len)
    for i, (m, l) in enumerate(zip(mels, labels)):
        T = m.shape[0]
        mel_pad[i, :T]   = m
        label_pad[i, :T] = l
    return mel_pad, label_pad


# ============================================================
# TinyVAD (identical to distill.py)
# ============================================================
class TinyVAD(nn.Module):
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
            input_size=160, hidden_size=16,
            num_layers=1, batch_first=True, bidirectional=False
        )
        self.output = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        B, T, F = x.shape
        x = self.cnn(x.unsqueeze(1))
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        x, _ = self.gru(x)
        return torch.sigmoid(self.output(x)).squeeze(-1)


# ============================================================
# Fine-tuning (single fold)
# ============================================================
def finetune_fold(student: TinyVAD,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  save_path: str) -> float:
    """
    Fine-tune student for one LOSO fold.
    Returns best F1 on the held-out speaker.
    """
    if FREEZE_CNN:
        for name, param in student.named_parameters():
            if name.startswith("cnn"):
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, student.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    bce     = nn.BCELoss()
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0
        for mel, labels in train_loader:
            mel, labels = mel.to(DEVICE), labels.to(DEVICE)
            loss = bce(student(mel), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, labels in val_loader:
                pred = (student(mel.to(DEVICE)).cpu() > 0.5).int()
                all_preds.extend(pred.flatten().tolist())
                all_labels.extend(labels.flatten().int().tolist())

        f1   = f1_score(all_labels, all_preds, zero_division=0)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        scheduler.step(f1)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(student.state_dict(), save_path)
            print(f"    --> Saved (F1={best_f1:.4f})")

    return best_f1


# ============================================================
# Main: LOSO cross-validation
# ============================================================
if __name__ == "__main__":
    # Load pre-trained weights once (reloaded fresh for each fold)
    pretrained_state = None
    if os.path.isfile(PRETRAINED):
        pretrained_state = torch.load(PRETRAINED, map_location=DEVICE, weights_only=True)
        print(f"Loaded pre-trained weights from {PRETRAINED}")
    else:
        print(f"WARNING: {PRETRAINED} not found — each fold starts from random init.")

    fold_results = []   # (speaker, f1, prec, rec) for summary

    for test_spk in ALL_SPEAKERS:
        train_spks = [s for s in ALL_SPEAKERS if s != test_spk]
        print(f"\n{'='*60}")
        print(f"Fold: test={test_spk}  train={train_spks}")
        print(f"{'='*60}")

        train_set = TorgoDataset(train_spks)
        test_set  = TorgoDataset([test_spk])

        if len(train_set) == 0 or len(test_set) == 0:
            print(f"  Skipping fold (empty dataset).")
            continue

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_fn, num_workers=0)
        test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Fresh model for each fold
        student = TinyVAD().to(DEVICE)
        if pretrained_state is not None:
            student.load_state_dict(copy.deepcopy(pretrained_state))

        save_path = os.path.join(SAVE_DIR, f"tinyvad_loso_{test_spk}.pt")
        best_f1   = finetune_fold(student, train_loader, test_loader, save_path)

        # Final evaluation on test speaker with best checkpoint
        student.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, labels in test_loader:
                pred = (student(mel.to(DEVICE)).cpu() > 0.5).int()
                all_preds.extend(pred.flatten().tolist())
                all_labels.extend(labels.flatten().int().tolist())

        f1   = f1_score(all_labels, all_preds, zero_division=0)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        fold_results.append((test_spk, f1, prec, rec))
        print(f"  Final test: F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"LOSO Summary")
    print(f"{'='*60}")
    print(f"{'Speaker':<12} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print(f"{'-'*40}")
    for spk, f1, prec, rec in fold_results:
        print(f"{spk:<12} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")
    if fold_results:
        avg_f1   = sum(r[1] for r in fold_results) / len(fold_results)
        avg_prec = sum(r[2] for r in fold_results) / len(fold_results)
        avg_rec  = sum(r[3] for r in fold_results) / len(fold_results)
        print(f"{'-'*40}")
        print(f"{'Macro avg':<12} {avg_f1:>8.4f} {avg_prec:>8.4f} {avg_rec:>8.4f}")

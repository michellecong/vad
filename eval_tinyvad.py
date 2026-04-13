"""
Catastrophic Forgetting Evaluation for TinyVAD after TORGO fine-tuning.

Compares F1 on LibriParty val set between:
  - Pre-trained model (before TORGO fine-tuning)
  - Each LOSO fold checkpoint (after fine-tuning on all-but-one TORGO speakers)

Also reports TORGO test F1 for each fold, so you can see the trade-off:
  LibriParty F1 drop  vs.  TORGO F1 gain

Usage:
  python eval_forgetting.py
"""

import os
import json
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# ============================================================
# Paths  (edit if your paths differ)
# ============================================================
LIBRIPARTY_VAL_JSON = "/courses/CS6140.202630/students/cong.m/results/VAD/save/valid.json"
PRETRAINED          = "/courses/CS6140.202630/students/cong.m/results/TinyVAD/best_tinyvad.pt"
# LOSO_DIR          = "/courses/CS6140.202630/students/cong.m/results/TinyVAD/loso"          # lr=1e-4
# LOSO_DIR          = "/courses/CS6140.202630/students/cong.m/results/TinyVAD/loso_lr5e5"   # lr=5e-5
LOSO_DIR            = "/courses/CS6140.202630/students/cong.m/results/TinyVAD/loso_mixed"   # mixed replay
TORGO_DIR           = os.path.join(os.path.dirname(__file__), "TORGO")

# Teacher (CRDNN) paths — same as distill.py
TEACHER_HPARAMS = "/courses/CS6140.202630/students/cong.m/results/VAD/hyperparams.yaml"
TEACHER_CKPT    = "/courses/CS6140.202630/students/cong.m/results/VAD/save/CKPT+epoch_72/model.ckpt"

# ============================================================
# Config (must match finetune_torgo.py)
# ============================================================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS      = 40
BATCH_SIZE  = 32
HOP_LENGTH  = 160
CHUNK_DUR   = 5.0
SILENCE_PHN = {"sil", "noi", "sp", "h#", "epi", "pau"}
ALL_SPEAKERS = ["F01", "F03", "F04", "M01", "M02", "M04", "M05", "MC01"]

# ============================================================
# Feature Extraction
# ============================================================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=400, hop_length=HOP_LENGTH, n_mels=N_MELS
)

def extract_features(wav: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mel = mel_transform(wav)
        mel = (mel + 1e-6).log()
        return mel.T


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
# LibriParty Dataset  (mirrors distill.py)
# ============================================================
class LibriPartyDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item    = self.data[self.keys[idx]]
        wav_info = item["wav"]
        wav, sr = torchaudio.load(
            wav_info["file"],
            frame_offset=wav_info["start"],
            num_frames=wav_info["stop"] - wav_info["start"]
        )
        wav = wav.squeeze()
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        mel    = extract_features(wav).cpu()
        T      = len(mel)
        labels = torch.zeros(T)
        for seg in item.get("speech", []):
            s = int(seg[0] / 0.01)
            e = int(seg[1] / 0.01)
            labels[s:min(e, T)] = 1.0
        return mel, labels


# ============================================================
# TORGO Dataset  (mirrors finetune_torgo.py)
# ============================================================
def find_phn(phn_dir, stem):
    if not os.path.isdir(phn_dir):
        return None
    for s in [stem, stem.lstrip("0") or "0"]:
        for ext in (".phn", ".PHN"):
            c = os.path.join(phn_dir, s + ext)
            if os.path.isfile(c):
                return c
    return None


def collect_files(speakers):
    sessions  = ["Session1", "Session2", "Session2_3", "Session3"]
    mic_types = ["headMic", "arrayMic"]
    entries, seen = [], set()
    for spk in speakers:
        spk_dir = os.path.join(TORGO_DIR, spk)
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
                    if wav_path in seen:
                        continue
                    seen.add(wav_path)
                    phn_path = find_phn(phn_dir, os.path.splitext(fname)[0])
                    if phn_path is not None:
                        entries.append((wav_path, phn_path))
    return entries


def sample_mask_to_frame_labels(mask, n_frames):
    labels = torch.zeros(n_frames)
    for t in range(n_frames):
        s = t * HOP_LENGTH
        chunk = mask[s:min(s + HOP_LENGTH, len(mask))]
        if len(chunk) > 0 and chunk.mean() >= 0.5:
            labels[t] = 1.0
    return labels


class TorgoSpeakerDataset(Dataset):
    def __init__(self, speakers):
        self.chunks  = []
        chunk_samples = int(CHUNK_DUR * SAMPLE_RATE)
        for wav_path, phn_path in collect_files(speakers):
            try:
                wav, sr = torchaudio.load(wav_path)
                wav = wav.squeeze(0)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            except Exception:
                continue
            n_samples   = len(wav)
            speech_mask = torch.zeros(n_samples)
            with open(phn_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    s, e = int(parts[0]), min(int(parts[1]), n_samples)
                    if parts[2].lower() not in SILENCE_PHN and s < n_samples:
                        speech_mask[s:e] = 1.0
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

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ============================================================
# TinyVAD  (identical to finetune_torgo.py)
# ============================================================
class TinyVAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=(1, 2)), nn.Dropout2d(0.1)
        )
        self.gru    = nn.GRU(160, 16, num_layers=1, batch_first=True, bidirectional=False)
        self.output = nn.Sequential(nn.Linear(16, 8), nn.LeakyReLU(0.01), nn.Linear(8, 1))

    def forward(self, x):
        B, T, F = x.shape
        x = self.cnn(x.unsqueeze(1))
        B, C, T, F = x.shape
        x, _ = self.gru(x.permute(0, 2, 1, 3).reshape(B, T, C * F))
        return torch.sigmoid(self.output(x)).squeeze(-1)


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate(model: TinyVAD, loader: DataLoader):
    """Evaluate a TinyVAD model; returns (f1, prec, rec)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, labels in loader:
            pred = (model(mel.to(DEVICE)).cpu() > 0.5).int()
            all_preds.extend(pred.flatten().tolist())
            all_labels.extend(labels.flatten().int().tolist())
    return (f1_score(all_labels, all_preds, zero_division=0),
            precision_score(all_labels, all_preds, zero_division=0),
            recall_score(all_labels, all_preds, zero_division=0))


def evaluate_teacher(hparams: dict, loader: DataLoader):
    """Evaluate the CRDNN teacher model; returns (f1, prec, rec)."""
    for mod in hparams["modules"].values():
        mod.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, labels in loader:
            x    = mel.to(DEVICE)
            lens = torch.ones(x.shape[0]).to(DEVICE)
            x    = hparams["modules"]["mean_var_norm"](x, lens)
            out  = hparams["modules"]["cnn"](x)
            out  = out.reshape(out.shape[0], out.shape[1], out.shape[2] * out.shape[3])
            out, _ = hparams["modules"]["rnn"](out)
            out  = hparams["modules"]["dnn"](out)
            pred = (torch.sigmoid(out.squeeze(-1)).cpu() > 0.5).int()
            all_preds.extend(pred.flatten().tolist())
            all_labels.extend(labels.flatten().int().tolist())
    return (f1_score(all_labels, all_preds, zero_division=0),
            precision_score(all_labels, all_preds, zero_division=0),
            recall_score(all_labels, all_preds, zero_division=0))


def load_model(ckpt_path: str) -> TinyVAD:
    model = TinyVAD().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model


def load_teacher():
    """Load CRDNN teacher using same method as distill.py. Returns hparams dict."""
    from hyperpyyaml import load_hyperpyyaml  # installed on the cluster
    with open(TEACHER_HPARAMS) as f:
        hparams = load_hyperpyyaml(f)
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=False)
    hparams["model"].load_state_dict(ckpt)
    for mod in hparams["modules"].values():
        mod.to(DEVICE).eval()
    return hparams


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # --- Shared TORGO loader (all speakers, for global baselines) ---
    print("Loading TORGO (all annotated speakers)...")
    torgo_all = TorgoSpeakerDataset(ALL_SPEAKERS)
    torgo_all_loader = DataLoader(torgo_all, batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_fn, num_workers=0)

    # --- LibriParty val loader ---
    print("Loading LibriParty val set...")
    libri_val    = LibriPartyDataset(LIBRIPARTY_VAL_JSON)
    libri_loader = DataLoader(libri_val, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"  {len(libri_val)} examples")

    # ----------------------------------------------------------------
    # Baseline 1: pre-trained TinyVAD (before TORGO fine-tuning)
    # ----------------------------------------------------------------
    print("\n[Baseline] Pre-trained TinyVAD ...")
    base_model = load_model(PRETRAINED)
    base_libri_f1, base_libri_prec, base_libri_rec = evaluate(base_model, libri_loader)
    base_torgo_f1, base_torgo_prec, base_torgo_rec = evaluate(base_model, torgo_all_loader)
    print(f"  LibriParty val : F1={base_libri_f1:.4f}  Prec={base_libri_prec:.4f}  Rec={base_libri_rec:.4f}")
    print(f"  TORGO (all)    : F1={base_torgo_f1:.4f}  Prec={base_torgo_prec:.4f}  Rec={base_torgo_rec:.4f}")

    # ----------------------------------------------------------------
    # Baseline 2: CRDNN teacher (upper bound / ceiling)
    # ----------------------------------------------------------------
    teacher_torgo_f1 = None
    if os.path.isfile(TEACHER_HPARAMS) and os.path.isfile(TEACHER_CKPT):
        print("\n[Ceiling] CRDNN teacher ...")
        hparams = load_teacher()
        teacher_torgo_f1, teacher_torgo_prec, teacher_torgo_rec = evaluate_teacher(
            hparams, torgo_all_loader
        )
        teacher_libri_f1, teacher_libri_prec, teacher_libri_rec = evaluate_teacher(
            hparams, libri_loader
        )
        print(f"  LibriParty val : F1={teacher_libri_f1:.4f}  Prec={teacher_libri_prec:.4f}  Rec={teacher_libri_rec:.4f}")
        print(f"  TORGO (all)    : F1={teacher_torgo_f1:.4f}  Prec={teacher_torgo_prec:.4f}  Rec={teacher_torgo_rec:.4f}")
    else:
        print("\n[Ceiling] Teacher checkpoint not found, skipping.")

    # ----------------------------------------------------------------
    # Per-fold LOSO evaluation
    # ----------------------------------------------------------------
    results = []  # (speaker, torgo_f1, libri_f1, libri_drop, torgo_gain)

    for test_spk in ALL_SPEAKERS:
        ckpt = os.path.join(LOSO_DIR, f"tinyvad_loso_{test_spk}.pt")
        if not os.path.isfile(ckpt):
            print(f"\n[{test_spk}] checkpoint not found, skipping: {ckpt}")
            continue

        print(f"\n[{test_spk}] Loading {ckpt}")
        model = load_model(ckpt)

        # TORGO: held-out speaker only (true LOSO test)
        torgo_spk_data   = TorgoSpeakerDataset([test_spk])
        torgo_spk_loader = DataLoader(torgo_spk_data, batch_size=BATCH_SIZE,
                                      shuffle=False, collate_fn=collate_fn, num_workers=0)

        # Pre-trained model on same held-out speaker (to measure per-speaker gain)
        pre_spk_f1, _, _ = evaluate(base_model, torgo_spk_loader)

        torgo_f1, torgo_prec, torgo_rec = evaluate(model, torgo_spk_loader)
        libri_f1, libri_prec, libri_rec = evaluate(model, libri_loader)
        libri_drop  = base_libri_f1 - libri_f1
        torgo_gain  = torgo_f1 - pre_spk_f1

        print(f"  TORGO  ({test_spk}): F1={torgo_f1:.4f}  Prec={torgo_prec:.4f}  Rec={torgo_rec:.4f}"
              f"  (gain vs pre-trained: {torgo_gain:+.4f})")
        print(f"  LibriParty val  : F1={libri_f1:.4f}  Prec={libri_prec:.4f}  Rec={libri_rec:.4f}"
              f"  (drop: {libri_drop:+.4f})")

        results.append((test_spk, pre_spk_f1, torgo_f1, torgo_gain, libri_f1, libri_drop))

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    W = 75
    print(f"\n{'='*W}")
    print("Full Evaluation Summary")
    print(f"{'='*W}")
    print(f"  Pre-trained TinyVAD  — LibriParty F1: {base_libri_f1:.4f} | TORGO F1 (all): {base_torgo_f1:.4f}")
    if teacher_torgo_f1 is not None:
        print(f"  CRDNN teacher        — LibriParty F1: {teacher_libri_f1:.4f} | TORGO F1 (all): {teacher_torgo_f1:.4f}  [ceiling]")
    print(f"{'='*W}")
    print(f"{'Speaker':<10} {'Pre F1':>8} {'Post F1':>9} {'Gain':>7}  {'LibriParty':>10} {'Drop':>7}")
    print(f"{'':10} {'(TORGO)':>8} {'(TORGO)':>9} {'':>7}  {'F1 after':>10} {'':>7}")
    print(f"{'-'*W}")
    for spk, pre_f1, post_f1, gain, libri_f1, drop in results:
        forget_flag = "  !" if drop > 0.05 else ""
        print(f"{spk:<10} {pre_f1:>8.4f} {post_f1:>9.4f} {gain:>+7.4f}  "
              f"{libri_f1:>10.4f} {drop:>+7.4f}{forget_flag}")
    if results:
        print(f"{'-'*W}")
        print(f"{'Macro avg':<10} "
              f"{sum(r[1] for r in results)/len(results):>8.4f} "
              f"{sum(r[2] for r in results)/len(results):>9.4f} "
              f"{sum(r[3] for r in results)/len(results):>+7.4f}  "
              f"{sum(r[4] for r in results)/len(results):>10.4f} "
              f"{sum(r[5] for r in results)/len(results):>+7.4f}")
    print(f"{'='*W}")
    print("Drop > 0.05 (marked !) suggests significant catastrophic forgetting.")
    print("Gain = fine-tuned TORGO F1 - pre-trained TORGO F1 (same held-out speaker).")

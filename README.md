# TinyVAD

Lightweight Voice Activity Detection via knowledge distillation from a SpeechBrain CRDNN teacher, with personalisation for dysarthric speech (TORGO dataset).

---

## Part 1 — Teacher Model Training

The teacher (CRDNN) is trained with SpeechBrain on the LibriParty corpus. The recipe follows the standard SpeechBrain VAD pipeline and is not detailed here.

| File | Role |
|------|------|
| `hparams/train.yaml` | SpeechBrain hyperparameter file for teacher training |
| `commonlanguage_prepare.py` | Data preparation for Common Language corpus |
| `libriparty_prepare.py` | Data preparation for LibriParty corpus |
| `musan_prepare.py` | Data preparation for MUSAN noise corpus |
| `data_augment.py` | Augmentation utilities (noise, RIR) used during teacher training |
| `train.py` | SpeechBrain training script for the CRDNN teacher |

---

## Part 2 — Knowledge Distillation

The goal is to compress the CRDNN teacher (109,700 parameters) into a much smaller student model suitable for edge deployment.

### Model: `tiny_model.py`

Defines **TinyVAD** (9,937 parameters, 11× smaller than the teacher) and shared feature extraction utilities used across all downstream scripts.

Architecture: `CNN (1→8→16 channels, MaxPool) → GRU (hidden=16, unidirectional) → DNN (16→8→1)`

Input: 40-mel log-mel spectrogram, 10 ms hop (16 kHz).

### Training: `distill.py`

Trains TinyVAD on LibriParty using knowledge distillation (Hinton et al. 2015):

- **Soft loss** — BCE against temperature-scaled teacher outputs (T=3.0), encouraging the student to mimic the teacher's confidence distribution
- **Hard loss** — BCE against ground-truth VAD labels
- Combined as `loss = 0.3 * hard + 0.7 * soft`
- Adam optimiser, lr=1e-3, ReduceLROnPlateau scheduler, 30 epochs
- Best checkpoint saved to `best_tinyvad.pt` based on validation F1

### Experiment log: `distill_record.ipynb`

Records the full distillation run end-to-end:

1. Defines TinyVAD inline and prints the parameter count / compression ratio
2. Runs `distill.py` and captures the epoch-by-epoch training log (loss, F1, Prec, Rec)
3. Shows that the student reaches **F1 = 0.9223** on LibriParty val vs. the teacher's 0.9463

---

## Part 3 — TORGO Fine-tuning

Fine-tunes the distilled TinyVAD on the [TORGO](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) dysarthric speech dataset to improve VAD on atypical speech.

**Evaluation protocol — LOSO (Leave-One-Speaker-Out):** for each of the 8 annotated speakers (F01, F03, F04, M01, M02, M04, M05, MC01), the model is trained on the remaining 7 speakers and evaluated on the held-out speaker. This gives a speaker-independent measure of adaptation.

**Labelling:** `.phn`/`.PHN` phoneme annotation files are used. Phonemes `sil`, `noi`, `sp`, `h#`, `epi`, `pau` are mapped to silence; all others to speech. Files without a matching annotation are excluded.

**Shared fine-tuning strategy:** CNN layers are frozen (spectral features are domain-agnostic); only GRU + DNN are updated to adapt to dysarthric temporal patterns.

Two scripts implement different anti-forgetting strategies:

### `finetune_torgo_loso.py` — LOSO baseline

Standard LOSO fine-tuning with BCE loss on TORGO only. Tested at two learning rates:

- **lr=1e-4** (default): stronger adaptation, but causes significant catastrophic forgetting on LibriParty (macro drop ≈ 0.064, 7/8 speakers exceed the 0.05 threshold)
- **lr=5e-5**: more conservative; reduces forgetting for some speakers (notably M02) but at the cost of weaker adaptation overall

### `finetune_mixed.py` — Mixed-data replay

Mixes TORGO training with LibriParty **experience replay** to prevent catastrophic forgetting:

```
loss = TORGO_loss + 0.3 × LibriParty_replay_loss
```

A LibriParty batch is drawn every step (via an infinite cycle iterator) alongside each TORGO batch. The model is thus reminded of typical speech patterns throughout fine-tuning. Result: macro forgetting drops to **0.011** (vs. 0.064 for the baseline) while maintaining similar TORGO gain.

### `eval_tinyvad.py` — Evaluation

Evaluates any combination of checkpoints on both datasets:

- Pre-trained TinyVAD baseline (before fine-tuning)
- CRDNN teacher ceiling
- Each LOSO fold checkpoint on its held-out TORGO speaker
- All checkpoints on the LibriParty val set (to measure retention / catastrophic forgetting)

Prints a full summary table with per-speaker TORGO gain and LibriParty drop, flagging speakers where forgetting exceeds 0.05.

### Experiment log: `torgo_finetune_record.ipynb`

Records the complete TORGO fine-tuning pipeline:

1. Downloads and extracts the TORGO dataset
2. Runs the LOSO baseline (`finetune_torgo_loso.py`) and evaluates with `eval_tinyvad.py`
3. Lowers the learning rate to 5e-5 and re-runs, comparing the forgetting trade-off
4. Runs mixed-data fine-tuning (`finetune_mixed.py`) and evaluates the final results
5. Summarises the three-way comparison: mixed-data replay achieves the best balance (highest TORGO gain, lowest forgetting)

### Analysis: `analysis.ipynb`

Standalone data analysis and visualisation notebook. Reproduces all experimental results from the two record notebooks and generates publication-ready figures:

- Distillation training curves (loss, F1, Prec, Rec over 30 epochs)
- Per-speaker TORGO F1 before and after fine-tuning for all three strategies
- Catastrophic forgetting bar charts and heatmaps
- Gain vs. forgetting scatter plot (adaptation–retention trade-off)
- Macro-level comparison and radar chart summary

---

## Results Summary

| Model | LibriParty F1 | TORGO F1 (macro) |
|-------|--------------|-----------------|
| CRDNN teacher | 0.9463 | 0.6320 |
| TinyVAD (distilled) | 0.9223 | 0.7368 |
| + LOSO lr=1e-4 | 0.8580 | 0.8045 |
| + LOSO lr=5e-5 | 0.8791 | 0.7935 |
| **+ Mixed-data replay** | **0.9117** | **0.8061** |

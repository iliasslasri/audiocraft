# MAGNET Audio Inpainting

Text-guided audio inpainting using MAGNeT's non-autoregressive masked generation. Masks a temporal region of audio in EnCodec token space and re-generates it conditioned on a new text description.

## Setup

### CLAP Checkpoint

```bash
huggingface-cli download lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt --local-dir .
```

### Dependencies

```bash
pip install laion_clap scipy
```

## Usage

### Basic Inpainting

```bash
# Middle gap (10s–20s)
python inpaint.py --start 10.0 --end 20.0 --output_dir my_results

# Audio continuation (15s–30s)
python inpaint.py --start 15.0 --end 30.0

# Full regeneration baseline (0s–30s)
python inpaint.py --start 0.0 --end 30.0
```

### Advanced Features

```bash
# Boundary-aware soft masking (transition zone in tokens, 50 ≈ 1s at 50Hz)
python inpaint.py --start 10.0 --end 20.0 --soft_mask_transition 50

# Position-aware CFG (stronger guidance at gap center, weaker at boundaries)
python inpaint.py --start 10.0 --end 20.0 --position_aware_cfg

# Combined
python inpaint.py --start 10.0 --end 20.0 --soft_mask_transition 50 --position_aware_cfg
```

### Running All Experiments

```bash
sbatch run_experiments.sh
```

Runs 4 experiment groups with automatic logging and plotting:

| Group | Experiments | Description |
|-------|------------|-------------|
| 1 — Baselines | full regen, middle gap, continuation | Hard mask, uniform CFG |
| 2 — Soft Mask | τ = 25, 50, 100 tokens | Boundary-aware transition zones |
| 3 — Pos-CFG | middle gap, continuation | Position-aware CFG scheduling |
| 4 — Combined | middle gap, continuation | Soft mask + position-aware CFG |

Results go to `results/<experiment_name>/`, logs to `experiment_logs/`.

### Plotting

```bash
python plot_metrics.py --path results/baseline_middle_gap/inpainting_metrics.json --output plots/
```

Generates 4 plot files:
- `clap_consistency.png` — gap + full audio CLAP vs target
- `boundary_smoothness.png` — spectral flux ratio at mask edges
- `semantic_shift.png` — shift-to-target + drift-from-original
- `metrics_dashboard.png` — combined overview with FAD score

Supports both old (flat list) and new (nested) JSON formats.

## Evaluation Metrics

All per-sample metrics are saved to `inpainting_metrics.json` with structure:
```json
{ "config": {...}, "fad": 12.34, "per_sample_metrics": [...] }
```

### CLAP Text-Audio Consistency

Cosine similarity between CLAP embeddings of audio and text. Computed for full audio and gap region, against both target and original descriptions.

### Boundary Smoothness Score

Spectral flux (L2 norm of STFT magnitude diffs) at mask boundaries.

| Metric | Meaning |
|--------|---------|
| `boundary_*_peak_flux` | Peak discontinuity near left/right boundary |
| `boundary_*_flux_ratio` | Ratio vs original (1.0 = equally smooth) |
| `boundary_avg_flux_ratio` | Average of both boundaries |

### Semantic Shift Ratio

How much the gap content shifted toward the target description.

| Metric | Meaning |
|--------|---------|
| `semantic_shift_to_target` | `CLAP(inp_gap, target) / CLAP(orig_gap, target)` — >1.0 = improved |
| `semantic_drift_from_original` | `CLAP(inp_gap, original) / CLAP(orig_gap, original)` — <1.0 = changed |

### Fréchet Audio Distance (FAD)

CLAP-based FAD computed as an aggregate over all generated vs reference samples at the end of each experiment run. Lower = better. Needs ≥30 samples for stability.

## Directory Structure

```
inpainting/
├── inpaint.py              # Main inpainting pipeline
├── inpainting_metrics.py   # All evaluation metrics
├── plot_metrics.py         # Visualization (4 plot types)
├── run_experiments.sh      # Full experiment suite (sbatch)
├── generate_samples.py     # Generate base audio samples
├── research_proposal.md    # Research directions
├── data/
│   ├── magnet_descriptions.csv
│   └── magnet_descriptions_slight_change.csv
└── README.md
```
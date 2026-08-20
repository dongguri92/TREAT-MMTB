# TREAT-MMTB 2026 — Task 1: TB Cavity Detection & Segmentation

Our entry for Task 1 of the MICCAI TREAT-MMTB 2026 challenge: patient-level
detection and pixel-level segmentation of tuberculous cavities on chest
radiographs.

Metric: `final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

**Final rank: 7th.** Full experiment log, failure analysis, and negative
results are in [RESULTS.md](RESULTS.md). ([한국어 문서](README.ko.md))

---

## Two stages of development

The project went through two architectural transitions, each prompted by a
failure the previous system could not resolve.

| Stage | System | Internal | External | Code |
|---|---|---|---|---|
| 0 | Scratch-trained multi-task U-Net | 0.6692 | — | — |
| 1 | EVA-X + classification-driven decision | 0.7356 | 0.5463 | [`evax_baseline/`](evax_baseline/) |
| 2 | **X-Raydar two-stage (final)** | **0.7879** | **0.5838** | [`xraydar_two_stage/`](xraydar_two_stage/) |

**Stage 0 → 1.** A carefully tuned scratch-trained model plateaued because
several small, isolated cavities produced almost no foreground response across
architectures and loss functions — a representation limitation rather than a
thresholding problem. We transferred to EVA-X, a self-supervised chest
radiograph foundation model.

**Stage 1 → 2.** Within the EVA-X system, the classification and segmentation
heads had different error modes, and mask-based evidence proved less reliable
than a dedicated classifier — especially under external domain shift. We
therefore separated the two tasks entirely, transferring X-Raydar (trained on
over 1.6M chest X-rays) and fine-tuning independent networks for detection and
localization.

---

## Stage 1 — EVA-X with a classification-driven decision

Code: [`evax_baseline/`](evax_baseline/)

```
Input CXR (1 channel)
  → percentile(1,99) clip → [0,1]
  → crop lower 15% → resize+pad to 1024 → CLAHE(2.0) → z-score
  → EVA-X small ViT (patch16, embed 384, depth 12, SwiGLU, RoPE)
      ├─ features from blocks 2,5,8,11 → SimpleFeaturePyramid (ViTDet)
      │    → FPN decoder → segmentation logits
      └─ block 11 feature → GAP → dropout → linear → classification logit
  → classification-driven decision → restore to original DICOM grid
```

29.9M parameters. No mmcv or mmsegmentation — only timm (verified with 1.0.22).

### Decision rule

```
present = cls_prob >= 0.5      # averaged over original and horizontally flipped views

present and (P >= 0.5) non-empty  → mask = (P >= 0.5)
present and (P >= 0.5) empty      → mask = (P >= 0.5 * p_max)
not present                       → mask = empty

cavity = 1  iff  restored mask is non-empty   (guarantees CSV/NIfTI consistency)
```

The second branch prevents a rule violation: a positive classification with an
empty mask would produce CSV=1 alongside an empty NIfTI. We use a threshold
**relative to `p_max`** rather than an absolute one, because an absolute value
tuned on the internal probability distribution does not transfer.

### External progression

| Submission | Detection | Dice | Final |
|---|---|---|---|
| EVA-X λ=0.5 + segmentation veto | 0.6713 | 0.1568 | 0.5170 |
| λ=0.1 + classification-driven (veto removed) | 0.6862 | 0.1644 | 0.5296 |
| + percentile normalization | 0.7063 | 0.1657 | 0.5441 |
| **+ horizontal-flip TTA (classification head only)** | **0.7089** | **0.1667** | **0.5463** |

> All three adopted changes **lowered** the internal score yet improved the
> external one. See [RESULTS.md](RESULTS.md) for the full comparison.

---

## Stage 2 — X-Raydar two-stage system (final submission)

Code: [`xraydar_two_stage/`](xraydar_two_stage/) *(to be added via pull request)*

The central design choice is **task ownership**. A ground-truth mask determines
the ground-truth class, but pixel-wise segmentation risk differs from
image-level risk: one spurious region creates a false positive, while missing a
tiny cavity creates a complete classification false negative. Forcing both
through one path made each worse.

**Classifier owns the presence decision.** X-Raydar XNet38 (Inception-v3) at
1024 px, fine-tuned with binary cross-entropy (auxiliary head weight 0.4),
AdamW at 1e-4, weight decay 0.05, EMA decay 0.99. Sigmoid probabilities from
three EMA checkpoints (epochs 215, 282, 299) are averaged at a fixed threshold
of 0.5 — no calibration layer, logit bias, or modality-specific threshold.

**Segmentation only localizes positives.** Two independent X-Raydar encoders at
1024 px: a five-level U-Net (transposed-convolution decoder with
768/288/192/64-channel skips) and an FPN pixel decoder with a 15-query
Mask2Former. Both are trained on cavity-positive samples only. Their outputs
are merged by a **centered-logit ensemble**, expressing each decoder relative
to its own operating boundary before averaging:

```
z = ½[logit(p_unet) − logit(0.06)] + ½[logit(p_m2f) − logit(0.54)]
mask = I[z ≥ 0]
```

If the classifier is negative, segmentation is skipped and an all-zero mask is
written. An empty positive mask retains pixels above half its maximum, with a
peak-pixel safeguard as a final fallback.

**Preprocessing.** DICOM pixels divided by the maximum code implied by
`BitsStored`; pixel-padding values excluded from histograms; presentation
polarity applied once. Two XGBoost regressors predict image-specific lower and
upper percentile ranks, whose intensity interval is mapped to [0,1]. Public
Shenzhen radiographs were added through a quality-controlled cache; Montgomery
was excluded after mask inspection.

### Internal progression

| Setting | Accuracy | Dice | Score |
|---|---|---|---|
| EVA-X, no mask veto | 0.8919 | 0.2995 | 0.7142 |
| EVA-X, segmentation-confidence veto | 0.9189 | 0.3078 | 0.7356 |
| 512 px X-Raydar + UPerNet | 0.9189 | 0.4148 | 0.7677 |
| 1024 px X-Raydar + U-Net, global boundary | 0.9550 | 0.4300 | 0.7975 |
| 1024 px X-Raydar + U-Net, CR/XC boundaries | 0.9550 | 0.4337 | 0.7986 |
| **Final temporal classifier + mask ensemble** | 0.9369 | **0.4400** | 0.7879 |

Final classifier AUROC 0.9815 (52 TP, 52 TN, 1 FP, 6 FN at threshold 0.5).
Ungated positive-case ensemble Dice was 0.4944; gated mean Dice 0.4400.

**External: 0.5838** (detection 0.7631, Dice 0.1653).

---

## What carried across both stages

**Segmentation confidence should not overturn a stronger image-level
decision.** In the internal phase, letting a confidently negative segmentation
map veto a positive classification raised the EVA-X score from 0.7142 to
0.7356. On the external set, where Dice collapsed to 0.05–0.20 for every team,
the same rule overturned correct decisions — removing it raised our external
score from 0.5170 to 0.5296. Stage 2 encodes this principle structurally: the
classifier alone decides presence, and segmentation never influences it.

**Internal score differences did not predict external behaviour.** In three of
four changes we measured, the internal and external effects had opposite signs.
Mechanism-level reasoning — *which component fails first, and does the whole
system fail with it?* — proved a better guide than the internal leaderboard.

---

## Setup

### Environment
```bash
conda create -n miccai python=3.11 -y
conda activate miccai
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==1.0.22 numpy==1.26.4 opencv-python-headless pydicom==3.0.2 \
            SimpleITK scipy scikit-learn albumentations tqdm matplotlib \
            pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.1.0 pylibjpeg-openjpeg==2.2.1
```

### Pre-trained weights (EVA-X small)
```bash
mkdir -p ~/eva_x_backup && cd ~/eva_x_backup
wget https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_small_patch16_merged520k_mim.pt
```

### Data paths
Edit the four constants at the top of `evax_baseline/datasets.py`:
```python
TRAIN_DCM_DIR  = ".../data_original/train/CXR"
TRAIN_MASK_DIR = ".../data_original/train/CXR_label"
VAL_DCM_DIR    = ".../data_original/val/CXR"
VAL_MASK_DIR   = ".../data_original/val/CXR_label"
```
Challenge data is not included in this repository (distribution terms).

---

## Reproduction (Stage 1)

### Training
```bash
cd evax_baseline
python main.py --model evax_seg --tag evax_pct_l01 --channels 1 \
    --lambda_cls 0.1 --target_size 1024 --batch_size 8 \
    --optimizer adamw --initial_lr 5e-5 \
    --scheduler cosine --warmup 5 --max_epochs 150
```
Batch size 8 fits on a 24GB A5000 (flash attention); 150 epochs take ~5–6 hours.

### Inference
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --out_dir results_final \
    --detection cls --cls_threshold 0.5 --min_pixels 0
```
Evaluate with the official `evaluate_task1.py` from the
[challenge repository](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026).

### Diagnostic sweep
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --sweep
```
Prints threshold-wise accuracy for four detection strategies (cls / seg-max /
top-100 / soft-area), a false-negative breakdown, and the cases where the
classification and segmentation heads disagree.

### Docker
```bash
cd evax_baseline/task1_submit_pct_l01_tta
# place weights/best_evax_cls.pth first
docker build -f Dockerfile_task1 -t rami-task1:latest .
docker run --rm --network none \
    -v /path/to/input:/input:ro -v /path/to/output:/output \
    rami-task1:latest
```

---

## Files

### `evax_baseline/`

| File | Role |
|---|---|
| `main.py` | Training entry point |
| `models.py` | `modeltype()` factory (multitask_unet / evax_seg) |
| `models_evax.py` | `EVAXSegNet` — EVA-X backbone + FPN decoder + classification head |
| `eva_x.py` | `checkpoint_filter_fn` from the official EVA-X repository |
| `datasets.py` | Preprocessing pipeline and data loaders |
| `training.py` | `fit()` / `compute_lr()` / `make_optimizer()` |
| `utils.py` | DiceCE, Tversky, Boundary losses, dice metric, checkpointing |
| `inference_evax.py` | Inference, threshold sweeps, decision rules |
| `tta.py` | Test-time augmentation applied to the classification head only |
| `task1_submit_pct_l01_tta/` | Submission Docker (predict + Dockerfile + requirements) |

### `xraydar_two_stage/`

To be added via pull request.

---

## Negative results

Documented in [RESULTS.md](RESULTS.md) so the same ground is not covered twice.
In short, none of the following helped within the EVA-X system: **a larger
backbone (EVA-X base), 3-channel input, Tversky/Boundary losses, crop ratio,
mask threshold, DropPath, a deeper classification head, size-weighted sampling,
DICOM VOI windowing, and stronger augmentation.**

The only setting that broke the internal Dice ceiling of ~0.31 was an **ROI
two-stage oracle experiment** (0.31 on full images → 0.80 when cropped around
the ground-truth box), suggesting that the bottleneck was the input condition
rather than model capacity. This observation motivated the task separation in
Stage 2.

---

## References

```bibtex
@article{yao2025eva,
  title={EVA-X: A foundation model for general chest X-ray analysis with self-supervised learning},
  author={Yao, Jingfeng and Wang, Xinggang and Song, Yuehao and Zhao, Huangxuan and
          Ma, Jun and Chen, Yajie and Liu, Wenyu and Wang, Bo},
  journal={npj Digital Medicine}, volume={8}, number={1}, pages={678}, year={2025}
}

@article{dicentecid2024xraydar,
  title={Development and validation of open-source deep neural networks for
         comprehensive chest X-ray reading: a retrospective, multicentre study},
  author={Dicente Cid, Yashin and Macpherson, Matthew and Gervais-Andre, Louise and others},
  journal={The Lancet Digital Health}, volume={6}, pages={e44--e57}, year={2024}
}
```

- EVA-X: https://github.com/hustvl/EVA-X
- X-Raydar: https://github.com/MachineLearningWarwickMed/x-raydar
- Challenge: https://github.com/mi2rl-challenge/treat-mmtb.miccai2026

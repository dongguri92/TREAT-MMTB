# TREAT-MMTB 2026 — Task 1: TB Cavity Detection & Segmentation

Our entry for Task 1 of the MICCAI TREAT-MMTB 2026 challenge: patient-level
detection and pixel-level segmentation of tuberculous cavities on chest
radiographs.

Metric: `final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

**Final rank: 7th.** Full experiment log, failure analysis, and negative
results are in [RESULTS.md](RESULTS.md). ([한국어 문서](README.ko.md))

---

## Results

### External set (final phase)

| Submission | Detection | Dice | Final |
|---|---|---|---|
| EVA-X λ=0.5 + segmentation veto (internal-phase system) | 0.6713 | 0.1568 | 0.5170 |
| λ=0.1 + classification-driven (veto removed) | 0.6862 | 0.1644 | 0.5296 |
| + percentile normalization | 0.7063 | 0.1657 | 0.5441 |
| **+ horizontal-flip TTA (best in this repository)** | **0.7089** | **0.1667** | **0.5463** |
| 5-fold ensemble | 0.6906 | 0.1979 | 0.5428 |

The team's final submission (0.5838) was a multi-backbone fusion model
developed by a collaborator and is outside the scope of this repository.

### Internal validation (111 cases / 58 pos, 53 neg)

| System | Detection | Dice | Final |
|---|---|---|---|
| nnU-Net (plain, 5-fold) | 0.7658 | 0.2725 | 0.6178 |
| Scratch-trained multi-task U-Net + scale aug | 0.8378 | 0.2756 | 0.6692 |
| EVA-X λ=0.5 + segmentation veto | **0.9189** | **0.3078** | **0.7356** |
| EVA-X λ=0.1, percentile + TTA (submitted externally) | 0.8829 | 0.2948 | 0.7065 |

> **Note**: internal score differences did not predict external behaviour.
> All three changes we adopted (veto removal, percentile normalization, TTA)
> *lowered* the internal score yet improved the external one. See RESULTS.md.

---

## Architecture

```
Input CXR (1 channel)
  → percentile(1,99) clip → [0,1]
  → crop lower 15% → resize+pad to 1024 → CLAHE(2.0) → z-score
  → EVA-X small ViT (patch16, embed 384, depth 12, SwiGLU, RoPE)
      ├─ features from blocks 2,5,8,11 → SimpleFeaturePyramid (ViTDet)
      │    → FPN decoder → segmentation logits
      └─ block 11 feature → GAP → dropout → linear → classification logit
  → classification-driven decision (below) → restore to original DICOM grid
```

29.9M parameters (22M backbone + 7.9M decoder/heads).
No mmcv or mmsegmentation required — only timm (>=0.9, verified with 1.0.22).

---

## Files

| File | Role |
|---|---|
| `main.py` | Training entry point. Single split / k-fold / full-data training |
| `models.py` | `modeltype()` factory (multitask_unet / evax_seg) |
| `models_evax.py` | `EVAXSegNet` — EVA-X backbone + FPN decoder + classification head |
| `eva_x.py` | `checkpoint_filter_fn` from the official EVA-X repository |
| `datasets.py` | Preprocessing pipeline, k-fold splitting, full-data loader |
| `training.py` | `fit()` / `compute_lr()` / `make_optimizer()` |
| `utils.py` | DiceCE, Tversky, Boundary losses, dice metric, checkpointing |
| `inference_evax.py` | Inference, threshold sweeps, decision rules |
| `tta.py` | Test-time augmentation applied to the classification head only |
| `task1_submit_pct_l01_tta/` | **Submission Docker** (predict + Dockerfile + requirements) |

The evaluation script (`evaluate_task1.py`) comes from the
[official challenge repository](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026).

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
Edit the four constants at the top of `datasets.py`:
```python
TRAIN_DCM_DIR  = ".../data_original/train/CXR"
TRAIN_MASK_DIR = ".../data_original/train/CXR_label"
VAL_DCM_DIR    = ".../data_original/val/CXR"
VAL_MASK_DIR   = ".../data_original/val/CXR_label"
```
Challenge data is not included in this repository (distribution terms).

---

## Reproduction

### Training (externally submitted model)
```bash
python main.py --model evax_seg --tag evax_pct_l01 --channels 1 \
    --lambda_cls 0.1 --target_size 1024 --batch_size 8 \
    --optimizer adamw --initial_lr 5e-5 \
    --scheduler cosine --warmup 5 --max_epochs 150
```
Batch size 8 fits on a 24GB A5000 (flash attention); 150 epochs take ~5–6 hours.

### k-fold / full-data training
```bash
# Split the combined 555 train+val images into 5 folds, hold out fold 0
python main.py --model evax_seg --fold 0 --n_folds 5 --tag evax_fold0 ...

# Train on all 555 images (no held-out set — fix the epoch count in advance)
python main.py --model evax_seg --use_all --tag evax_all555 --max_epochs 52 ...
```

### Inference
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --out_dir results_final \
    --detection cls --cls_threshold 0.5 --min_pixels 0
```
Evaluate with the official `evaluate_task1.py`.

### Diagnostic sweep
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --sweep
```
Prints threshold-wise accuracy for four detection strategies (cls / seg-max /
top-100 / soft-area), a false-negative breakdown, and the list of cases where
the classification and segmentation heads disagree.

### Docker
```bash
cd task1_submit_pct_l01_tta
# place weights/best_evax_cls.pth first
docker build -f Dockerfile_task1 -t rami-task1:latest .
docker run --rm --network none \
    -v /path/to/input:/input:ro -v /path/to/output:/output \
    rami-task1:latest
```

---

## Key design decisions

### 1. Checkpoint selection follows the challenge metric
We save the epoch maximizing **`0.7 × cls_acc + 0.3 × dice`**, not `val_dice`.
Detection carries more than twice the weight of Dice, so selecting on Dice
alone loses systematically. (When `--lambda_cls 0`, classification accuracy is
meaningless and only Dice is used.)

### 2. Percentile normalization
We rescale from the 1st and 99th percentiles rather than the observed minimum
and maximum. A handful of extreme pixels — burned-in markers, detector
artifacts, black borders — would otherwise determine the scale for the entire
image, and those pixels differ across devices and institutions. Unlike the
DICOM window tags (present in only 53 of 111 internal cases), this applies one
rule to every image. It was our single largest external gain (+0.0145).

### 3. Classification-driven decision
```
present = cls_prob >= 0.5      # averaged over original and flipped views

present & (P >= 0.5) non-empty  → mask = (P >= 0.5)
present & (P >= 0.5) empty      → mask = (P >= 0.5 * p_max)
not present                     → mask = empty

cavity = 1 ⟺ restored mask is non-empty   (guarantees CSV/NIfTI consistency)
```
The second branch prevents a rule violation: if the classifier says positive
but segmentation never crosses its threshold, the CSV would say 1 while the
mask is empty. We use a threshold **relative to `p_max`** rather than an
absolute one, because an absolute value tuned on internal data does not
transfer to a different probability distribution.

### 4. Why the segmentation veto was removed
In the internal phase, letting a confidently negative segmentation map
(`p_max < 0.005`) override a positive classification was beneficial
(0.7142 → 0.7356). On the external set, however, Dice collapsed to 0.05–0.20
for every team, so a low `p_max` no longer indicated absence — only that the
segmentation head had not responded. Removing the veto raised our external
score from 0.5170 to 0.5296.

**The appropriate way to combine two heads is not a fixed property of the
architecture but a function of their relative reliability in the target
domain.** This is the main methodological observation of our entry.

### 5. TTA on the classification head only
Averaging masks across views blurs boundaries, and the effect on Dice is hard
to predict. Our Dice was already among the highest in the competition, so we
applied TTA only where there was room to improve — the classification output.

### 6. Aggressive scale augmentation
The single largest gain during the scratch-trained phase.
`A.Affine(scale=(0.5,1.4), rotate=(-30,30), p=0.5)` widens both the range and
the probability relative to the nnU-Net default `scale=(0.7,1.4), p=0.2`.

---

## Negative results

Documented in [RESULTS.md](RESULTS.md) so the same ground is not covered twice.
In short, none of the following helped: **a larger backbone (EVA-X base),
3-channel input, Tversky/Boundary losses, crop ratio, mask threshold, DropPath,
a deeper classification head, size-weighted sampling, DICOM VOI windowing,
stronger augmentation, and 5-fold ensembling.**

The only setting that broke the internal Dice ceiling of ~0.31 was an **ROI
two-stage oracle experiment** (0.31 on full images → 0.80 when cropped around
the ground-truth box), suggesting that the bottleneck is the input condition
rather than model capacity. The practical pipeline was not implemented within
the challenge timeline.

---

## Reference

EVA-X pre-trained model:
```bibtex
@article{yao2025eva,
  title={EVA-X: A foundation model for general chest X-ray analysis with self-supervised learning},
  author={Yao, Jingfeng and Wang, Xinggang and Song, Yuehao and Zhao, Huangxuan and
          Ma, Jun and Chen, Yajie and Liu, Wenyu and Wang, Bo},
  journal={npj Digital Medicine}, volume={8}, number={1}, pages={678}, year={2025}
}
```
- EVA-X: https://github.com/hustvl/EVA-X
- Challenge: https://github.com/mi2rl-challenge/treat-mmtb.miccai2026

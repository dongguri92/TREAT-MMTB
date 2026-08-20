# Results

MICCAI TREAT-MMTB 2026 — Task 1 (Cavity Detection & Segmentation)

`final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

**Final rank: 7th** (external full set, final score 0.5838)

This document records the EVA-X stage in detail — the experiments, the failure
analysis, and what did not work. The final submission was the X-Raydar
two-stage system; see the [README](README.md) for its design and results.

---

## External phase

The organizers' external set draws from five international institutions
(Korea, Mongolia, Peru, the Philippines, and others).

| # | Submission | Detection | Dice | Final |
|---|---|---|---|---|
| 1 | EVA-X λ=0.5 + segmentation veto (internal-phase system) | 0.6713 | 0.1568 | 0.5170 |
| 2 | λ=0.1 + classification-driven (veto removed) | 0.6862 | 0.1644 | 0.5296 |
| 3 | **+ percentile normalization** | 0.7063 | 0.1657 | **0.5441** |
| 4 | **+ horizontal-flip TTA (classification head only)** | 0.7089 | 0.1667 | **0.5463** |
| 5 | λ=0.7 (reverted) | 0.6774 | 0.1674 | 0.5244 |

Rows 3–4 are the best results from the EVA-X system. The final submission,
an X-Raydar two-stage system with separated detection and segmentation, scored
**0.5838** (detection 0.7631, Dice 0.1653).

### Observations

**Segmentation is far more fragile under domain shift.** From internal to
external, detection accuracy fell from 0.9189 to 0.6713 (73% retained), while
Dice fell from 0.3078 to 0.1568 (51% retained). Every team on the leaderboard
landed in the 0.05–0.20 Dice range, so this reflects the task rather than any
one system.

**Consequently, a decision rule that depends on segmentation backfired
externally.** In the internal phase, allowing a confidently negative
segmentation map to override a positive classification was beneficial
(internal final 0.7142 → 0.7356). Externally, a low `p_max` no longer indicates
the absence of a cavity — only that the segmentation head did not respond — so
the veto overturns correct decisions. Removing it raised the external final
score from 0.5170 to 0.5296.

**Internal score differences did not predict external behaviour.** In three of
four changes, the internal and external effects had **opposite signs**.

| Change | Internal Δ | External Δ |
|---|---|---|
| Veto removal | **−0.0161** | **+0.0126** |
| Percentile normalization | **−0.0053** | **+0.0145** |
| Classification TTA | **−0.0077** | **+0.0022** |
| λ 0.1 → 0.7 | +0.0074 | **−0.0197** |

Selecting external submissions by internal leaderboard position was therefore
unreliable, and mechanism-level reasoning — *which component fails first, and
does the whole system fail with it?* — proved a better guide. All three changes
we adopted (veto removal, percentile normalization, TTA) lowered the internal
score yet improved the external one.

---

## Internal validation (111 cases / 58 positive, 53 negative)

| System | Detection | Dice | Final |
|---|---|---|---|
| nnU-Net (plain, 5-fold) | 0.7658 | 0.2725 | 0.6178 |
| Scratch-trained multi-task U-Net + scale aug | 0.8378 | 0.2756 | 0.6692 |
| EVA-X, segmentation-based detection (λ=0) | 0.8468 | 0.2999 | 0.6805 |
| EVA-X λ=0.5 + segmentation veto | **0.9189** | **0.3078** | **0.7356** |
| EVA-X λ=0.1, min–max, classification-driven | 0.9099 | 0.2751 | 0.7195 |
| EVA-X λ=0.1, percentile | 0.8919 | 0.3000 | 0.7143 |
| + classification TTA (submitted externally) | 0.8829 | 0.2948 | 0.7065 |

### Interaction between λ and the decision rule

The classification-loss weight λ controls how aggressive the classification
head is, and **which decision rule is preferable changes with it.**

| λ | Cls accuracy (alone) | With veto | Classification-driven |
|---|---|---|---|
| 0.1 | **0.9099** | 0.6981 | **0.7195** |
| 0.3 | 0.8919 | 0.6996 | 0.7149 |
| 0.5 | 0.8919 | **0.7356** | 0.7145 |

With a small λ the two heads behave similarly, disagreements are rare (four
cases at λ=0.1, all of which the classifier got right), and the veto has
nothing to correct. With a larger λ the classifier becomes aggressive and
produces false positives on normal lungs — exactly the situation in which a
confidently negative segmentation map is complementary (at λ=0.5 the veto
corrected 4 of 9 disagreements for a net gain of 2 cases).

---

## Final decision rule (this repository)

```
present = cls_prob >= 0.5      # averaged over original and horizontally flipped views

present and (P >= 0.5) non-empty  → mask = (P >= 0.5)
present and (P >= 0.5) empty      → mask = (P >= 0.5 * p_max)
not present                       → mask = empty

cavity = 1  iff  restored mask is non-empty   (CSV/NIfTI consistency guaranteed)
```

`P` is the foreground probability map and `p_max = max P`.

The second branch is necessary because a positive classification with an empty
segmentation mask would produce CSV=1 with an empty NIfTI, violating the
challenge rules. Three internal cases (77, 158, 203) hit this branch, and all
three were ground-truth positives. We use a threshold **relative to `p_max`**
rather than an absolute one: an absolute value tuned on the internal
probability distribution does not transfer, whereas a relative one is
scale-invariant and always yields a non-empty mask.

TTA is applied **only to the classification head**. Averaging masks across
views blurs boundaries and the effect on Dice is unpredictable, and our Dice
was already among the highest in the competition.

---

## Failure analysis by lesion size (internal, λ=0.5 + veto)

Restricted to the 50 cases with a non-empty prediction.

| Cavity size | n | Dice | Precision | Recall | Area ratio (pred/GT) |
|---|---|---|---|---|---|
| Small | 16 | 0.266 | 0.300 | 0.292 | 1.51 |
| Medium | 22 | 0.321 | 0.396 | 0.366 | 2.14 |
| Large | 12 | **0.569** | **0.868** | 0.482 | 0.58 |

**Large cavities are localized correctly but only half-covered** (precision
0.868, recall 0.482). **Small and medium cavities are over-drawn by 1.5–2× yet
still score poorly** — the problem is position and shape, not size. All 8
completely missed cases and all 12 cases with exactly zero Dice are small or
medium; none are large.

A board-certified radiologist reviewed the 12 zero-Dice cases and found that
the model had segmented **other lucent lesions within the lung fields** —
air-containing regions surrounded by other structures — rather than failing on
resolution or preprocessing.

---

## ROI two-stage experiment (oracle)

To test whether the model can delineate a boundary *given* the correct
location, we trained and evaluated only on crops around the ground-truth
bounding box — an oracle condition unavailable at inference.

| ROI context | Oracle Dice |
|---|---|
| 1.5× | 0.799 |
| 3.0× | 0.802 |
| 4.0× / 5.0× | ~0.80 |

**Dice rises from 0.31 on full images to 0.80 under the ROI condition** with
the same model and the same decoder; only the input changed. Enlarging the
context from 1.5× to 5× left Dice unchanged, indicating that the magnification
benefit saturates and that the gain comes from **being told where to look**
rather than from resolution.

This is an upper bound. In practice the first stage supplies imperfect boxes,
and for the 12 mislocalized cases an ROI crop would only delineate the wrong
structure more precisely. The practical pipeline (three-level coordinate
inversion, per-component merging) was not implemented within the challenge
timeline, and the corresponding code is not included in this repository.

---

## Negative results

Recorded so the same ground is not covered twice. **None of the following moved
internal Dice out of the 0.26–0.31 range.**

| Attempt | Result | Note |
|---|---|---|
| **EVA-X base** backbone (86M) | λ0.5 ~0.691 / λ0.3 ~0.711 | Worse than small; overfits when fine-tuned on 444 images |
| **3-channel input** (raw / CLAHE 2.0 / CLAHE 1.0) | Below 1-channel | Pre-training assumes grayscale replicated across 3 channels |
| **Focal Tversky** (α0.3 β0.7 γ1.33) | Dice 0.2963 | Effectively identical to DiceCE |
| **Dice + Boundary** (w=0.5) | Dice 0.2975 | Identical |
| **Lower crop 20%** (default 15%) | ~0.714 | No difference |
| **Mask threshold sweep** | 0.2756 → 0.2760 | Probability maps are strongly bimodal |
| **DropPath 0.1** | ~0.26 | No change |
| **Deeper classification head** | Decreased | |
| **Size-weighted sampling** (small 2–5×) | Detection 0.847–0.892 | Detection dropped; likely from reduced exposure to negatives |
| **DICOM VOI windowing** | 0.8829 / 0.2899 / 0.7050 | Only 53 of 111 cases carry window tags, mixing two normalizations |
| **Stronger augmentation** (random CLAHE clip, higher degradation rates) | 0.8829 / 0.2984 / 0.7075 | Detection down by one case |
| **Classification threshold sweep** (0.2–0.7) | Identical across 0.4–0.7 | Classification probabilities are extremely bimodal |

### Percentile normalization (adopted)

Rescaling from the 1st and 99th percentiles rather than the observed minimum
and maximum. This removes the influence of a handful of extreme pixels
(burned-in markers, detector artifacts, black borders) that would otherwise fix
the scale for the whole image. Unlike DICOM window tags, it applies uniformly
to every image.

| Model | Detection | Dice | Final |
|---|---|---|---|
| min–max λ=0.1 | 0.9099 | 0.2751 | 0.7195 |
| percentile λ=0.05 | 0.9009 | 0.3020 | **0.7212** |
| percentile λ=0.1 | 0.8919 | 0.3000 | 0.7143 |
| percentile λ=0.5 | 0.8829 | 0.2967 | 0.7070 |
| percentile λ=0.7 | 0.9009 | **0.3033** | 0.7216 |
| percentile λ=0.9 | 0.8919 | 0.2659 | 0.7041 |

The internal gain was small, but externally this was our single largest
improvement (+0.0145). The internal set of 111 cases is comparatively
homogeneous, whereas the external set spans five institutions with much wider
variation in extreme pixel values.

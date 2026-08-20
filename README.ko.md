# TREAT-MMTB 2026 — Task 1: TB Cavity Detection & Segmentation

MICCAI TREAT-MMTB 2026 챌린지 Task 1 (흉부 X선에서 결핵성 공동(cavity)의
환자 단위 검출 + 픽셀 단위 분할) 참가 코드.

평가 지표: `final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

**최종 순위 7위.** 상세한 실험 기록·실패 분석·negative results는
[RESULTS.ko.md](RESULTS.ko.md) 참조. ([English](README.md))

---

## 성능

### External set (본선)

| 제출 구성 | detection | Dice | final |
|---|---|---|---|
| EVA-X λ=0.5 + segmentation veto (예선 제출본) | 0.6713 | 0.1568 | 0.5170 |
| λ=0.1 + classification-driven (veto 제거) | 0.6862 | 0.1644 | 0.5296 |
| + percentile 정규화 | 0.7063 | 0.1657 | 0.5441 |
| **+ horizontal-flip TTA (본 저장소 최고)** | **0.7089** | **0.1667** | **0.5463** |

### Internal validation (111 cases / 58 pos, 53 neg)

| 시스템 | detection | Dice | final |
|---|---|---|---|
| nnU-Net (plain, 5-fold) | 0.7658 | 0.2725 | 0.6178 |
| from-scratch multi-task U-Net + scale aug | 0.8378 | 0.2756 | 0.6692 |
| EVA-X λ=0.5 + segmentation veto | **0.9189** | **0.3078** | **0.7356** |
| EVA-X λ=0.1 percentile + TTA (external 제출본) | 0.8829 | 0.2948 | 0.7065 |

> **주의**: internal 점수차가 external을 예측하지 못한다. 채택한 세 변경
> (veto 제거, percentile, TTA)은 모두 internal 점수가 **낮아졌음에도**
> external에서 개선되었다. 자세한 내용은 RESULTS.ko.md 참조.

---

## 구조

```
입력 CXR (1채널)
  → percentile(1,99) clip → [0,1]
  → 하부 15% crop → resize+pad 1024 → CLAHE(2.0) → z-score
  → EVA-X small ViT (patch16, embed 384, depth 12, SwiGLU, RoPE)
      ├─ blocks 2,5,8,11 feature → SimpleFeaturePyramid(ViTDet) → FPNDecoder → seg logits
      └─ block 11 feature → GAP → dropout → linear → cls logit
  → classification-driven decision (아래) → 원본 그리드로 역변환
```

총 29.9M 파라미터 (백본 22M + 디코더/헤드 7.9M).
mmcv / mmsegmentation 불필요, timm(>=0.9, 검증 1.0.22)만 있으면 동작.

---

## 파일

| 파일 | 역할 |
|---|---|
| `main.py` | 학습 진입점 |
| `models.py` | `modeltype()` 팩토리 (multitask_unet / evax_seg) |
| `models_evax.py` | `EVAXSegNet` — EVA-X 백본 + FPN 디코더 + cls head |
| `eva_x.py` | EVA-X 공식 저장소의 `checkpoint_filter_fn` (사전학습 로드용) |
| `datasets.py` | 전처리 파이프라인 및 데이터 로더 |
| `training.py` | `fit()` / `compute_lr()` / `make_optimizer()` |
| `utils.py` | DiceCE, Tversky, Boundary loss, dice metric, checkpoint 저장 |
| `inference_evax.py` | 추론 + threshold sweep + 결정 규칙 |
| `tta.py` | 분류 헤드 전용 test-time augmentation |
| `task1_submit_pct_l01_tta/` | **제출용 Docker** (predict + Dockerfile + requirements) |

평가 스크립트(`evaluate_task1.py`)는
[챌린지 공식 저장소](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026)의
것을 사용한다.

---

## 설정

### 환경
```bash
conda create -n miccai python=3.11 -y
conda activate miccai
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==1.0.22 numpy==1.26.4 opencv-python-headless pydicom==3.0.2 \
            SimpleITK scipy scikit-learn albumentations tqdm matplotlib \
            pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.1.0 pylibjpeg-openjpeg==2.2.1
```

### 사전학습 가중치 (EVA-X small)
```bash
mkdir -p ~/eva_x_backup && cd ~/eva_x_backup
wget https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_small_patch16_merged520k_mim.pt
```

### 데이터 경로
`datasets.py` 상단의 4개 상수를 각자 환경에 맞게 수정:
```python
TRAIN_DCM_DIR  = ".../data_original/train/CXR"
TRAIN_MASK_DIR = ".../data_original/train/CXR_label"
VAL_DCM_DIR    = ".../data_original/val/CXR"
VAL_MASK_DIR   = ".../data_original/val/CXR_label"
```
챌린지 데이터는 저장소에 포함되어 있지 않다(배포 규약).

---

## 재현

### 학습 (external 제출 모델)
```bash
python main.py --model evax_seg --tag evax_pct_l01 --channels 1 \
    --lambda_cls 0.1 --target_size 1024 --batch_size 8 \
    --optimizer adamw --initial_lr 5e-5 \
    --scheduler cosine --warmup 5 --max_epochs 150
```
A5000 24GB 기준 batch 8까지 가능(flash attention). 150 epoch 약 5~6시간.

### 추론 + 평가
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --out_dir results_final \
    --detection cls --cls_threshold 0.5 --min_pixels 0
```
평가는 챌린지 공식 `evaluate_task1.py`로 수행한다.

### 진단용 sweep
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --sweep
```
cls / seg-max / top-100 / soft-area 4가지 detection 방식의 threshold 별 정확도,
false negative 진단, cls-seg 불일치 케이스 목록을 출력.

### Docker
```bash
cd task1_submit_pct_l01_tta
# weights/best_evax_cls.pth 를 먼저 배치할 것
docker build -f Dockerfile_task1 -t rami-task1:latest .
docker run --rm --network none \
    -v /path/to/input:/input:ro -v /path/to/output:/output \
    rami-task1:latest
```

---

## 핵심 설계 결정

### 1. 체크포인트 선택 기준 = 챌린지 지표
`val_dice`가 아니라 **`0.7 × cls_acc + 0.3 × dice`** 가 최대인 epoch을 저장.
detection 가중치가 Dice의 2배 이상이므로 Dice만 보고 고르면 체계적으로 손해다.
(`--lambda_cls 0`인 경우에는 cls_acc가 무의미하므로 dice만 사용.)

### 2. Percentile 정규화
min–max의 기준점을 실제 min/max 대신 p1/p99로 바꾼다. 극단값 몇 픽셀
(라벨 마커, 검출기 인공물, 검은 테두리)이 전체 스케일을 좌우하는 문제를
없애며, DICOM window 태그와 달리 모든 영상에 동일하게 적용된다.
external에서 +0.0145로 단일 변경 중 가장 큰 개선이었다.

### 3. Classification-driven decision
```
present = cls_prob >= 0.5            # 원본/좌우반전 view의 평균

present & (P >= 0.5) non-empty  → mask = (P >= 0.5)
present & (P >= 0.5) empty      → mask = (P >= 0.5 * p_max)
not present                     → mask = empty

cavity = 1 ⟺ 복원된 mask가 non-empty   (CSV/NIfTI 일관성 자동 보장)
```
두 번째 분기는 분류기가 양성인데 segmentation이 threshold를 못 넘겨
CSV=1 / mask=empty가 되는 규칙 위반을 막는다. **절대 threshold 대신 `p_max`에
대한 상대값**을 쓰는 이유는 절대값이 internal 확률 분포에 맞춰진 값이라
external로 전이되지 않기 때문이다.

### 4. 왜 segmentation veto를 제거했는가
예선에서는 "segmentation 확률이 극히 낮으면(`p_max < 0.005`) 분류기의 양성
판정을 뒤집는" 규칙이 이득이었다(internal 0.7142 → 0.7356). 그러나 external에서
Dice가 전 팀 0.05~0.20으로 붕괴하면서, 낮은 `p_max`가 병변 부재가 아니라 단지
segmentation head의 무반응을 뜻하게 되었다. veto를 제거하자 external final이
0.5170 → 0.5296으로 올랐다.

**적절한 결합 규칙은 아키텍처의 고정 속성이 아니라 두 헤드의 상대적 신뢰도의
함수다.** 이것이 본 참가의 주된 방법론적 관찰이다.

### 5. TTA는 분류 헤드에만
마스크를 여러 view로 평균하면 경계가 흐려져 Dice가 어디로 움직일지 예측하기
어렵다. 본 시스템의 Dice는 참가팀 중 상위권이었으므로, 개선 여지가 있는
분류 쪽에만 적용했다.

### 6. 공격적 scale augmentation
from-scratch 단계에서 가장 큰 향상을 준 변경.
`A.Affine(scale=(0.5,1.4), rotate=(-30,30), p=0.5)` — nnU-Net 기본값
`scale=(0.7,1.4), p=0.2` 대비 범위와 확률을 모두 키웠다.

---

## Negative results

같은 시도를 반복하지 않도록 [RESULTS.ko.md](RESULTS.ko.md)에 정리되어 있다.
요약하면 **백본 크기(base), 3채널 입력, Tversky/Boundary loss, crop 비율,
mask threshold, DropPath, cls head 강화, 크기 가중 샘플링, DICOM VOI window,
강한 augmentation** 모두 개선을 주지 못했다.

유일하게 internal Dice 0.31의 벽을 넘은 것은 **ROI 2단계 oracle 실험**
(전체 이미지 0.31 → ROI 조건 0.80)이며, 이는 병목이 모델 용량이 아니라 입력
조건임을 시사한다. 실전 파이프라인은 시간 제약으로 미구현.

---

## 참고

EVA-X 사전학습 모델:
```bibtex
@article{yao2025eva,
  title={EVA-X: A foundation model for general chest X-ray analysis with self-supervised learning},
  author={Yao, Jingfeng and Wang, Xinggang and Song, Yuehao and Zhao, Huangxuan and
          Ma, Jun and Chen, Yajie and Liu, Wenyu and Wang, Bo},
  journal={npj Digital Medicine}, volume={8}, number={1}, pages={678}, year={2025}
}
```
- EVA-X: https://github.com/hustvl/EVA-X
- 챌린지: https://github.com/mi2rl-challenge/treat-mmtb.miccai2026

# TREAT-MMTB 2026 — Task 1: TB Cavity Detection & Segmentation

MICCAI TREAT-MMTB 2026 챌린지 Task 1 (흉부 X선에서 결핵성 공동(cavity)의
환자 단위 검출 + 픽셀 단위 분할) 참가 코드.

평가 지표: `final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

---

## 현재 성능 (internal validation, 111 cases / 58 pos, 53 neg)

| 시스템 | Detection | Dice | Final |
|---|---|---|---|
| nnU-Net (plain, 5-fold) | 0.7658 | 0.2725 | 0.6178 |
| from-scratch multi-task U-Net + scale aug | 0.8378 | 0.2756 | 0.6692 |
| EVA-X, seg 기반 detection (λ=0) | 0.8468 | 0.2999 | 0.6805 |
| **EVA-X + cls head(λ=0.5) + combo detection** | **0.9189** | **0.3078** | **0.7356** |

최종 제출본은 마지막 행. Docker 이미지로 `--network none` 환경에서
동일 점수 재현 확인 완료.

---

## 구조

```
입력 CXR (1채널)
  → 하부 15% crop → resize+pad 1024 → CLAHE(2.0) → z-score
  → EVA-X small ViT (patch16, embed 384, depth 12, SwiGLU, RoPE)
      ├─ blocks 2,5,8,11 feature → SimpleFeaturePyramid(ViTDet) → FPNDecoder → seg logits
      └─ block 11 feature → GAP → dropout → linear → cls logit
  → combo detection (아래 참조) → 원본 그리드로 역변환
```

총 29.9M 파라미터 (백본 22M + 디코더/헤드 7.9M).
mmcv / mmsegmentation 불필요, timm(>=0.9, 검증 1.0.22)만 있으면 동작.

---

## 파일

| 파일 | 역할 |
|---|---|
| `main_3ch.py` | 학습 진입점. **1채널/3채널 모두 지원**하며 최종 모델은 1채널로 학습됨 |
| `models.py` | `modeltype()` 팩토리 (multitask_unet / evax_seg) |
| `models_evax.py` | `EVAXSegNet` — EVA-X 백본 + FPN 디코더 + cls head |
| `eva_x.py` | EVA-X 공식 저장소의 `checkpoint_filter_fn` (사전학습 가중치 로드용) |
| `datasets.py` | **1채널 파이프라인 (최종 모델이 사용)** |
| `datasets_3ch.py` | 3채널 실험용 (원본/CLAHE2.0/CLAHE1.0). 성능 낮아 미채택 |
| `training.py` | `fit()` / `compute_lr()` / `make_optimizer()` |
| `utils.py` | DiceCE, Tversky, Boundary loss, dice metric, checkpoint 저장 |
| `inference_evax.py` | 추론 + threshold sweep + combo detection |
| `evaluate_task1.py` | 챌린지 평가 스크립트 |
| `task1_submit_evax_cls/` | 제출용 Docker (predict + Dockerfile + requirements) |
| `figs/` | 학습 곡선 PNG |

---

## 설정

### 환경
```bash
conda create -n miccai python=3.11 -y
conda activate miccai
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==1.0.22 numpy==1.26.4 opencv-python-headless pydicom==3.0.2 \
            SimpleITK scipy albumentations tqdm matplotlib \
            pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.1.0 pylibjpeg-openjpeg==2.2.1
```

### 사전학습 가중치 (EVA-X small)
```bash
mkdir -p ~/eva_x_backup && cd ~/eva_x_backup
wget https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_small_patch16_merged520k_mim.pt
```
base 변형이 필요하면 `eva_x_base_patch16_merged520k_mim.pt` (단, 아래 negative
results 참조 — base는 성능이 더 낮았음).

### 데이터 경로
`datasets.py` / `datasets_3ch.py` 상단의 4개 상수를 각자 환경에 맞게 수정:
```python
TRAIN_DCM_DIR  = ".../data_original/train/CXR"
TRAIN_MASK_DIR = ".../data_original/train/CXR_label"
VAL_DCM_DIR    = ".../data_original/val/CXR"
VAL_MASK_DIR   = ".../data_original/val/CXR_label"
```
챌린지 데이터는 저장소에 포함되어 있지 않음(배포 규약).

---

## 재현

### 학습 (최종 모델)
```bash
python main_3ch.py --model evax_seg --tag evax_cls_l05 --channels 1 \
    --lambda_cls 0.5 --target_size 1024 --batch_size 8 \
    --optimizer adamw --initial_lr 5e-5 \
    --scheduler cosine --warmup 5 --max_epochs 150
```
A5000 24GB 기준 batch 8까지 가능(flash attention 사용). 150 epoch 약 5~6시간.

### 추론 + 평가
```bash
python inference_evax.py --model evax_seg --weights best_evax_cls_l05.pth \
    --target_size 1024 --out_dir results_final \
    --detection combo --t_veto 0.005 --min_pixels 0

python evaluate_task1.py --gt-csv data_original/val/test.csv \
    --pred-csv data_original/val/results_final/prediction.csv \
    --gt-mask-dir data_original/val/CXR_label \
    --pred-mask-dir data_original/val/results_final
```

### 진단용 sweep
```bash
python inference_evax.py --model evax_seg --weights best_evax_cls_l05.pth \
    --target_size 1024 --sweep
```
cls / seg-max / top-100 / soft-area 4가지 detection 방식의 threshold 별 정확도,
false negative 진단, cls-seg 불일치 케이스 목록을 출력.

### Docker
```bash
cd task1_submit_evax_cls
# weights/best_evax_cls.pth 를 먼저 배치할 것
docker build -f Dockerfile_task1 -t rami-task1-evax:latest .
docker run --rm --network none \
    -v /path/to/input:/input:ro -v /path/to/output:/output \
    rami-task1-evax:latest
```

---

## 핵심 설계 결정

### 1. 체크포인트 선택 기준 = 챌린지 지표
`val_dice`가 아니라 **`0.7 × cls_acc + 0.3 × dice`** 가 최대인 epoch을 저장.
detection 가중치가 Dice의 2배 이상이므로, Dice만 보고 고르면 체계적으로
손해를 본다. 예: dice 최고 epoch 70(final 0.6937)보다
epoch 83(dice 0.2768, cls 0.8829 → final 0.7010)이 실제로 더 좋다.

### 2. combo detection (cls 기본 + seg veto)
두 head의 오류 성격이 다르다. seg는 음성 케이스에서 매우 보수적이라
false positive가 거의 없고, cls는 민감하지만 정상 폐를 양성으로 오인한다.
**일치하는 케이스는 건드리지 않고(Dice 보존), 불일치 케이스만 조정:**

```
cls_pos = cls_prob >= 0.5,  seg_pos = seg_max >= 0.5
일치            → seg 그대로 (mask = seg_prob >= 0.5)
cls 양성 & seg_max >= t_veto → cls 채택, mask = seg_prob >= t_veto
cls 양성 & seg_max <  t_veto → seg의 강한 음성 신뢰, mask = empty
cls 음성 & seg 양성          → seg 채택
cavity = 1 ⟺ mask non-empty  (CSV/마스크 일관성 자동 보장)
```
`t_veto = 0.005`. 더 낮추면 validation 1케이스를 더 얻지만
seg 확률 0.001~0.003 경계에 과적합되므로 보수적 값을 채택.

### 3. λ(cls loss 가중치)가 combo의 효용을 좌우
λ가 작으면 cls head가 seg 인코더를 그대로 따라가 두 head가 유사해지고,
불일치가 거의 없어 combo의 이득이 사라진다(λ=0.1: 불일치 4건, 전부 cls가 정답).
λ가 크면 cls가 공격적(recall 위주)이 되어 정상 폐에 false positive를 내고,
바로 그 케이스에서 seg의 확신 있는 음성이 상보적으로 작동한다
(λ=0.5: 불일치 9건, 그중 4건을 seg veto가 정정, 순 +2).
따라서 **combo를 쓴다면 λ는 큰 쪽이 유리**하다.

### 4. cosine + warmup (poly 아님)
poly decay로 학습하면 후반부(epoch ~88)에서 validation Dice가 0으로 붕괴했다.
cosine annealing + 5 epoch linear warmup으로 150 epoch까지 안정적으로 학습된다.

### 5. 공격적 scale augmentation
from-scratch 단계에서 가장 큰 성능 향상을 준 변경.
`A.Affine(scale=(0.5,1.4), rotate=(-30,30), p=0.5)` — nnU-Net 기본값
`scale=(0.7,1.4), p=0.2` 대비 범위와 확률을 모두 키웠다
(detection 0.775 → 0.820). 다만 `p`가 scale과 rotation에 함께 걸리므로
어느 쪽이 기여했는지는 분리 검증되지 않음.

---

## Negative results

같은 시도를 반복하지 않도록 기록. **모든 시도에서 Dice가 0.26~0.31 구간을
벗어나지 못했다.**

| 시도 | 결과 | 비고 |
|---|---|---|
| EVA-X **base** 백본 (86M) | λ0.5: ~0.691 / λ0.3: ~0.711 | small보다 낮음. 444장으로 fine-tune 시 과적합(train loss 0.37에서 val 정체) |
| **3채널 입력** (원본/CLAHE2.0/CLAHE1.0) | 1채널보다 낮음 | EVA-X 사전학습이 "grayscale 3채널 복제"를 가정하므로 서로 다른 채널을 주면 표현이 깨짐 |
| **Focal Tversky** (α0.3 β0.7 γ1.33) | dice 0.2963 | DiceCE(0.2942~0.2999)와 사실상 동일 |
| **Dice + Boundary** (w=0.5) | dice 0.2975 | 위와 동일. cosine 후반부에서 loss 종류와 무관하게 수렴 |
| **하부 crop 20%** (기본 15%) | ~0.714 | 차이 없음 |
| **mask threshold sweep** (from-scratch 모델) | 0.2756 → 0.2760 | 확률맵이 극단적으로 이분화되어 threshold가 무의미 |
| **3채널 multi-task U-Net** | 진행 중 판단 보류 | U-Net 계열은 800 epoch 이상 필요, 짧은 학습으로 판단 불가 |

> base / crop20 수치는 학습 중 validation 근사값(`0.7×cls_acc + 0.3×dice`)이며
> combo를 적용한 실제 evaluate 값이 아님. 어느 쪽이든 최종 제출본(0.7356)에
> 미치지 못해 채택하지 않았다.

---

## 남은 과제

Dice가 **0.30 근처에서 구조적으로 막혀 있다.** 백본 크기, λ, crop 비율,
loss 함수, threshold를 모두 바꿔봤지만 0.26~0.31을 벗어나지 못했다.
리더보드 상위권 점수를 역산하면 그들의 Dice는 0.45~0.60 수준으로 추정되며,
이는 튜닝 차이가 아니라 접근 방식의 차이로 보인다. 검토할 방향:

1. **해상도 예산.** FPN 디코더가 H/4(256×256)에서 로짓을 만든 뒤 4배
   업샘플한다. 1024 입력에서 40~80픽셀인 병변이 결정 단계에서는 10~20픽셀에
   불과하다. → **ROI 2단계**(1단계로 위치 검출 → crop 확대 → 2단계 정밀 분할).
   먼저 GT bbox로 crop한 oracle 조건에서 Dice 상한을 재보면 해상도가 병목인지
   반나절 안에 확인 가능.
2. **Annotation 컨벤션.** 예측을 육안 검토하면 병변을 지나치게 크게 잡거나
   지나치게 작게 잡는 오류가 양방향으로 나타난다. CXR 마스크에는 'air'
   (공기 음영만)와 'anatomy'(주변 음영 포함) 두 가지 주석 전략이 알려져 있고
   (CheXmask, Sci Data 2024), GT에 두 방식이 섞여 있다면 모델은 그 중간을
   학습하게 되어 어느 쪽과도 맞지 않는다. GT 마스크의 군집 여부 확인 필요.
3. **Layer-wise LR decay.** EVA-X 공식 segmentation recipe는 LLRD 0.85를
   사용하지만 현재 코드는 백본 전체에 동일 lr을 적용한다. Dice보다는
   external test 강건성에 기여할 것으로 예상.
4. **작은 고립 병변.** from-scratch 모델과 EVA-X가 **동일한 케이스들을**
   놓친다(seg 확률 최대값 < 0.001). 사전학습으로도 해소되지 않는 subgroup.

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

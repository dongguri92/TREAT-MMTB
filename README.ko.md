# TREAT-MMTB 2026 — Task 1: TB Cavity Detection & Segmentation

MICCAI TREAT-MMTB 2026 챌린지 Task 1 (흉부 X선에서 결핵성 공동(cavity)의
환자 단위 검출 + 픽셀 단위 분할) 참가 코드.

평가 지표: `final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

**최종 순위 6위.** 상세한 실험 기록·실패 분석·negative results는
[RESULTS.ko.md](RESULTS.ko.md) 참조. ([English](README.md))

---

## 두 단계의 전환

이 프로젝트는 두 번의 구조 전환을 거쳤고, 각각은 이전 시스템이 풀지 못한
실패에서 출발했다.

| 단계 | 시스템 | Internal | External | 코드 |
|---|---|---|---|---|
| 0 | from-scratch multi-task U-Net | 0.6692 | — | — |
| 1 | EVA-X + classification-driven 결정 | 0.7356 | 0.5463 | [`evax_baseline/`](evax_baseline/) |
| 2 | **X-Raydar two-stage (최종)** | **0.7879** | **0.5838** | [`xraydar_two_stage/`](xraydar_two_stage/) |

**0 → 1.** 세심하게 튜닝한 from-scratch 모델이 벽에 부딪혔다. 작고 고립된
공동 몇 건이 아키텍처와 loss를 바꿔도 모델이 거의 반응 하지 못했고 이는
threshold 문제가 아니라 표현의 한계로 판단했다. 흉부 X선 자기지도 foundation
model인 EVA-X로 전환했다.

**1 → 2.** EVA-X 시스템 안에서 분류 헤드와 분할 헤드의 오류 양상이 달랐고,
마스크 기반 근거는 전용 분류기보다 신뢰도가 낮았다 — 특히 external 도메인
이동에서 두드러졌다. 두 과제를 완전히 분리하고 160만 장 이상의 흉부
X선으로 학습된 X-Raydar를 전이해 검출과 국소화를 별도 네트워크로 최적화했다.

### 대표 결과

Internal validation (공개된 111건):

| 단계 | 시스템 | Accuracy | Dice | Score |
|---|---|---|---|---|
| 0 | nnU-Net (plain, 5-fold) | 0.7658 | 0.2725 | 0.6178 |
| 0 | from-scratch multi-task U-Net + scale aug | 0.8378 | 0.2756 | 0.6692 |
| 1 | EVA-X λ=0.5 + segmentation veto | 0.9189 | 0.3078 | 0.7356 |
| 2 | 1024px X-Raydar + U-Net | 0.9550 | 0.4300 | 0.7975 |
| 2 | **X-Raydar, EMA ensemble + dual-decoder** | 0.9369 | **0.4400** | 0.7879 |

External set:

| 단계 | 시스템 | Detection | Dice | Final |
|---|---|---|---|---|
| 1 | EVA-X λ=0.5 + segmentation veto | 0.6713 | 0.1568 | 0.5170 |
| 1 | EVA-X, classification-driven + percentile + TTA | 0.7089 | 0.1667 | 0.5463 |
| 2 | **X-Raydar two-stage (최종 제출)** | **0.7631** | 0.1653 | **0.5838** |

단계를 거치며 detection은 꾸준히 올랐지만 Dice는 끝내 잘 전이되지 않았다 —
external에서는 리더보드 전체가 0.05~0.20 사이에 머물렀다.

---

## 1단계 — EVA-X + classification-driven 결정

코드: [`evax_baseline/`](evax_baseline/)

```
입력 CXR (1채널)
  → percentile(1,99) clip → [0,1]
  → 하부 15% crop → resize+pad 1024 → CLAHE(2.0) → z-score
  → EVA-X small ViT (patch16, embed 384, depth 12, SwiGLU, RoPE)
      ├─ blocks 2,5,8,11 feature → SimpleFeaturePyramid(ViTDet)
      │    → FPNDecoder → seg logits
      └─ block 11 feature → GAP → dropout → linear → cls logit
  → classification-driven decision → 원본 그리드로 역변환
```

총 29.9M 파라미터. mmcv / mmsegmentation 불필요, timm(검증 1.0.22)만 있으면 동작.

### 결정 규칙

```
present = cls_prob >= 0.5            # 원본/좌우반전 view의 평균

present 이고 (P >= 0.5) 가 비어 있지 않음  → mask = (P >= 0.5)
present 이고 (P >= 0.5) 가 비어 있음        → mask = (P >= 0.5 * p_max)
present 아님                                → mask = empty

cavity = 1 ⟺ 복원된 mask가 non-empty      (CSV/NIfTI 일관성 자동 보장)
```

두 번째 분기는 분류기가 양성인데 마스크가 비어 CSV=1 / mask=empty가 되는
규칙 위반을 막는다. **절대 threshold 대신 `p_max`에 대한 상대값**을 쓰는
이유는, 절대값이 internal 확률 분포에 맞춰진 값이라 external로 전이되지
않기 때문이다.

### 이 단계 안에서의 External 진행

| 제출 | detection | Dice | final |
|---|---|---|---|
| EVA-X λ=0.5 + segmentation veto | 0.6713 | 0.1568 | 0.5170 |
| λ=0.1 + classification-driven (veto 제거) | 0.6862 | 0.1644 | 0.5296 |
| + percentile 정규화 | 0.7063 | 0.1657 | 0.5441 |
| **+ horizontal-flip TTA (분류 헤드만)** | **0.7089** | **0.1667** | **0.5463** |

> 채택한 세 변경 모두 internal 점수는 **낮아졌으나** external에서 개선되었다.
> 자세한 비교는 [RESULTS.ko.md](RESULTS.ko.md) 참조.

---

## 2단계 — X-Raydar 2단계 시스템 (최종 제출)

코드: [`xraydar_two_stage/`](xraydar_two_stage/) *(PR로 추가 예정)*

핵심 설계는 **task ownership**이다. GT 마스크가 GT 클래스를
결정하지만, 픽셀 단위 분할의 위험과 이미지 단위 판정의 위험은 성격이 다르다.
가짜 영역 하나가 위양성을 만들고, 작은 공동 하나를 놓치면 분류 자체가
위음성이 된다. 둘을 한 경로에 밀어넣으면 양쪽 다 나빠졌다.

**분류기가 존재 판정을 소유한다.** X-Raydar XNet38(Inception-v3) 1024px,
binary cross-entropy(보조 헤드 가중치 0.4), AdamW lr 1e-4, weight decay 0.05,
EMA decay 0.99로 fine-tune. EMA checkpoint 세 시점(epoch 215, 282, 299)의
sigmoid 확률을 평균해 threshold 0.5로 판정한다 — calibration layer, logit
bias, modality별 threshold 모두 사용하지 않는다.

**분할은 양성 케이스의 국소화만 담당한다.** 1024px에서 독립적인 X-Raydar
인코더 두 개: 5단계 U-Net(768/288/192/64채널 skip을 쓰는 transposed
convolution 디코더)과 FPN pixel decoder + 15-query Mask2Former. 둘 다 공동
양성 샘플로만 학습한다. 두 출력은 **centered-logit 앙상블**로 합치는데,
각 디코더를 자신의 경계 기준으로 중심화한 뒤 평균한다:

```
z = ½[logit(p_unet) − logit(0.06)] + ½[logit(p_m2f) − logit(0.54)]
mask = I[z ≥ 0]
```

분류기가 음성이면 분할을 건너뛰고 전부 0인 마스크를 쓴다. 양성인데 마스크가
비면 최대값의 절반 이상인 픽셀을 남기고, 그래도 비면 peak 픽셀을 보존한다.

**전처리.** DICOM 픽셀을 `BitsStored`가 함의하는 최대 코드로 나누고, pixel
padding 값을 히스토그램에서 제외하며, presentation polarity를 한 번만
적용한다. XGBoost 회귀 두 개가 영상별 하한·상한 백분위 순위를 예측하고, 그
강도 구간을 [0,1]로 매핑한다. 공개 Shenzhen 데이터를 품질 관리된 캐시로
추가했고, Montgomery는 마스크 검토 후 제외했다.

### 이 단계 안에서의 개발 (internal)

| 설정 | Accuracy | Dice | Score |
|---|---|---|---|
| 512px X-Raydar + UPerNet | 0.9189 | 0.4148 | 0.7677 |
| 1024px X-Raydar + U-Net | 0.9550 | 0.4300 | 0.7975 |
| **X-Raydar, EMA ensemble + dual-decoder (final)** | 0.9369 | **0.4400** | 0.7879 |

두 과제를 분리하자마자 Dice가 0.31에서 0.41로 올랐고, 1024px로 키우며 0.02가
더 붙었다. 최종 시스템은 정확도를 조금 내주고 Dice를 최대로 가져간 구성이며,
분류기는 AUROC 0.9815를 기록했다(threshold 0.5에서 TP 52, TN 52, FP 1, FN 6).
분류기 gate 없이 양성 케이스만의 앙상블 Dice는 0.4944, gate 적용 후 평균
Dice는 0.4400이었다.

**External: 0.5838** (detection 0.7631, Dice 0.1653) — 최종 제출본.

---

## 두 단계를 관통하는 관찰

**분할의 확신도가 더 강한 이미지 단위 판정을 뒤집어서는 안 된다.** 예선에서는
확신 있게 음성인 분할 맵이 양성 분류를 뒤집게 하는 것이 이득이었다
(EVA-X 0.7142 → 0.7356). 그러나 external에서 전 팀의 Dice가 0.05~0.20으로
붕괴하자 같은 규칙이 맞는 판정을 뒤집었고, 제거하자 external이 0.5170 →
0.5296으로 올랐다. 2단계는 이 원칙을 구조에 새겼다 — 분류기가 단독으로
존재를 판정하고, 분할은 그 결정에 관여하지 않는다.

**internal 점수차가 external을 예측하지 못했다.** 측정한 네 변경 중 세 번에서
internal과 external의 효과가 부호까지 반대였다. "어느 구성 요소가 먼저
무너지고, 전체가 그와 함께 무너지는가" 같은 메커니즘 수준의 논거가 internal
리더보드보다 나은 지침이었다.

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
`evax_baseline/datasets.py` 상단의 4개 상수를 각자 환경에 맞게 수정:
```python
TRAIN_DCM_DIR  = ".../data_original/train/CXR"
TRAIN_MASK_DIR = ".../data_original/train/CXR_label"
VAL_DCM_DIR    = ".../data_original/val/CXR"
VAL_MASK_DIR   = ".../data_original/val/CXR_label"
```
챌린지 데이터는 저장소에 포함되어 있지 않다(배포 규약).

---

## 재현 (1단계)

### 학습
```bash
cd evax_baseline
python main.py --model evax_seg --tag evax_pct_l01 --channels 1 \
    --lambda_cls 0.1 --target_size 1024 --batch_size 8 \
    --optimizer adamw --initial_lr 5e-5 \
    --scheduler cosine --warmup 5 --max_epochs 150
```
A5000 24GB 기준 batch 8까지 가능(flash attention). 150 epoch 약 5~6시간.

### 추론
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --out_dir results_final \
    --detection cls --cls_threshold 0.5 --min_pixels 0
```
평가는 [챌린지 공식 저장소](https://github.com/mi2rl-challenge/treat-mmtb.miccai2026)의
`evaluate_task1.py`로 수행한다.

### 진단용 sweep
```bash
python inference_evax.py --model evax_seg --weights best_evax_pct_l01.pth \
    --target_size 1024 --sweep
```
cls / seg-max / top-100 / soft-area 4가지 detection 방식의 threshold 별 정확도,
false negative 진단, cls-seg 불일치 케이스 목록을 출력.

### Docker
```bash
cd evax_baseline/task1_submit_pct_l01_tta
# weights/best_evax_cls.pth 를 먼저 배치할 것
docker build -f Dockerfile_task1 -t rami-task1:latest .
docker run --rm --network none \
    -v /path/to/input:/input:ro -v /path/to/output:/output \
    rami-task1:latest
```

---

## 파일

### `evax_baseline/`

| 파일 | 역할 |
|---|---|
| `main.py` | 학습 진입점 |
| `models.py` | `modeltype()` 팩토리 (multitask_unet / evax_seg) |
| `models_evax.py` | `EVAXSegNet` — EVA-X 백본 + FPN 디코더 + cls head |
| `eva_x.py` | EVA-X 공식 저장소의 `checkpoint_filter_fn` |
| `datasets.py` | 전처리 파이프라인 및 데이터 로더 |
| `training.py` | `fit()` / `compute_lr()` / `make_optimizer()` |
| `utils.py` | DiceCE, Tversky, Boundary loss, dice metric, checkpoint 저장 |
| `inference_evax.py` | 추론 + threshold sweep + 결정 규칙 |
| `tta.py` | 분류 헤드 전용 test-time augmentation |
| `task1_submit_pct_l01_tta/` | 제출용 Docker (predict + Dockerfile + requirements) |

### `xraydar_two_stage/`

PR로 추가 예정.

---

## Negative results

같은 시도를 반복하지 않도록 [RESULTS.ko.md](RESULTS.ko.md)에 정리되어 있다.
요약하면 EVA-X 시스템 안에서 **백본 크기(base), 3채널 입력, Tversky/Boundary
loss, crop 비율, mask threshold, DropPath, cls head 강화, 크기 가중 샘플링,
DICOM VOI window, 강한 augmentation** 모두 개선을 주지 못했다.

유일하게 internal Dice 0.31의 벽을 넘은 것은 **ROI 2단계 oracle 실험**
(전체 이미지 0.31 → GT 박스 주변 crop 조건 0.80)이며, 이는 병목이 모델
용량이 아니라 입력 조건임을 시사했다. 이 관찰이 2단계의 과제 분리로
이어졌다.

---

## 참고

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
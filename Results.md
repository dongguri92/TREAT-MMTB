# Results

MICCAI TREAT-MMTB 2026 — Task 1 (Cavity Detection & Segmentation)

`final = 0.7 × detection_accuracy + 0.3 × mean_Dice`

---

## External phase (본선)

조직위 external set. 5개 국제 기관(한국, 몽골, 페루, 필리핀 등) 데이터.

| 제출 | detection | Dice | final | 순위 |
|---|---|---|---|---|
| EVA-X λ=0.5 + segmentation veto (예선 제출본) | 0.6713 | 0.1568 | 0.5170 | 4위 |
| **EVA-X λ=0.1 + classification-driven (현행)** | **0.6862** | **0.1644** | **0.5296** | **4위** |

두 번째 제출로 detection +0.0149, Dice +0.0076, final +0.0126.

### 관찰

**Segmentation이 도메인 이동에 훨씬 취약하다.** internal → external에서
detection은 0.9189 → 0.6713 (73% 유지)인 반면 Dice는 0.3078 → 0.1568
(51% 유지)로 떨어졌다. 리더보드 상위권 전체가 Dice 0.11~0.22 구간에 몰려
있어, 특정 팀의 문제가 아니라 과제 자체의 성질로 보인다.

**따라서 segmentation에 의존하는 결정 규칙이 external에서 역효과였다.**
예선에서는 "segmentation 확률이 극히 낮으면 분류기의 양성 판정을 뒤집는"
veto 규칙이 이득이었다(internal final 0.7142 → 0.7356). 그러나 external에서는
낮은 `p_max`가 병변 부재를 뜻하지 않고 단지 segmentation head가 반응하지
못했음을 뜻하므로, veto가 맞는 판정을 뒤집는다. veto를 제거하자 external
final이 0.5170 → 0.5296으로 올랐다.

**Internal 점수차가 external을 예측하지 못한다.** 같은 변경(veto 제거)이
internal에서는 **−0.0161**, external에서는 **+0.0126**으로 부호까지 반대였다.
따라서 internal 리더보드 순위로 external 제출을 고르는 것은 신뢰할 수 없고,
"어느 구성 요소가 취약할 때 전체가 함께 무너지는가" 같은 메커니즘 수준의
논거가 더 유효했다.

---

## Internal validation (예선, 111 cases / 58 pos, 53 neg)

| 시스템 | detection | Dice | final |
|---|---|---|---|
| nnU-Net (plain, 5-fold) | 0.7658 | 0.2725 | 0.6178 |
| from-scratch multi-task U-Net + scale aug | 0.8378 | 0.2756 | 0.6692 |
| EVA-X, seg 기반 detection (λ=0) | 0.8468 | 0.2999 | 0.6805 |
| EVA-X λ=0.5 + segmentation veto | **0.9189** | **0.3078** | **0.7356** |
| EVA-X λ=0.1 + classification-driven | 0.9099 | 0.2751 | 0.7195 |

### λ와 결정 규칙의 상호작용

λ(분류 손실 가중치)는 분류 헤드의 공격성을 조절하고, 그에 따라 **어떤 결정
규칙이 유리한지가 바뀐다.**

| λ | cls 단독 정확도 | veto 규칙 final | cls-driven final |
|---|---|---|---|
| 0.1 | **0.9099** | 0.6981 | **0.7195** |
| 0.3 | 0.8919 | 0.6996 | 0.7149 |
| 0.5 | 0.8919 | **0.7356** | 0.7145 |

λ가 작으면 두 헤드가 유사해져 불일치가 거의 없고(λ=0.1: 4건, 전부 cls가 정답)
veto가 개입할 여지가 없다. λ가 크면 분류기가 공격적이 되어 정상 폐에 위양성을
내고, 바로 그 경우에 segmentation의 확신 있는 음성이 상보적으로 작동한다
(λ=0.5: 불일치 9건 중 4건을 veto가 정정, 순 +2건).

---

## 최종 결정 규칙 (현행 제출본)

```
present = cls_prob >= 0.5

present 이고 (P >= 0.5) 가 비어 있지 않음  -> mask = (P >= 0.5)
present 이고 (P >= 0.5) 가 비어 있음        -> mask = (P >= 0.5 * p_max)
present 아님                                -> mask = empty

cavity = 1  ⟺  복원된 mask가 non-empty      (CSV/NIfTI 일관성 자동 보장)
```

`P`는 전경 확률맵, `p_max = max P`.

두 번째 분기가 필요한 이유: 분류기가 양성이라 판정했는데 segmentation이
threshold 0.5를 넘기지 못하면 CSV=1 / mask=empty가 되어 규칙 위반이다.
internal에서 3건(77, 158, 203)이 여기 해당했고, 모두 GT 양성이었다.
**절대값 대신 `p_max`에 대한 상대 threshold**를 쓰는 이유는, 절대값은 internal
확률 분포에 맞춰진 값이라 external로 전이되지 않기 때문이다. 상대값은
스케일 불변이라 `p_max`가 얼마든 항상 non-empty를 보장한다.

이 보정으로 detection 0.9099를 유지하면서 Dice가 0.2713 → 0.2751로 올랐다.

---

## 크기별 실패 분석 (internal, λ=0.5 + veto 기준)

`analyze_masks.py` 출력. 예측이 존재하는 50건 기준.

| cavity 크기 | n | Dice | precision | recall | 면적비(pred/GT) |
|---|---|---|---|---|---|
| small | 16 | 0.266 | 0.300 | 0.292 | 1.51 |
| medium | 22 | 0.321 | 0.396 | 0.366 | 2.14 |
| large | 12 | **0.569** | **0.868** | 0.482 | 0.58 |

**large는 위치가 정확한데 GT의 절반만 그린다**(precision 0.868 / recall 0.482).
**small·medium은 GT의 1.5~2배를 그리는데도 Dice가 낮다** — 크기 문제가 아니라
위치·형태가 어긋난 것이다. 완전히 놓친 8건과 Dice가 정확히 0인 12건은 전부
small·medium이며, large에는 없다.

Dice가 0인 12건은 영상의학과 전문의 검토 결과 **폐야 내의 다른 투과성 병변**
(공기를 포함하며 다른 구조물에 둘러싸인 영역)을 잡은 것으로, 해상도나 전처리
문제가 아니었다.

---

## ROI 2단계 실험 (oracle)

"위치를 정확히 알려주면 우리 모델이 경계를 그릴 수 있는가"를 확인하기 위해,
GT bbox로 crop한 영역만 학습·평가했다(실전에서는 쓸 수 없는 oracle 조건).
`datasets_roi.py` 참조.

| ROI context | oracle Dice |
|---|---|
| 1.5× | 0.799 |
| 3.0× | 0.802 |
| 4.0× / 5.0× | ~0.80 |

**전체 이미지에서 0.31이던 Dice가 ROI 조건에서는 0.80이 된다.** 같은 모델,
같은 디코더인데 입력만 바뀌었다. 그리고 context를 1.5배에서 5배까지 키워도
Dice가 변하지 않았다 — 확대 배율이 포화됐다는 뜻이고, ROI의 이득이 해상도가
아니라 **위치를 알려준 것**에서 온다는 해석을 뒷받침한다.

다만 이는 상한이다. 실전에서는 1단계의 부정확한 박스가 들어오고, 위치를 잘못
잡은 12건은 ROI를 씌워도 엉뚱한 구조를 더 정밀하게 그릴 뿐이다. 실전
파이프라인(좌표 3중 역변환, 성분별 병합)은 아직 구현하지 않았다.

---

## Negative results

같은 시도를 반복하지 않도록 기록. **아래 모든 축에서 Dice가 0.26~0.31을
벗어나지 못했다.**

| 시도 | 결과 | 비고 |
|---|---|---|
| EVA-X **base** 백본 (86M) | λ0.5 ~0.691 / λ0.3 ~0.711 | small보다 낮음. 444장 fine-tune 시 과적합 |
| **3채널 입력** (원본/CLAHE2.0/CLAHE1.0) | 1채널보다 낮음 | 사전학습이 grayscale 3채널 복제를 가정 |
| **Focal Tversky** (α0.3 β0.7 γ1.33) | Dice 0.2963 | DiceCE와 사실상 동일 |
| **Dice + Boundary** (w=0.5) | Dice 0.2975 | 동일 |
| **하부 crop 20%** (기본 15%) | ~0.714 | 차이 없음 |
| **mask threshold sweep** | 0.2756 → 0.2760 | 확률맵이 이분화되어 무의미 |
| **DropPath 0.1** | ~0.26 | 변화 없음 |
| **cls head 강화** (hidden layer) | 하락 | |
| **크기 가중 샘플링** (small 2~5배) | detection 0.847~0.892 | detection 하락. 음성 노출 감소가 원인으로 추정 |
| **DICOM VOI window 적용** | 0.8829 / 0.2899 / 0.7050 | 111건 중 53건만 window 태그 존재 → 두 종류 정규화가 섞임 |
| **cls threshold sweep** (0.2~0.7) | 0.4~0.7 구간 완전 동일 | cls 확률이 극단적으로 이분화 |

### Percentile 정규화 (채택 검토 중)

min–max의 기준점을 실제 min/max 대신 p1/p99로 바꾼 것. 극단값(라벨 마커,
검출기 인공물, 검은 테두리) 몇 픽셀이 전체 스케일을 좌우하는 문제를 없앤다.
window와 달리 **모든 영상에 동일하게 적용**되므로 커버리지 문제가 없다.

| 모델 | detection | Dice | final |
|---|---|---|---|
| min-max λ=0.1 | 0.9099 | 0.2751 | 0.7195 |
| percentile λ=0.1 | 0.8919 | 0.3000 | 0.7143 |
| percentile λ=0.5 | 0.8829 | 0.2967 | 0.7070 |
| **percentile λ=0.7** | 0.9009 | **0.3033** | **0.7216** |
| percentile λ=0.9 | 0.8919 | 0.2659 | 0.7041 |

Dice는 일관되게 오르지만(0.2751 → 0.30 근처) detection이 소폭 떨어진다.
internal 차이가 작아 external 실측이 필요하다.

---

## 남은 과제

1. **위치 오류 12건.** Dice 0인 케이스가 전부 여기서 나온다. 폐야 내 다른
   투과성 병변을 cavity와 구분하는 문제이며, 전처리·해상도·백본 크기로는
   해결되지 않았다. ROI 2단계를 도입해도 이 12건은 남는다.
2. **ROI 실전 파이프라인.** oracle에서 Dice 0.80이 확인됐으나 좌표 역변환과
   성분별 병합이 미구현.
3. **External Dice 붕괴.** 전 팀이 0.11~0.22에 몰려 있어 과제 자체의 성질로
   보인다. 원인이 주석 컨벤션 차이인지 병변 양상 차이인지 미확인.
"""
tta.py — classification-only test-time augmentation for EVAXSegNet
==================================================================
분류 헤드의 출력만 여러 view로 평균하고, segmentation 확률맵은 원본 view의
것을 그대로 사용한다.

이유: 최종 점수는 0.7 x detection + 0.3 x Dice 이고, 우리 시스템의 Dice는
이미 참가팀 중 상위권이지만 detection에는 개선 여지가 있다. 마스크를
평균하면 경계가 바뀌어 Dice에 영향을 줄 수 있으므로, 이득이 명확한
분류 쪽에만 적용한다.

    from tta import predict_cls_tta
    fg_prob, cls_prob = predict_cls_tta(model, x, mode="flip")

    fg_prob  : 원본 view의 전경 확률맵 (H, W) — 단일 예측과 동일
    cls_prob : view 평균 분류 확률 (float)

사용 변형은 학습에서 쓴 것과 같은 계열이어야 한다. 본 파이프라인은
HorizontalFlip(p=0.5)으로 학습했으므로 좌우반전은 모델이 이미 익숙하다.
"""

import torch


TTA_MODES = ("none", "flip")


@torch.no_grad()
def predict_cls_tta(model, x, mode="flip", return_std=False):
    """분류 확률만 TTA로 평균. 세그멘테이션 맵은 원본 예측을 그대로 반환.

    model : EVAXSegNet (model.return_cls = True 상태)
    x     : (1, C, H, W) 전처리된 입력 텐서
    반환  : (fg_prob (H, W), cls_prob (float))
            return_std=True 이면 (fg_prob, cls_prob, cls_std)
    """
    # 원본 view — 마스크는 여기서만 가져온다
    seg_out, cls_logit = model(x)
    fg_prob = torch.softmax(seg_out, dim=1)[0, 1]
    cls_probs = [torch.sigmoid(cls_logit)[0, 0].item()]

    if mode == "flip":
        _, cls_logit_f = model(torch.flip(x, dims=[-1]))
        cls_probs.append(torch.sigmoid(cls_logit_f)[0, 0].item())

    n = len(cls_probs)
    cls_prob = sum(cls_probs) / n

    if return_std:
        std = (sum((c - cls_prob) ** 2 for c in cls_probs) / n) ** 0.5
        return fg_prob, cls_prob, std
    return fg_prob, cls_prob
"""
inference.py (multi-task, 해상도 가변)
=====================================
학습과 동일 전처리로 val 예측 -> prediction.csv + 원본 grid 마스크.
    load -> crop_lower(15%) -> resize_and_pad(target_size) -> CLAHE -> zscore
역변환: pred(target_size) -> unpad -> crop 크기로 resize -> 하부 band 0으로 채움
        -> 원본 grid. 이후 small-component 제거 + cls FP suppression.

Usage:
  # threshold 스윕 (마스크 저장 안 함, detection만)
  python inference.py --weights best_mtl_scale.pth --target_size 1024 --sweep

  # 최종 저장
  python inference.py --weights best_mtl_scale.pth --target_size 1024 \
      --out_dir results_scale --cls_threshold 0.5

  python inference.py --weights best_1536.pth --target_size 1536 \
      --out_dir results_1536 --cls_threshold 0.5
"""

import os
import csv
import glob
import argparse

import numpy as np
import cv2
import torch
import SimpleITK as sitk

from datasets import load_dicom_normalized, apply_clahe, zscore, LOWER_CROP_FRAC
from models import modeltype

VAL_DCM_DIR = os.path.expanduser("~/Miccai/data_original/val/CXR")
VAL_BASE = os.path.expanduser("~/Miccai/data_original/val")
GT_CSV = os.path.join(VAL_BASE, "test.csv")
CLAHE_CLIP = 2.0


def crop_lower(img, frac=LOWER_CROP_FRAC):
    h = img.shape[0]
    keep = int(round(h * (1.0 - frac)))
    return img[:keep, :]


def resize_and_pad_info(img, target_size, is_mask=False):
    """datasets.resize_and_pad와 동일 + pad 정보 반환."""
    h, w = img.shape[:2]
    th = tw = target_size
    scale = min(th / h, tw / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    padded = np.zeros((th, tw), dtype=img.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    padded[top:top + nh, left:left + nw] = resized
    return padded, (top, left, nh, nw)


def unpad_resize_restore(pred_ts, pad_info, crop_h, crop_w, orig_h, orig_w):
    top, left, nh, nw = pad_info
    valid = pred_ts[top:top + nh, left:left + nw]
    crop_mask = cv2.resize(valid.astype(np.uint8), (crop_w, crop_h),
                           interpolation=cv2.INTER_NEAREST)
    full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    hh = min(crop_h, orig_h)
    ww = min(crop_w, orig_w)
    full[:hh, :ww] = crop_mask[:hh, :ww]
    return (full > 0).astype(np.uint8)


def remove_small_components(mask, min_pixels=50):
    try:
        from scipy import ndimage
    except Exception:
        return mask
    lbl, n = ndimage.label(mask > 0)
    if n == 0:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = (lbl == i)
        if comp.sum() >= min_pixels:
            out[comp] = 1
    return out


def preprocess(dcm_path, target_size, crop_frac=LOWER_CROP_FRAC):
    img = load_dicom_normalized(dcm_path)          # [0,1], 원본 크기
    orig_h, orig_w = img.shape[:2]
    img = crop_lower(img, crop_frac)                          # crop (학습과 동일)
    crop_h, crop_w = img.shape[:2]
    img, pad_info = resize_and_pad_info(img, target_size, is_mask=False)
    img = apply_clahe(img, clip=CLAHE_CLIP)
    img = zscore(img)
    t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).unsqueeze(0).float()
    return t, orig_h, orig_w, crop_h, crop_w, pad_info


def read_gt_csv(path):
    gt = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = str(r["cavity"]).strip().lower()
            gt[r["our_id"].strip()] = 0 if s in ("0", "none", "no", "false", "") else 1
    return gt


def load_model(weights, device, model_name="multitask_unet",
               target_size=1024, pretrained=None, variant="small"):
    if model_name == "evax_seg":
        model = modeltype("evax_seg", in_channels=1, img_size=target_size,
                          pretrained_path=pretrained, variant=variant).to(device)
    else:
        model = modeltype("multitask_unet", in_channels=1).to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {weights} [{model_name}] "
          f"(epoch {ckpt.get('epoch','?')}, dice {ckpt.get('best_metric','?')})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best_mtl_scale.pth")
    ap.add_argument("--model", default="multitask_unet",
                    choices=["multitask_unet", "evax_seg"])
    ap.add_argument("--pretrained", default=None,
                    help="evax_seg 백본 사전학습 경로 (state_dict를 덮어쓰므로 "
                         "아키텍처만 맞으면 None이어도 무방)")
    ap.add_argument("--target_size", type=int, default=1024,
                    help="학습에 쓴 해상도와 반드시 동일하게")
    ap.add_argument("--out_dir", default=None,
                    help="결과 폴더명 (val/ 하위). 미지정 시 results_<weights이름>")
    ap.add_argument("--cls_threshold", type=float, default=0.5)
    ap.add_argument("--min_pixels", type=int, default=50,
                    help="0이면 small-component 제거 비활성화")
    ap.add_argument("--detection", choices=["cls", "seg", "combo"], default="cls",
                    help="cls / seg / combo(일치는 seg, 불일치만 cls+seg veto)")
    ap.add_argument("--t_veto", type=float, default=0.01,
                    help="combo: 불일치에서 cls=present여도 seg_max<t_veto면 negative로 veto")
    ap.add_argument("--no_suppress", action="store_true",
                    help="cls absent일 때 마스크 비우는 후처리 끄기 (--detection cls에서만 의미)")
    ap.add_argument("--sweep", action="store_true",
                    help="threshold 스윕만 (마스크 저장 안 함)")
    ap.add_argument("--variant", default="small", choices=["small", "base"])
    ap.add_argument("--crop_frac", type=float, default=0.15,
                    help="학습 때 쓴 값과 반드시 동일하게")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device, model_name=args.model,
                       target_size=args.target_size, pretrained=args.pretrained,
                       variant=args.variant)

    dcm_paths = sorted(glob.glob(os.path.join(VAL_DCM_DIR, "*.dcm")),
                       key=lambda p: int(os.path.basename(p).replace(".dcm", "")))
    print(f"{len(dcm_paths)} cases | target_size={args.target_size}")

    # ---------------- sweep 모드 ----------------
    if args.sweep:
        gt = read_gt_csv(GT_CSV)
        TOPK = 100
        cls_probs, seg_max, seg_topk, seg_area = {}, {}, {}, {}
        with torch.no_grad():
            for dp in dcm_paths:
                cid = os.path.basename(dp).replace(".dcm", "")
                x, *_ = preprocess(dp, args.target_size, args.crop_frac)
                model.return_cls = True
                seg_out, cls_logit = model(x.to(device))
                cls_probs[cid] = torch.sigmoid(cls_logit)[0, 0].item()
                fg = torch.softmax(seg_out, dim=1)[0, 1]
                flat = fg.flatten()
                seg_max[cid] = flat.max().item()
                seg_topk[cid] = torch.topk(flat, TOPK).values.mean().item()
                seg_area[cid] = flat.sum().item()

        cids = list(cls_probs.keys())

        def sweep(name, scores, thresholds, fmt="{:>7.3f}"):
            print(f"\nsweep — detection from {name}:")
            print(f"{'thr':>9} {'acc':>7} {'pred_pos':>9} {'TP':>4} {'FP':>4} {'FN':>4}")
            best = (0, None)
            for thr in thresholds:
                correct = tp = fp = fn = pred_pos = 0
                for cid in cids:
                    pred = int(scores[cid] >= thr)
                    g = gt.get(cid, 0)
                    pred_pos += pred
                    correct += int(pred == g)
                    if pred == 1 and g == 1: tp += 1
                    if pred == 1 and g == 0: fp += 1
                    if pred == 0 and g == 1: fn += 1
                acc = correct / len(cids)
                if acc > best[0]:
                    best = (acc, thr)
                print(f"{fmt.format(thr):>9} {acc:>7.4f} {pred_pos:>9} "
                      f"{tp:>4} {fp:>4} {fn:>4}")
            print(f"  best: acc={best[0]:.4f} @ thr={best[1]}")

        prob_thrs = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        sweep("cls head", cls_probs, prob_thrs)
        sweep("seg max prob", seg_max, prob_thrs)
        sweep(f"seg top-{TOPK} mean", seg_topk, prob_thrs)
        sweep("seg soft area (sum of fg prob)",
              seg_area, [0.1, 1, 5, 10, 50, 100, 500, 1000, 5000],
              fmt="{:>7.1f}")

        # FN 진단: GT=1 인데 seg max < 0.5 -> 밑에 신호가 남아 있나?
        fns = sorted([(c, seg_max[c], seg_topk[c], seg_area[c], cls_probs[c])
                      for c in cids
                      if gt.get(c, 0) == 1 and seg_max[c] < 0.5],
                     key=lambda r: -r[1])
        print(f"\nFN 진단 — GT=1 이지만 seg max<0.5 인 {len(fns)} cases:")
        print(f"{'id':>6} {'seg_max':>8} {'top100':>8} {'area':>9} {'cls':>6}")
        for c, mx, tk, ar, cp in fns:
            print(f"{c:>6} {mx:>8.4f} {tk:>8.4f} {ar:>9.2f} {cp:>6.3f}")

        # TN 여유 확인: GT=0 이고 seg max<0.5 인 케이스들의 상위 신호
        tns = sorted([(c, seg_max[c], seg_topk[c], seg_area[c])
                      for c in cids
                      if gt.get(c, 0) == 0 and seg_max[c] < 0.5],
                     key=lambda r: -r[1])[:10]
        print(f"\nTN 상위 10 (GT=0, seg max<0.5) — 낮출 때 뚫고 올라올 후보:")
        print(f"{'id':>6} {'seg_max':>8} {'top100':>8} {'area':>9}")
        for c, mx, tk, ar in tns:
            print(f"{c:>6} {mx:>8.4f} {tk:>8.4f} {ar:>9.2f}")

        # cls / seg 불일치 (thr 0.5 기준)
        dis = [(c, cls_probs[c], seg_max[c], gt.get(c, 0)) for c in cids
               if (cls_probs[c] >= 0.5) != (seg_max[c] >= 0.5)]
        print(f"\ncls/seg 불일치 {len(dis)} cases (thr 0.5):")
        print(f"{'id':>6} {'cls':>6} {'seg':>6} {'GT':>3}  누가 맞음")
        for c, cp, sp, g in dis:
            winner = "seg" if (int(sp >= 0.5) == g) else "cls"
            print(f"{c:>6} {cp:>6.3f} {sp:>6.3f} {g:>3}  {winner}")
        return

    # ---------------- 최종 저장 ----------------
    out_name = args.out_dir or ("results_" +
                                os.path.basename(args.weights).replace(".pth", ""))
    results_dir = os.path.join(VAL_BASE, out_name)
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    n_pos = 0
    with torch.no_grad():
        for i, dp in enumerate(dcm_paths):
            cid = os.path.basename(dp).replace(".dcm", "")
            x, oh, ow, ch, cw, pad_info = preprocess(dp, args.target_size, args.crop_frac)
            x = x.to(device)

            model.return_cls = True
            seg_out, cls_logit = model(x)
            cls_prob = torch.sigmoid(cls_logit)[0, 0].item()
            fg_prob = torch.softmax(seg_out, dim=1)[0, 1]
            seg_prob = fg_prob.max().item()

            combo_present = None
            if args.detection == "seg":
                pred_ts = (fg_prob >= args.cls_threshold).cpu().numpy().astype(np.uint8)
                score = seg_prob
            elif args.detection == "combo":
                cls_pos = cls_prob >= 0.5
                seg_pos = seg_prob >= 0.5
                if cls_pos == seg_pos:                        # 일치 -> seg 그대로
                    pred_ts = (fg_prob >= 0.5).cpu().numpy().astype(np.uint8)
                    combo_present = seg_pos
                elif cls_pos and seg_prob >= args.t_veto:     # cls 믿고 살림
                    pred_ts = (fg_prob >= args.t_veto).cpu().numpy().astype(np.uint8)
                    combo_present = True
                elif cls_pos and seg_prob < args.t_veto:      # seg 강한 veto
                    pred_ts = np.zeros(fg_prob.shape, dtype=np.uint8)
                    combo_present = False
                else:                                          # cls absent, seg present
                    pred_ts = (fg_prob >= 0.5).cpu().numpy().astype(np.uint8)
                    combo_present = True
                score = None
            else:
                pred_ts = seg_out.argmax(1)[0].cpu().numpy().astype(np.uint8)
                score = cls_prob

            pred_orig = unpad_resize_restore(pred_ts, pad_info, ch, cw, oh, ow)
            pred_orig = remove_small_components(pred_orig, min_pixels=args.min_pixels)

            if args.detection == "combo":
                # 일관성: present <-> 마스크 non-empty. 마스크 실제 상태로 확정
                present = pred_orig.sum() > 0
            else:
                present = score >= args.cls_threshold
                if args.detection == "cls" and not present and not args.no_suppress:
                    pred_orig = np.zeros_like(pred_orig)
            n_pos += int(present)
            rows.append((cid, int(present)))

            sitk.WriteImage(sitk.GetImageFromArray(pred_orig.astype(np.uint8)),
                            os.path.join(results_dir, f"{cid}.nii.gz"))

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dcm_paths)}] {cid} "
                      f"cls={cls_prob:.3f} seg={seg_prob:.3f} present={int(present)} "
                      f"fg={int(pred_orig.sum())}")

    csv_path = os.path.join(results_dir, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["our_id", "cavity"])
        w.writerows(rows)

    print(f"\npredicted present (cavity=1): {n_pos}/{len(rows)}")
    print(f"-> {results_dir}")
    print(f"\nevaluate:")
    print(f"python evaluate_task1.py --gt-csv {GT_CSV} "
          f"--pred-csv {csv_path} "
          f"--gt-mask-dir {VAL_BASE}/CXR_label "
          f"--pred-mask-dir {results_dir}")


if __name__ == "__main__":
    main()
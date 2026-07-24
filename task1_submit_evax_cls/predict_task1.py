"""
predict_task1.py — TREAT-MMTB 2026 Task 1 inference (EVA-X + combo detection)
============================================================================
Backbone : EVA-X small (CXR foundation model) + FPN decoder + cls head
Weights  : best_evax_cls.pth (fine-tuned; 백본까지 전부 포함, 사전학습 .pt 불필요)

I/O:
    /input/<our_id>/*.dcm  ->  /output/<our_id>.nii.gz  +  /output/prediction.csv

Preprocessing (학습/검증과 동일, 1채널):
    load DICOM -> crop_lower(15%) -> resize_and_pad(1024) -> CLAHE(2.0) -> zscore

Detection = COMBO (일치 케이스는 seg, 불일치만 cls+seg-veto):
    cls_pos = cls_prob >= 0.5
    seg_pos = seg_max  >= 0.5
    if cls_pos == seg_pos:                 # 일치 -> seg 그대로
        mask = (seg_prob >= 0.5)
    elif cls_pos and seg_max >= T_VETO:    # cls 믿고 살림
        mask = (seg_prob >= T_VETO)
    elif cls_pos and seg_max <  T_VETO:    # seg의 강한 negative veto
        mask = empty
    else:                                  # cls absent, seg present
        mask = (seg_prob >= 0.5)
    cavity = 1 iff mask non-empty          # CSV/mask 항상 일관
"""

import os
import csv
import glob
import argparse

import numpy as np
import cv2
import pydicom
import SimpleITK as sitk
import torch
from tqdm import tqdm

from models_evax import EVAXSegNet

TARGET_SIZE = 1024
CLAHE_CLIP = 2.0
LOWER_CROP_FRAC = 0.15
SEG_THRESHOLD = 0.5
T_VETO = 0.005                    # combo: 불일치에서 cls veto하는 seg 하한


# --------------------------------------------------------------------------
def load_dicom_normalized(path):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def crop_lower(img, frac=LOWER_CROP_FRAC):
    h = img.shape[0]
    keep = int(round(h * (1.0 - frac)))
    return img[:keep]


def resize_and_pad(img, target_size):
    h, w = img.shape[:2]
    th = tw = target_size
    scale = min(th / h, tw / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((th, tw), dtype=img.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    padded[top:top + nh, left:left + nw] = resized
    return padded, (top, left, nh, nw)


def apply_clahe(img01, clip=CLAHE_CLIP, grid=(8, 8)):
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(u8).astype(np.float32) / 255.0


def zscore(img):
    m, s = img.mean(), img.std()
    return (img - m) / (s + 1e-8)


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


# --------------------------------------------------------------------------
def load_model(weights_path, device):
    # pretrained_path=None: fine-tuned 가중치가 백본까지 덮어쓰므로 불필요
    model = EVAXSegNet(in_channels=1, num_classes=2,
                       img_size=TARGET_SIZE, pretrained_path=None).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model.return_cls = True
    return model


def find_dcm(case_dir):
    paths = sorted(glob.glob(os.path.join(case_dir, "*.dcm")))
    if not paths:
        paths = sorted(p for p in glob.glob(os.path.join(case_dir, "*"))
                       if os.path.isfile(p))
    return paths[0] if paths else None


@torch.no_grad()
def run_case(model, device, case_dir, output_dir):
    our_id = os.path.basename(os.path.normpath(case_dir))
    dcm_path = find_dcm(case_dir)

    if dcm_path is None:
        sitk.WriteImage(sitk.GetImageFromArray(np.zeros((1, 1), np.uint8)),
                        os.path.join(output_dir, f"{our_id}.nii.gz"))
        return our_id, 0

    xray_itk = sitk.ReadImage(dcm_path)
    ref = sitk.GetArrayFromImage(xray_itk)
    is_3d = (xray_itk.GetDimension() == 3)
    if is_3d:
        orig_h, orig_w = ref.shape[1], ref.shape[2]
    else:
        orig_h, orig_w = ref.shape[0], ref.shape[1]

    img = load_dicom_normalized(dcm_path)
    img = crop_lower(img)
    crop_h, crop_w = img.shape[:2]
    img, pad_info = resize_and_pad(img, TARGET_SIZE)
    img = apply_clahe(img)
    img = zscore(img)
    x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).unsqueeze(0).float().to(device)

    seg_out, cls_logit = model(x)
    cls_prob = torch.sigmoid(cls_logit)[0, 0].item()
    fg_prob = torch.softmax(seg_out, dim=1)[0, 1]          # (H,W)
    seg_max = fg_prob.max().item()

    # ---- COMBO ----
    cls_pos = cls_prob >= 0.5
    seg_pos = seg_max >= SEG_THRESHOLD
    if cls_pos == seg_pos:
        pred_ts = (fg_prob >= SEG_THRESHOLD)
    elif cls_pos and seg_max >= T_VETO:
        pred_ts = (fg_prob >= T_VETO)
    elif cls_pos and seg_max < T_VETO:
        pred_ts = torch.zeros_like(fg_prob, dtype=torch.bool)
    else:                                                   # cls absent, seg present
        pred_ts = (fg_prob >= SEG_THRESHOLD)
    pred_ts = pred_ts.cpu().numpy().astype(np.uint8)

    pred = unpad_resize_restore(pred_ts, pad_info, crop_h, crop_w, orig_h, orig_w)
    cavity = int(pred.sum() > 0)                            # CSV/mask 일관

    pred_out = pred[None, :, :].astype(np.uint8) if is_3d else pred.astype(np.uint8)
    pred_itk = sitk.GetImageFromArray(pred_out)
    pred_itk.CopyInformation(xray_itk)
    sitk.WriteImage(pred_itk, os.path.join(output_dir, f"{our_id}.nii.gz"))
    return our_id, cavity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--weights",
                    default="/workspace/weights/best_evax_cls.pth")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = load_model(args.weights, device)
    print(f"loaded {args.weights}", flush=True)

    case_dirs = sorted(d for d in glob.glob(os.path.join(args.input, "*"))
                       if os.path.isdir(d))
    print(f"{len(case_dirs)} cases | T_VETO={T_VETO}", flush=True)

    rows = [run_case(model, device, d, args.output)
            for d in tqdm(case_dirs, desc="Inference")]

    with open(os.path.join(args.output, "prediction.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["our_id", "cavity"])
        w.writerows(rows)

    n_pos = sum(c for _, c in rows)
    print(f"cavity=1: {n_pos}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
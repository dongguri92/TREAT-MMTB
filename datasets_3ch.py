"""
datasets_3ch.py  (3-channel variant of datasets.py)
===================================================
채널 구성 (정보량 증대 목적, baseline의 Ch2=원본복제 대신 CLAHE1.0 사용):
    Ch0 : 원본 (전역 대비)
    Ch1 : CLAHE clip=2.0 (국소 대비, 기존 1채널 파이프라인과 동일)
    Ch2 : CLAHE clip=1.0 (약한 국소 대비, baseline과 동일 clip)

전처리 순서:
    load original DICOM + mask
      -> crop_lower(15%)
      -> geometric aug (flip / rotate / scale / elastic)   [image+mask]
      -> resize_and_pad(target_size)
      -> intensity aug (밝기/대비/감마/노이즈)  [원본 1채널에만, CLAHE 이전]
      -> 3채널 생성: [원본, CLAHE2.0, CLAHE1.0]
      -> 채널별 z-score
    분류 라벨은 mask.sum()>0.

intensity aug를 CLAHE 이전 단계(원본)에 한 번만 걸어, 세 채널이 동일한
소스에서 파생되도록 한다. (채널별로 따로 aug하면 채널 간 관계가 깨짐.)
"""

import os
import glob
import numpy as np
import cv2
import pydicom
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    _HAS_ALBU = True
except Exception:
    _HAS_ALBU = False


# CLAHE clip 값 (채널 순서대로; None = 원본)
CHANNEL_CLIPS = [None, 2.0, 1.0]


def load_dicom_normalized(path):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        img = img.max() - img
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def load_mask(path):
    m = sitk.GetArrayFromImage(sitk.ReadImage(path))
    m = np.squeeze(m)
    return (m > 0).astype(np.uint8)


def resize_and_pad(img, target_size, is_mask=False):
    h, w = img.shape[:2]
    th, tw = target_size, target_size
    scale = min(th / h, tw / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    padded = np.zeros((th, tw), dtype=img.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    padded[top:top + nh, left:left + nw] = resized
    return padded


def apply_clahe(img01, clip=2.0, grid=(8, 8)):
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(u8).astype(np.float32) / 255.0


def zscore(img):
    m, s = img.mean(), img.std()
    return (img - m) / (s + 1e-8)


def make_3ch(img01):
    """정규화된 원본 1채널 -> [원본, CLAHE2.0, CLAHE1.0], 채널별 z-score."""
    chans = []
    for clip in CHANNEL_CLIPS:
        c = img01 if clip is None else apply_clahe(img01, clip=clip)
        chans.append(zscore(c))
    return np.stack(chans, axis=0)          # (3, H, W)


def build_geometric_aug():
    if not _HAS_ALBU:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.5, 1.4), rotate=(-30, 30),
                 translate_percent=(0.0, 0.0),
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    ], additional_targets={'mask': 'mask'})


def build_intensity_aug():
    # 원본 1채널에 적용 (CLAHE 이전). 3채널이 동일 소스에서 파생되도록.
    if not _HAS_ALBU:
        return None
    return A.Compose([
        A.GaussNoise(p=0.15),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.3),
        A.Downscale(scale_range=(0.5, 0.9),
                    interpolation_pair={'downscale': cv2.INTER_NEAREST,
                                        'upscale': cv2.INTER_LINEAR},
                    p=0.2),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
    ])


LOWER_CROP_FRAC = 0.15


def crop_lower(img, frac=LOWER_CROP_FRAC):
    H = img.shape[0]
    keep = int(round(H * (1 - frac)))
    return img[:keep, :]


TRAIN_DCM_DIR = "/home/djk25/Miccai/data_original/train/CXR"
TRAIN_MASK_DIR = "/home/djk25/Miccai/data_original/train/CXR_label"
VAL_DCM_DIR = "/home/djk25/Miccai/data_original/val/CXR"
VAL_MASK_DIR = "/home/djk25/Miccai/data_original/val/CXR_label"


class CXRCavityDataset3ch(Dataset):
    def __init__(self, dcm_dir, mask_dir, ids, train=True,
                 target_size=1024, clahe_clip=2.0):
        self.dcm_dir = dcm_dir
        self.mask_dir = mask_dir
        self.ids = list(ids)
        self.train = train
        self.target_size = target_size
        self.geo_aug = build_geometric_aug() if train else None
        self.int_aug = build_intensity_aug() if train else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        img = load_dicom_normalized(os.path.join(self.dcm_dir, f"{cid}.dcm"))
        mask = load_mask(os.path.join(self.mask_dir, f"{cid}.nii.gz"))

        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        img = crop_lower(img)
        mask = crop_lower(mask)

        # 1) geometric aug (image + mask)
        if self.train and self.geo_aug is not None:
            out = self.geo_aug(image=img, mask=mask)
            img, mask = out['image'], out['mask']
        elif self.train and self.geo_aug is None:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])

        # 2) resize + pad
        img = resize_and_pad(img, self.target_size, is_mask=False)
        mask = resize_and_pad(mask, self.target_size, is_mask=True)

        # 3) intensity aug on ORIGINAL 1ch, BEFORE CLAHE
        if self.train and self.int_aug is not None:
            img = self.int_aug(image=img)['image']
            img = np.clip(img, 0.0, 1.0)

        # 4) 3채널 생성 + 채널별 z-score
        img3 = make_3ch(img)                       # (3, H, W)

        mask = (mask > 0).astype(np.uint8)
        cls_label = np.float32(mask.sum() > 0)

        img_t = torch.from_numpy(np.ascontiguousarray(img3)).float()   # (3,H,W)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).long()
        cls_t = torch.tensor([cls_label], dtype=torch.float32)

        return {'image': img_t, 'mask': mask_t, 'cls': cls_t, 'id': cid}


def _list_ids(dcm_dir):
    files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _cavity_presence(mask_dir, ids):
    labels = []
    for cid in ids:
        m = load_mask(os.path.join(mask_dir, f"{cid}.nii.gz"))
        labels.append(int(m.sum() > 0))
    return labels


def dataloader(batch_size=3, target_size=1024, clahe_clip=2.0,
               num_workers=8, seed=42):
    train_ids = _list_ids(TRAIN_DCM_DIR)
    val_ids = _list_ids(VAL_DCM_DIR)

    tr_labels = _cavity_presence(TRAIN_MASK_DIR, train_ids)
    va_labels = _cavity_presence(VAL_MASK_DIR, val_ids)
    print(f"[3ch] train {len(train_ids)} (pos {sum(tr_labels)}) | "
          f"val {len(val_ids)} (pos {sum(va_labels)})")

    train_ds = CXRCavityDataset3ch(TRAIN_DCM_DIR, TRAIN_MASK_DIR, train_ids,
                                   train=True, target_size=target_size)
    val_ds = CXRCavityDataset3ch(VAL_DCM_DIR, VAL_MASK_DIR, val_ids,
                                 train=False, target_size=target_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False)

    return train_loader, val_loader
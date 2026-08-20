"""
datasets.py  (v2 - dynamic preprocessing from original DICOM)
============================================================
Per-iteration pipeline (your requested order):
    load original DICOM + mask
      -> geometric augmentation (flip / rotate / scale / elastic)  [image+mask]
      -> resize_and_pad to 1024 (aspect-ratio preserved)
      -> CLAHE                                                     [image only]
      -> intensity augmentation (brightness / contrast / gamma / noise) [image]
      -> z-score normalization
    classification label derived from mask (sum > 0).

Inputs (original data):
    <dcm_dir>/<id>.dcm
    <mask_dir>/<id>.nii.gz
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

import csv as _csv
from torch.utils.data import WeightedRandomSampler


#def load_dicom_normalized(path):
#    ds = pydicom.dcmread(path, force=True)
#    img = ds.pixel_array.astype(np.float32)
#    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
#        img = img.max() - img
#    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#    return img

def load_dicom_normalized(path, p_lo=1.0, p_hi=99.0):
    ds = pydicom.dcmread(path, force=True)
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        img = img.max() - img
    lo, hi = np.percentile(img, [p_lo, p_hi])
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


TRAIN_CSV = "/home/djk25/Miccai/data_original/train/train.csv"
SIZE_WEIGHTS = {'small': 1.0, 'medium': 1.0, 'large': 1.0, 'none': 1.0}

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


def build_geometric_aug():
    """Spatial transforms on BOTH image and mask, on original resolution.
    Mirrors nnU-Net's spatial augmentation (rotation / scaling / elastic) plus
    mirroring. Probabilities/ranges chosen close to nnU-Net defaults."""
    if not _HAS_ALBU:
        return None
    return A.Compose([
        # horizontal mirror only (CXR has fixed up-down anatomy)
        A.HorizontalFlip(p=0.5),
        # rotation + scaling (nnU-Net: rot +-30 deg approx, scale 0.7-1.4)
        A.Affine(scale=(0.5, 1.4), rotate=(-30, 30),
                 translate_percent=(0.0, 0.0),
                 interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST, p=0.5),
        # elastic deformation
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    ], additional_targets={'mask': 'mask'})


def build_intensity_aug():
    #Intensity transforms on IMAGE ONLY, after CLAHE.
    #Mirrors nnU-Net's intensity augmentation: gaussian noise, gaussian blur,
    #brightness (multiplicative), contrast, simulate-low-resolution, gamma.
    if not _HAS_ALBU:
        return None
    return A.Compose([
        # gaussian noise
        A.GaussNoise(p=0.15),
        # gaussian blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # multiplicative brightness + contrast
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.3),
        # simulate low resolution: downscale then upscale back
        A.Downscale(scale_range=(0.5, 0.9),
                    interpolation_pair={'downscale': cv2.INTER_NEAREST,
                                        'upscale': cv2.INTER_LINEAR},
                    p=0.2),
        # gamma (both directions)
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
    ])

def _size_sample_weights(csv_path, ids, weights=None):
    """cavity 크기 라벨(large/medium/small/none)로 샘플 가중치 산출.
    batch_dice=True가 큰 병변에 편향되는 것을 데이터 등장 빈도로 보정한다."""
    weights = weights or SIZE_WEIGHTS
    size = {}
    try:
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                size[str(row['our_id']).strip()] = str(row['cavity']).strip().lower()
    except Exception as e:
        print(f"  [WARN] size csv 읽기 실패({e}) — 균등 샘플링으로 진행")
        return None
    w = [weights.get(size.get(str(cid), 'none'), 1.0) for cid in ids]
    from collections import Counter
    cnt = Counter(size.get(str(cid), '?') for cid in ids)
    print(f"  size 분포: {dict(cnt)} | 가중치 {weights}")
    return w

"""
def build_intensity_aug():
    if not _HAS_ALBU:
        return None
    return A.Compose([
        A.GaussNoise(p=0.25),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.15),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),
            A.MedianBlur(blur_limit=5),
            A.MotionBlur(blur_limit=7),
        ], p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.4,
                                   contrast_limit=0.4, p=0.5),
        A.RandomGamma(gamma_limit=(60, 160), p=0.5),
        A.Downscale(scale_range=(0.4, 0.9),
                    interpolation_pair={'downscale': cv2.INTER_NEAREST,
                                        'upscale': cv2.INTER_LINEAR}, p=0.25),
        A.Sharpen(alpha=(0.1, 0.4), p=0.15),
    ])
"""

LOWER_CROP_FRAC = 0.15 

def crop_lower(img, frac=LOWER_CROP_FRAC):
    """하부 frac 비율 제거. 상부 (1-frac)만 남김. img: (H, W)."""
    H = img.shape[0]
    keep = int(round(H * (1 - frac)))
    return img[:keep, :]

TRAIN_DCM_DIR = "/home/djk25/Miccai/data_original/train/CXR"
TRAIN_MASK_DIR = "/home/djk25/Miccai/data_original/train/CXR_label"
VAL_DCM_DIR = "/home/djk25/Miccai/data_original/val/CXR"
VAL_MASK_DIR = "/home/djk25/Miccai/data_original/val/CXR_label"

class CXRCavityDataset(Dataset):
    def __init__(self, dcm_dir, mask_dir, ids, train=True,
                 target_size=1024, clahe_clip=2.0, crop_frac=LOWER_CROP_FRAC):
        self.dcm_dir = dcm_dir
        self.mask_dir = mask_dir
        self.ids = list(ids)
        self.train = train
        self.target_size = target_size
        self.clahe_clip = clahe_clip
        self.geo_aug = build_geometric_aug() if train else None
        self.int_aug = build_intensity_aug() if train else None
        self.crop_frac = crop_frac

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        img = load_dicom_normalized(os.path.join(self.dcm_dir, f"{cid}.dcm"))
        mask = load_mask(os.path.join(self.mask_dir, f"{cid}.nii.gz"))

        # mask를 image 크기에 맞춤 (원본 CXR과 mask 크기가 다른 케이스 대응)
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # 하부 15% crop (이미지 + 마스크 같이)
        img = crop_lower(img, self.crop_frac)
        mask = crop_lower(mask, self.crop_frac)

        # 1) geometric aug on ORIGINAL resolution (image + mask together)
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

        # 3) CLAHE on resized image
        img = apply_clahe(img, clip=self.clahe_clip)

        # 4) intensity aug AFTER CLAHE (image only)
        if self.train and self.int_aug is not None:
            img = self.int_aug(image=img)['image']

        # 5) z-score
        img = zscore(img)

        mask = (mask > 0).astype(np.uint8)
        cls_label = np.float32(mask.sum() > 0)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).long()
        cls_t = torch.tensor([cls_label], dtype=torch.float32)

        return {'image': img_t, 'mask': mask_t, 'cls': cls_t, 'id': cid}

# =============================================================================
#  5-fold 교차검증용 — datasets.py 맨 아래(dataloader 함수 뒤)에 붙여넣기
#  기존 CXRCavityDataset / dataloader 는 건드리지 않는다.
# =============================================================================

class _CXRCavityDataset(Dataset):
    """(case_id, dcm_dir, mask_dir) 튜플 목록을 받는 데이터셋.

    train/val 폴더가 섞인 목록을 다룰 수 있어 k-fold 분할에 사용한다.
    전처리·augmentation은 CXRCavityDataset과 동일하다.
    """

    def __init__(self, items, train=True, target_size=1024,
                 clahe_clip=2.0, crop_frac=LOWER_CROP_FRAC):
        self.items = list(items)          # [(cid, dcm_dir, mask_dir), ...]
        self.train = train
        self.target_size = target_size
        self.clahe_clip = clahe_clip
        self.crop_frac = crop_frac
        self.geo_aug = build_geometric_aug() if train else None
        self.int_aug = build_intensity_aug() if train else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cid, dcm_dir, mask_dir = self.items[idx]
        img = load_dicom_normalized(os.path.join(dcm_dir, f"{cid}.dcm"))
        mask = load_mask(os.path.join(mask_dir, f"{cid}.nii.gz"))

        # 원본 CXR과 mask 크기가 다른 케이스 대응
        if mask.shape != img.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # 하부 crop (이미지 + 마스크 같이)
        img = crop_lower(img, self.crop_frac)
        mask = crop_lower(mask, self.crop_frac)

        # 1) geometric aug on ORIGINAL resolution (image + mask together)
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

        # 3) CLAHE on resized image
        img = apply_clahe(img, clip=self.clahe_clip)

        # 4) intensity aug AFTER CLAHE (image only)
        if self.train and self.int_aug is not None:
            img = self.int_aug(image=img)['image']

        # 5) z-score
        img = zscore(img)

        mask = (mask > 0).astype(np.uint8)
        cls_label = np.float32(mask.sum() > 0)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).long()
        cls_t = torch.tensor([cls_label], dtype=torch.float32)

        return {'image': img_t, 'mask': mask_t, 'cls': cls_t, 'id': cid}


def _fold_items(fold=0, n_folds=5, seed=42):
    """train + val 전체(555장)를 cavity 유무로 stratified k-fold 분할.
    반환: (train_items, val_items, train_labels, val_labels)"""
    from sklearn.model_selection import StratifiedKFold

    tr_ids = _list_ids(TRAIN_DCM_DIR)
    va_ids = _list_ids(VAL_DCM_DIR)
    items = ([(c, TRAIN_DCM_DIR, TRAIN_MASK_DIR) for c in tr_ids] +
             [(c, VAL_DCM_DIR, VAL_MASK_DIR) for c in va_ids])
    labels = (_cavity_presence(TRAIN_MASK_DIR, tr_ids) +
              _cavity_presence(VAL_MASK_DIR, va_ids))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr_idx, va_idx = list(skf.split(items, labels))[fold]

    return ([items[i] for i in tr_idx], [items[i] for i in va_idx],
            [labels[i] for i in tr_idx], [labels[i] for i in va_idx])


def dataloader_fold(fold=0, n_folds=5, batch_size=8, target_size=1024,
                    clahe_clip=2.0, num_workers=8, seed=42,
                    crop_frac=LOWER_CROP_FRAC):
    """k-fold 학습용 loader. train/val 폴더를 합쳐 사용하므로
    external phase 제출 모델 학습에만 쓴다(예선 val은 더 이상 held-out이 아님)."""
    tr_items, va_items, tr_lab, va_lab = _fold_items(fold, n_folds, seed)

    print(f"[fold {fold}/{n_folds}] train {len(tr_items)} (pos {sum(tr_lab)}) | "
          f"val {len(va_items)} (pos {sum(va_lab)})")

    train_ds = _CXRCavityDataset(tr_items, train=True,
                                 target_size=target_size,
                                 clahe_clip=clahe_clip, crop_frac=crop_frac)
    val_ds = _CXRCavityDataset(va_items, train=False,
                               target_size=target_size,
                               clahe_clip=clahe_clip, crop_frac=crop_frac)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False)

    return train_loader, val_loader


def _list_ids(dcm_dir):
    files = sorted(glob.glob(os.path.join(dcm_dir, "*.dcm")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _cavity_presence(mask_dir, ids):
    """Open each mask once to derive cavity presence (0/1) for stratification."""
    labels = []
    for cid in ids:
        m = load_mask(os.path.join(mask_dir, f"{cid}.nii.gz"))
        labels.append(int(m.sum() > 0))
    return labels


def dataloader(batch_size=3, target_size=1024, clahe_clip=2.0,
               num_workers=8, seed=42, crop_frac=LOWER_CROP_FRAC,
               size_weighted=False):
    """train 경로 전체를 train으로, val 경로 전체를 val로 사용.
    (internal validation set이 별도로 제공되므로 split 불필요)"""
    train_ids = _list_ids(TRAIN_DCM_DIR)
    val_ids = _list_ids(VAL_DCM_DIR)

    tr_labels = _cavity_presence(TRAIN_MASK_DIR, train_ids)
    va_labels = _cavity_presence(VAL_MASK_DIR, val_ids)
    print(f"train {len(train_ids)} (pos {sum(tr_labels)}) | "
          f"val {len(val_ids)} (pos {sum(va_labels)})")

    train_ds = CXRCavityDataset(TRAIN_DCM_DIR, TRAIN_MASK_DIR, train_ids,
                                train=True, target_size=target_size,
                                clahe_clip=clahe_clip, crop_frac=crop_frac)
    val_ds = CXRCavityDataset(VAL_DCM_DIR, VAL_MASK_DIR, val_ids,
                              train=False, target_size=target_size,
                              clahe_clip=clahe_clip, crop_frac=crop_frac)

    sampler = None
    if size_weighted:
        w = _size_sample_weights(TRAIN_CSV, train_ids)
        if w is not None:
            sampler = WeightedRandomSampler(w, num_samples=len(train_ids),
                                            replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False)

    return train_loader, val_loader

def dataloader_all(batch_size=8, target_size=1024, clahe_clip=2.0,
                   num_workers=8, seed=42, crop_frac=LOWER_CROP_FRAC, **kw):
    """train + val 555장 전부를 학습에 사용. held-out이 없으므로 검증
    지표는 훈련 성능이며 신뢰할 수 없다 — epoch을 미리 정해두고 쓸 것."""
    tr_ids = _list_ids(TRAIN_DCM_DIR)
    va_ids = _list_ids(VAL_DCM_DIR)
    items = ([(c, TRAIN_DCM_DIR, TRAIN_MASK_DIR) for c in tr_ids] +
             [(c, VAL_DCM_DIR, VAL_MASK_DIR) for c in va_ids])
    labels = (_cavity_presence(TRAIN_MASK_DIR, tr_ids) +
              _cavity_presence(VAL_MASK_DIR, va_ids))
    print(f"[ALL] train {len(items)} (pos {sum(labels)}) — held-out 없음, "
          f"검증 지표는 훈련 성능")

    train_ds = _CXRCavityDataset(items, train=True, target_size=target_size,
                                 clahe_clip=clahe_clip, crop_frac=crop_frac)
    val_ds = _CXRCavityDataset(items, train=False, target_size=target_size,
                               clahe_clip=clahe_clip, crop_frac=crop_frac)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False)
    return train_loader, val_loader

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
import random
from functools import partial
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


def resize_and_pad_info(img, target_size, is_mask=False):
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
    return padded, (top, left, nh, nw)


def resize_and_pad(img, target_size, is_mask=False):
    padded, _ = resize_and_pad_info(img, target_size, is_mask=is_mask)
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

TRAIN_DCM_DIR = os.environ.get(
    "TREAT_TRAIN_DCM_DIR", "/home/djk25/Miccai/data_original/train/CXR"
)
TRAIN_MASK_DIR = os.environ.get(
    "TREAT_TRAIN_MASK_DIR", "/home/djk25/Miccai/data_original/train/CXR_label"
)
VAL_DCM_DIR = os.environ.get(
    "TREAT_VAL_DCM_DIR", "/home/djk25/Miccai/data_original/val/CXR"
)
VAL_MASK_DIR = os.environ.get(
    "TREAT_VAL_MASK_DIR", "/home/djk25/Miccai/data_original/val/CXR_label"
)

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

        native_mask = (mask > 0).astype(np.uint8)
        native_shape = img.shape[:2]

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
        img, pad_info = resize_and_pad_info(
            img, self.target_size, is_mask=False
        )
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

        sample = {'image': img_t, 'mask': mask_t, 'cls': cls_t, 'id': cid}
        if not self.train:
            sample.update({
                'native_mask': torch.from_numpy(
                    np.ascontiguousarray(native_mask)
                ).unsqueeze(0).long(),
                'native_shape': torch.tensor(native_shape, dtype=torch.long),
                'crop_shape': torch.tensor(
                    [int(round(native_shape[0] * (1 - self.crop_frac))),
                     native_shape[1]],
                    dtype=torch.long,
                ),
                'pad_info': torch.tensor(pad_info, dtype=torch.long),
            })
        return sample


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


def _seed_worker(worker_id, base_seed):
    worker_seed = (base_seed + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def dataloader(batch_size=3, target_size=1024, clahe_clip=2.0,
               num_workers=8, seed=42, crop_frac=LOWER_CROP_FRAC):
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

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        generator=generator,
        worker_init_fn=partial(_seed_worker, base_seed=seed))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False)

    return train_loader, val_loader

###########################################################################

# ===========================================================================
#  self-test:  python datasets.py
# ===========================================================================

"""
if __name__ == "__main__":
    # TRAIN 경로로 테스트
    ds = CXRCavityDataset(TRAIN_DCM_DIR, TRAIN_MASK_DIR, ids=["100", "104", "105"],
                          train=True, target_size=1024, clahe_clip=2.0)

    s = ds[0]
    img, mask, cls, cid = s['image'], s['mask'], s['cls'], s['id']
    print(f"===== case {cid} (train=True) =====")
    print(f"image | shape {tuple(img.shape)} dtype {img.dtype}")
    print(f"      | min {img.min():.3f} max {img.max():.3f} "
          f"mean {img.mean():.3f} std {img.std():.3f}")
    print(f"mask  | shape {tuple(mask.shape)} dtype {mask.dtype} "
          f"unique {torch.unique(mask).tolist()} fg_px {int(mask.sum())}")
    print(f"cls   | shape {tuple(cls.shape)} value {cls.tolist()} "
          f"| matches mask? {int(cls.item()) == int(mask.sum() > 0)}")

    # 2) 여러 케이스 일관성
    print("\n===== 여러 케이스 =====")
    for i in range(len(ds)):
        x = ds[i]
        im, mk, cl = x['image'], x['mask'], x['cls']
        ok_norm = abs(im.mean().item()) < 0.15 and abs(im.std().item() - 1.0) < 0.25
        ok_mask = set(torch.unique(mk).tolist()).issubset({0, 1})
        ok_match = int(cl.item()) == int(mk.sum() > 0)
        print(f"{x['id']}: mean={im.mean():.3f} std={im.std():.3f} "
              f"mask_unique={torch.unique(mk).tolist()} cls={cl.item():.0f} "
              f"| norm{'O' if ok_norm else 'X'} "
              f"mask{'O' if ok_mask else 'X'} match{'O' if ok_match else 'X'}")

    # 3) val 모드 (augmentation 꺼짐) — VAL 경로로 테스트
    print("\n===== val 모드 (val 경로) =====")
    val_ids = _list_ids(VAL_DCM_DIR)[:1]
    ds_val = CXRCavityDataset(VAL_DCM_DIR, VAL_MASK_DIR, ids=val_ids,
                              train=False, target_size=1024, clahe_clip=2.0)
    v = ds_val[0]
    print(f"{v['id']}: mean={v['image'].mean():.3f} std={v['image'].std():.3f} "
          f"mask_unique={torch.unique(v['mask']).tolist()} cls={v['cls'].item():.0f}")

    # 4) dataloader 배치 테스트
    print("\n===== dataloader 배치 =====")
    train_loader, val_loader = dataloader(batch_size=3, num_workers=0, seed=42)
    b = next(iter(train_loader))
    print(f"batch image {tuple(b['image'].shape)} mask {tuple(b['mask'].shape)} "
          f"cls {b['cls'].squeeze().tolist()}")
    print(f"image mean {b['image'].mean():.3f} std {b['image'].std():.3f} "
          f"mask_unique {torch.unique(b['mask']).tolist()}")
    vb = next(iter(val_loader))
    print(f"val batch image {tuple(vb['image'].shape)} id {vb['id']}")
"""

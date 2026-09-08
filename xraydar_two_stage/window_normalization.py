"""Histogram-conditioned DICOM window normalization used during training."""
from pathlib import Path

import joblib
import numpy as np


PERCENTILES = np.array(
    [
        0, 0.1, 0.5, 1, 2, 5, 10, 20, 25, 40, 50,
        60, 75, 80, 90, 95, 98, 99, 99.5, 99.9, 100,
    ],
    dtype=np.float64,
)


def polarity_inverted(dataset) -> bool:
    photo = str(dataset.PhotometricInterpretation).upper()
    presentation = str(
        getattr(dataset, "PresentationLUTShape", "") or ""
    ).upper()
    if presentation == "INVERSE":
        return True
    if presentation == "IDENTITY":
        return False
    return photo == "MONOCHROME1"


def oriented_image_and_valid(dataset):
    raw = dataset.pixel_array.astype(np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError(f"expected a 2D X-ray, received {raw.shape}")
    padding = getattr(dataset, "PixelPaddingValue", None)
    padding_mask = (
        raw == float(padding)
        if padding is not None
        else np.zeros(raw.shape, dtype=bool)
    )
    valid_mask = ~padding_mask
    if not valid_mask.any():
        valid_mask = np.ones(raw.shape, dtype=bool)
        padding_mask = np.zeros(raw.shape, dtype=bool)
    maximum_code = float((1 << int(dataset.BitsStored)) - 1)
    image = raw / maximum_code
    if polarity_inverted(dataset):
        image = 1.0 - image
    return image.astype(np.float32, copy=False), valid_mask, padding_mask


def histogram_features(values):
    quantiles = np.percentile(values, PERCENTILES)
    return np.r_[
        quantiles,
        float(values.mean()),
        max(float(values.std()), 1e-6),
        float(np.mean(values <= 0.0)),
        float(np.mean(values >= 1.0)),
    ]


class XGBoostWindowNormalizer:
    def __init__(self, model_path):
        self.bundle = joblib.load(Path(model_path))

    def normalize(self, dataset):
        image, valid_mask, padding_mask = oriented_image_and_valid(dataset)
        valid = image[valid_mask]
        features = histogram_features(valid)
        models = self.bundle["models"]
        selected = self.bundle["selected"]
        low = float(models[0].predict(features[None, selected[0]])[0])
        high = float(models[1].predict(features[None, selected[1]])[0])
        ranks = np.array(
            [np.clip(low, 0.0, 0.49), np.clip(high, 0.51, 1.0)]
        )
        lower, upper = np.percentile(valid, ranks * 100)
        if upper <= lower:
            raise ValueError(
                f"invalid predicted window: lower={lower}, upper={upper}"
            )
        normalized = np.clip(
            (image - lower) / (upper - lower), 0.0, 1.0
        ).astype(np.float32)
        normalized[padding_mask] = 0.0
        return normalized

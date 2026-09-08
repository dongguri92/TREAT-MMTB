"""Offline two-stage mask-ensemble inference for TREAT-MMTB 2026 Task 1."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image

from model import (
    XRaydarCavityBinary,
    XRaydarMask2Former,
    XRaydarUNet,
    load_state,
)
from window_normalization import XGBoostWindowNormalizer


def find_dicom(case_directory: Path) -> Path:
    paths = sorted(case_directory.glob("*.dcm"))
    if not paths:
        paths = sorted(path for path in case_directory.iterdir() if path.is_file())
    if not paths:
        raise RuntimeError(f"no DICOM file found in {case_directory}")
    return paths[0]


def dicom_modality(dataset) -> str:
    """Return the normalized DICOM (0008,0060) Modality value."""
    return str(getattr(dataset, "Modality", "")).strip().upper()


def prepare_input(dataset, normalizer, runtime):
    input_size = int(runtime["input_size"])
    image = normalizer.normalize(dataset)
    height, width = image.shape
    scale = input_size / min(height, width)
    resized_height = max(input_size, round(height * scale))
    resized_width = max(input_size, round(width * scale))
    tensor = torch.from_numpy(image)[None, None]
    tensor = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0].clamp_(0, 1)
    x0 = max((resized_width - input_size) // 2, 0)
    tensor = tensor[:input_size, x0 : x0 + input_size]
    if tensor.shape != (input_size, input_size):
        raise RuntimeError(f"invalid model crop shape: {tuple(tensor.shape)}")
    xray_tensor = (
        (tensor - float(runtime["xraydar_mean"]))
        / float(runtime["xraydar_std"])
    )[None]
    mask2former_tensor = (
        (tensor - float(runtime["mask2former_input_mean"]))
        / float(runtime["mask2former_input_std"])
    )[None]
    geometry = {
        "original_size": (height, width),
        "resized_size": (resized_height, resized_width),
        "x0": x0,
    }
    return xray_tensor, mask2former_tensor, geometry


def restore_probability(probability, geometry, input_size):
    resized_height, resized_width = geometry["resized_size"]
    original_height, original_width = geometry["original_size"]
    canvas = np.zeros((resized_height, resized_width), dtype=np.float32)
    x0 = geometry["x0"]
    canvas[:input_size, x0 : x0 + input_size] = probability
    return np.asarray(
        Image.fromarray(canvas, mode="F").resize(
            (original_width, original_height), Image.Resampling.BILINEAR
        )
    )


def mask_for_reference(mask_2d, reference):
    mask_2d = (mask_2d > 0).astype(np.uint8)
    if reference.GetDimension() == 2:
        expected = (reference.GetSize()[1], reference.GetSize()[0])
        if mask_2d.shape != expected:
            raise RuntimeError(f"mask {mask_2d.shape} != reference {expected}")
        return mask_2d
    if reference.GetDimension() == 3 and reference.GetSize()[2] == 1:
        expected = (reference.GetSize()[1], reference.GetSize()[0])
        if mask_2d.shape != expected:
            raise RuntimeError(f"mask {mask_2d.shape} != reference {expected}")
        return mask_2d[None]
    raise RuntimeError(
        "unsupported X-ray reference dimension/size: "
        f"{reference.GetDimension()} / {reference.GetSize()}"
    )


def load_models(classifier_paths, unet_path, mask2former_path, device):
    if not classifier_paths:
        raise ValueError("at least one classifier checkpoint is required")
    classifiers = []
    for classifier_path in classifier_paths:
        classifier = XRaydarCavityBinary()
        load_state(classifier, classifier_path)
        classifiers.append(classifier.to(device).eval())
    unet = XRaydarUNet()
    load_state(unet, unet_path)
    mask2former = XRaydarMask2Former()
    load_state(mask2former, mask2former_path)
    return (
        classifiers,
        unet.to(device).eval(),
        mask2former.to(device).eval(),
    )


@torch.inference_mode()
def classify(classifiers, images, device, logit_bias):
    images = images.to(device, non_blocking=True)
    probability_sum = None
    for classifier in classifiers:
        logits = classifier(images).float()
        probability = (logits + logit_bias).sigmoid()
        probability_sum = (
            probability
            if probability_sum is None
            else probability_sum + probability
        )
    return (probability_sum / len(classifiers)).cpu().numpy()


@torch.inference_mode()
def segment(mask_model, images, device):
    logits = mask_model(images.to(device, non_blocking=True)).float()
    if isinstance(mask_model, XRaydarMask2Former):
        return logits.cpu().numpy()
    return logits.sigmoid().cpu().numpy()


def centered_logit_ensemble(
    unet_probability,
    mask2former_probability,
    unet_boundary,
    mask2former_boundary,
    unet_alpha,
):
    """Reproduce the released-validation ensemble on the original grid."""
    epsilon = 1e-5

    def logit(value):
        value = np.clip(value, epsilon, 1 - epsilon)
        return np.log(value / (1 - value))

    centered = (
        unet_alpha * (logit(unet_probability) - logit(unet_boundary))
        + (1 - unet_alpha)
        * (logit(mask2former_probability) - logit(mask2former_boundary))
    )
    # A probability representation makes the existing non-empty rescue policy
    # directly applicable. The validated decision boundary is 0.5.
    return 1 / (1 + np.exp(-np.clip(centered, -80, 80)))


def positive_mask(probability, threshold, rescue_fraction):
    mask = probability >= threshold
    rescue = "none"
    rescue_threshold = None
    if not mask.any():
        maximum = float(np.nanmax(probability))
        if np.isfinite(maximum) and maximum > 0:
            rescue_threshold = rescue_fraction * maximum
            mask = probability >= rescue_threshold
            rescue = "relative"
    if not mask.any():
        finite = np.nan_to_num(probability, nan=-np.inf)
        mask.flat[int(np.argmax(finite))] = True
        rescue = "peak"
    return mask, rescue, rescue_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/input")
    parser.add_argument("--output", default="/output")
    parser.add_argument(
        "--classifier-weights",
        nargs="+",
        default=[
            "/workspace/weights/classifier_1.pt",
            "/workspace/weights/classifier_2.pt",
            "/workspace/weights/classifier_3.pt",
        ],
    )
    parser.add_argument(
        "--unet-weights", default="/workspace/weights/unet.pt"
    )
    parser.add_argument(
        "--mask2former-weights",
        default="/workspace/weights/mask2former.pt",
    )
    parser.add_argument(
        "--window-model", default="/workspace/weights/window_normalizer.joblib"
    )
    parser.add_argument(
        "--runtime-config", default="/workspace/weights/runtime_config.json"
    )
    # The 1024 U-Net has high-resolution skip tensors. Four matches the
    # validated per-device micro-batch and is conservative for organizer GPUs.
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    case_directories = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not case_directories:
        raise RuntimeError(f"no case directories found under {input_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = json.loads(Path(args.runtime_config).read_text())
    if runtime.get("classification_probability_reduction") != "mean":
        raise RuntimeError(
            "unsupported classifier probability reduction: "
            f"{runtime.get('classification_probability_reduction')}"
        )
    input_size = int(runtime["input_size"])
    classification_threshold = float(runtime["classification_threshold"])
    unet_boundary = float(runtime["unet_boundary_probability"])
    mask2former_boundary = float(
        runtime["mask2former_boundary_probability"]
    )
    unet_alpha = float(runtime["unet_ensemble_weight"])
    ensemble_threshold = float(runtime["ensemble_threshold_probability"])
    for name, value in {
        "UNet boundary": unet_boundary,
        "Mask2Former boundary": mask2former_boundary,
        "ensemble threshold": ensemble_threshold,
    }.items():
        if not 0 < value < 1:
            raise RuntimeError(f"invalid {name}: {value}")
    if not 0 <= unet_alpha <= 1:
        raise RuntimeError(f"invalid UNet ensemble weight: {unet_alpha}")
    rescue_fraction = float(runtime["relative_rescue_fraction"])
    classifiers, unet, mask2former = load_models(
        args.classifier_weights,
        args.unet_weights,
        args.mask2former_weights,
        device,
    )
    normalizer = XGBoostWindowNormalizer(args.window_model)
    prepared = []
    for case_directory in case_directories:
        dicom_path = find_dicom(case_directory)
        dataset = pydicom.dcmread(dicom_path)
        modality = dicom_modality(dataset)
        reference = sitk.ReadImage(str(dicom_path))
        image, mask2former_image, geometry = prepare_input(
            dataset, normalizer, runtime
        )
        prepared.append(
            (
                case_directory.name,
                reference,
                image,
                mask2former_image,
                geometry,
                modality,
            )
        )

    rows = []
    relative_rescues = 0
    peak_rescues = 0
    modality_counts: dict[str, int] = {}
    for offset in range(0, len(prepared), args.batch_size):
        batch = prepared[offset : offset + args.batch_size]
        images = torch.stack([item[2] for item in batch])
        mask2former_images = torch.stack([item[3] for item in batch])
        class_probabilities = classify(
            classifiers,
            images,
            device,
            float(runtime["classification_logit_bias"]),
        )
        positive_indices = np.flatnonzero(
            class_probabilities >= classification_threshold
        )
        crop_probabilities = {}
        if len(positive_indices):
            selected = images[torch.as_tensor(positive_indices, dtype=torch.long)]
            selected_mask2former = mask2former_images[
                torch.as_tensor(positive_indices, dtype=torch.long)
            ]
            unet_probabilities = segment(unet, selected, device)
            mask2former_probabilities = segment(
                mask2former, selected_mask2former, device
            )
            crop_probabilities = {
                int(index): (unet_probability, mask2former_probability)
                for index, unet_probability, mask2former_probability in zip(
                    positive_indices,
                    unet_probabilities,
                    mask2former_probabilities,
                )
            }

        for index, item in enumerate(batch):
            case_id, reference, _, _, geometry, modality = item
            modality_counts[modality or "<MISSING>"] = (
                modality_counts.get(modality or "<MISSING>", 0) + 1
            )
            if index not in crop_probabilities:
                original_height, original_width = geometry["original_size"]
                mask_2d = np.zeros((original_height, original_width), dtype=bool)
            else:
                unet_crop, mask2former_crop = crop_probabilities[index]
                unet_probability = restore_probability(
                    unet_crop, geometry, input_size
                )
                mask2former_probability = restore_probability(
                    mask2former_crop, geometry, input_size
                )
                probability = centered_logit_ensemble(
                    unet_probability,
                    mask2former_probability,
                    unet_boundary,
                    mask2former_boundary,
                    unet_alpha,
                )
                mask_2d, rescue, _ = positive_mask(
                    probability, ensemble_threshold, rescue_fraction
                )
                relative_rescues += int(rescue == "relative")
                peak_rescues += int(rescue == "peak")
            cavity = int(mask_2d.any())
            mask = mask_for_reference(mask_2d, reference)
            prediction = sitk.GetImageFromArray(mask.astype(np.uint8))
            prediction.CopyInformation(reference)
            sitk.WriteImage(prediction, str(output_root / f"{case_id}.nii.gz"))
            rows.append((case_id, cavity))

    with (output_root / "prediction.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["our_id", "cavity"])
        writer.writerows(rows)
    print(
        f"completed {len(rows)} cases on {device}; "
        f"cavity-positive={sum(value for _, value in rows)}; "
        f"modalities={modality_counts}; "
        f"relative-rescues={relative_rescues}; peak-rescues={peak_rescues}",
        flush=True,
    )


if __name__ == "__main__":
    main()

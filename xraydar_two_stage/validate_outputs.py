"""Validate Task 1 CSV/NIfTI outputs against mounted DICOM case folders."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def reference_path(case_directory: Path) -> Path:
    files = sorted(case_directory.glob("*.dcm"))
    if not files:
        files = sorted(path for path in case_directory.iterdir() if path.is_file())
    if not files:
        raise AssertionError(f"no input file in {case_directory}")
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_directories = sorted(path for path in args.input.iterdir() if path.is_dir())
    expected_ids = [path.name for path in case_directories]
    csv_path = args.output / "prediction.csv"
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["our_id", "cavity"]:
            raise AssertionError(f"invalid CSV fields: {reader.fieldnames}")
        rows = list(reader)
    if len({row["our_id"] for row in rows}) != len(rows):
        raise AssertionError("duplicate our_id in prediction.csv")
    predictions = {row["our_id"]: row["cavity"] for row in rows}
    if sorted(predictions) != expected_ids:
        raise AssertionError(
            f"CSV/input ID mismatch: expected={len(expected_ids)} "
            f"got={len(predictions)}"
        )

    positives = 0
    for case_directory in case_directories:
        case_id = case_directory.name
        value = predictions[case_id]
        if value not in {"0", "1"}:
            raise AssertionError(f"invalid cavity value for {case_id}: {value}")
        mask_path = args.output / f"{case_id}.nii.gz"
        if not mask_path.is_file():
            raise AssertionError(f"missing mask: {mask_path}")
        reference = sitk.ReadImage(str(reference_path(case_directory)))
        mask = sitk.ReadImage(str(mask_path))
        if mask.GetSize() != reference.GetSize():
            raise AssertionError(f"size mismatch for {case_id}")
        for field, actual, expected in (
            ("spacing", mask.GetSpacing(), reference.GetSpacing()),
            ("origin", mask.GetOrigin(), reference.GetOrigin()),
            ("direction", mask.GetDirection(), reference.GetDirection()),
        ):
            if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise AssertionError(f"{field} mismatch for {case_id}")
        if mask.GetPixelID() != sitk.sitkUInt8:
            raise AssertionError(f"mask is not uint8 for {case_id}")
        array = sitk.GetArrayFromImage(mask)
        if not set(np.unique(array)).issubset({0, 1}):
            raise AssertionError(f"mask is not binary for {case_id}")
        present = int(array.any())
        if present != int(value):
            raise AssertionError(f"CSV/mask conflict for {case_id}")
        positives += present

    expected_files = {"prediction.csv"} | {
        f"{case_id}.nii.gz" for case_id in expected_ids
    }
    actual_files = {path.name for path in args.output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise AssertionError(
            f"unexpected/missing output files: expected={expected_files} "
            f"actual={actual_files}"
        )
    print(
        f"valid Task 1 output: cases={len(expected_ids)} positives={positives}",
        flush=True,
    )


if __name__ == "__main__":
    main()

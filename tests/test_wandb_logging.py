import sys
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import fit, validate
from utils import DiceCELoss


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        mask = torch.zeros(1, 8, 8, dtype=torch.long)
        if index % 2:
            mask[:, 2:4, 2:4] = 1
        return {
            "image": torch.zeros(1, 8, 8),
            "mask": mask,
            "cls": torch.tensor([float(index % 2)]),
            "id": str(index),
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.segment = torch.nn.Conv2d(1, 2, 1)
        self.classify = torch.nn.Linear(2, 1)
        self.return_cls = True

    def forward(self, images):
        segmentation = self.segment(images)
        classification = self.classify(segmentation.mean((2, 3)))
        return segmentation, classification


class NativeTinyDataset(Dataset):
    def __len__(self):
        return 111

    def __getitem__(self, index):
        mask = torch.zeros(1, 8, 8, dtype=torch.long)
        if index == 0:
            mask[:, 2:4, 2:4] = 1
        return {
            "image": torch.zeros(1, 8, 8),
            "mask": mask.clone(),
            "native_mask": mask,
            "native_shape": torch.tensor([8, 8]),
            "crop_shape": torch.tensor([8, 8]),
            "pad_info": torch.tensor([0, 0, 8, 8]),
            "cls": torch.tensor([float(index == 0)]),
            "id": str(index),
        }


class FakeRun:
    def __init__(self):
        self.rows = []
        self.summary = {}

    def log(self, row):
        self.rows.append(row)


def test_fit_logs_train_validation_and_epoch_metrics(tmp_path: Path):
    run = FakeRun()
    loader = DataLoader(TinyDataset(), batch_size=2)
    progress = []

    fit(
        TinyModel(),
        loader,
        loader,
        torch.device("cpu"),
        max_epochs=1,
        lambda_cls=0.1,
        ckpt_path=str(tmp_path / "best.pth"),
        optimizer_name="sgd",
        wandb_run=run,
        progress_callback=progress.append,
    )

    assert sum("train/batch_loss" in row for row in run.rows) == 2
    assert sum("validation/batch_loss" in row for row in run.rows) == 2
    epoch_rows = [row for row in run.rows if "epoch/weighted_composite" in row]
    assert len(epoch_rows) == 1
    assert epoch_rows[0]["epoch/validation_case_count"] == 4
    assert epoch_rows[0]["epoch/optimizer_steps"] == 2
    assert epoch_rows[0]["epoch/completed_train_steps"] == 2
    assert run.summary["best/epoch"] == 1
    assert [row["phase"] for row in progress] == [
        "scientific_train",
        "scientific_train",
        "scientific_validation",
        "scientific_validation",
    ]


def test_fit_accumulates_exact_effective_batches_and_logs_loss_components(
    tmp_path: Path,
):
    class TwentyCases(TinyDataset):
        def __len__(self):
            return 20

    run = FakeRun()
    loader = DataLoader(TwentyCases(), batch_size=1)
    details = cast(
        dict[str, object],
        fit(
            TinyModel(),
            loader,
            loader,
            torch.device("cpu"),
            max_epochs=1,
            lambda_cls=0.5,
            ckpt_path=str(tmp_path / "best-accumulated.pth"),
            optimizer_name="adamw",
            wandb_run=run,
            return_details=True,
            gradient_accumulation_steps=8,
        ),
    )

    train_rows = [row for row in run.rows if "train/total_loss" in row]
    assert len(train_rows) == 16
    assert all("train/segmentation_loss" in row for row in train_rows)
    assert all("train/classification_loss" in row for row in train_rows)
    assert all("train/learning_rate_group_0" in row for row in train_rows)
    assert sum(row["train/optimizer_step_completed"] for row in train_rows) == 2
    assert details["completed_train_steps"] == 2
    assert details["completed_micro_steps"] == 16
    assert details["gradient_accumulation_steps"] == 8
    epoch_row = next(row for row in run.rows if "epoch/weighted_composite" in row)
    assert epoch_row["epoch/micro_steps"] == 16
    assert epoch_row["epoch/optimizer_steps"] == 2


def test_native_combo_validation_covers_all_111_cases_once():
    loader = DataLoader(NativeTinyDataset(), batch_size=1, shuffle=False)
    expected_ids = [str(index) for index in range(111)]
    _, _, _, native = validate(
        TinyModel(),
        loader,
        DiceCELoss(),
        lambda_cls=0.5,
        device=torch.device("cpu"),
        use_amp=False,
        native_combo_expected_ids=expected_ids,
    )

    assert native is not None
    assert native["coverage"] == {
        "expected": 111,
        "observed": 111,
        "unique": 111,
        "missing": [],
        "unexpected": [],
        "duplicates": 0,
    }

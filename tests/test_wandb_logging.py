import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import fit


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


class FakeRun:
    def __init__(self):
        self.rows = []
        self.summary = {}

    def log(self, row):
        self.rows.append(row)


def test_fit_logs_train_validation_and_epoch_metrics(tmp_path: Path):
    run = FakeRun()
    loader = DataLoader(TinyDataset(), batch_size=2)

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
    )

    assert sum("train/batch_loss" in row for row in run.rows) == 2
    assert sum("validation/batch_loss" in row for row in run.rows) == 2
    epoch_rows = [row for row in run.rows if "epoch/weighted_composite" in row]
    assert len(epoch_rows) == 1
    assert epoch_rows[0]["epoch/validation_case_count"] == 4
    assert run.summary["best/epoch"] == 1

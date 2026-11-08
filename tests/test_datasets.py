"""Unit tests for dataset manifest loaders."""

import pytest
import torch
from src.datasets.drive_dataset import DRIVEManifestDataset
from src.datasets.rite_dataset import RITEManifestDataset
from src.datasets.idrid_dataset import IDRiDLesionDataset
from src.datasets.aptos_dataset import APTOSDataset
from src.datasets.odir_dataset import ODIRDataset
from src.datasets.optic_disc_cup_dataset import REFUGE2Dataset


def test_drive_manifest_dataset():
    ds = DRIVEManifestDataset(manifest_path="manifests/vessel/drive_train.csv", image_size=128, limit=2)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (1, 128, 128)
    assert item["fov"].shape == (1, 128, 128)


def test_rite_manifest_dataset():
    ds = RITEManifestDataset(manifest_path="manifests/artery_vein/rite_train.csv", image_size=128, limit=2)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (128, 128)
    assert item["mask"].dtype == torch.int64


def test_idrid_lesion_dataset():
    ds = IDRiDLesionDataset(manifest_path="manifests/dr_lesions/idrid_lesions_train.csv", image_size=128, limit=2)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (4, 128, 128)  # 4 lesion channels


def test_aptos_dataset():
    ds = APTOSDataset(manifest_path="manifests/dr_grading/aptos_train.csv", image_size=128, limit=2)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert isinstance(item["label"].item(), int)


def test_odir_dataset():
    ds = ODIRDataset(manifest_path="manifests/multi_disease/odir_train.csv", image_size=128, limit=2)
    assert len(ds) > 0
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["labels"].shape == (8,)  # 8 disease targets


def test_refuge2_dataset():
    ds = REFUGE2Dataset(manifest_path="manifests/optic_disc_cup/refuge2_train.csv", image_size=128, limit=2)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (2, 128, 128)  # 2 channels: OD and Cup
    assert "cdr_ref" in item
    assert "diagnosis" in item
    assert "fovea" in item
    assert item["fovea"].shape == (2,)
    assert item["image_id"] != ""



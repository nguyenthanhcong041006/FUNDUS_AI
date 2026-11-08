"""Standardized PyTorch Dataset loaders for FUNDUS AI."""

from .drive_dataset import DRIVEManifestDataset
from .rite_dataset import RITEManifestDataset
from .idrid_dataset import IDRiDLesionDataset
from .aptos_dataset import APTOSDataset
from .odir_dataset import ODIRDataset, ODIR_DISEASE_COLUMNS
from .optic_disc_cup_dataset import REFUGE2Dataset
from .transforms import JointFundusAugmentation

__all__ = [
    "DRIVEManifestDataset",
    "RITEManifestDataset",
    "IDRiDLesionDataset",
    "APTOSDataset",
    "ODIRDataset",
    "ODIR_DISEASE_COLUMNS",
    "REFUGE2Dataset",
    "JointFundusAugmentation",
]

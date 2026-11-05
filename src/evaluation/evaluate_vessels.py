"""Vessel model evaluation script supporting Internal (DRIVE) and External (STARE) validation."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.drive_dataset import DRIVEManifestDataset
from src.models.unet import UNet
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics


def evaluate_vessels_on_manifest(
    model_checkpoint: str = "checkpoints/vessels/best_vessel_model.pth",
    manifest_path: str = "manifests/vessel/drive_test.csv",
    image_size: int = 512,
    threshold: float = 0.5,
    save_csv: str | None = None
):
    print("===================================================================")
    print("[*] FUNDUS AI - Retinal Vessel Segmentation Evaluation")
    print(f"    Model checkpoint: {model_checkpoint}")
    print(f"    Target manifest:  {manifest_path}")
    print("===================================================================\n")

    p_ckpt = Path(model_checkpoint)
    if not p_ckpt.exists() and (PROJECT_ROOT / p_ckpt).exists():
        p_ckpt = PROJECT_ROOT / p_ckpt
    if not p_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(str(p_ckpt), map_location=device))
    model.eval()

    dataset = DRIVEManifestDataset(manifest_path=manifest_path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    records = []
    print(f"[*] Evaluating {len(dataset)} images on device: {device}...")

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            fov = batch["fov"].to(device)
            img_id = batch["image_id"][0]

            logits = model(image)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

            pred_bin = (prob > threshold).astype(np.uint8) * 255
            gt_bin = (target.squeeze().cpu().numpy() > 0).astype(np.uint8) * 255
            fov_bin = (fov.squeeze().cpu().numpy() > 0).astype(np.uint8) * 255

            m = calculate_segmentation_metrics(pred_bin, gt_bin, fov_mask=fov_bin)
            m["image_id"] = img_id
            records.append(m)

    df = pd.DataFrame(records)

    dice_mean = df["dice"].mean()
    iou_mean = df["iou"].mean()
    sens_mean = df["recall_sensitivity"].mean()
    spec_mean = df["specificity"].mean()

    print("\n----------------- SUMMARY METRICS -----------------")
    print(f"  Dataset:      {Path(manifest_path).stem}")
    print(f"  Total Images: {len(df)}")
    print(f"  Dice Score:   {dice_mean:.4f}  (std: {df['dice'].std():.4f})")
    print(f"  IoU Score:    {iou_mean:.4f}  (std: {df['iou'].std():.4f})")
    print(f"  Sensitivity:  {sens_mean:.4f}  (std: {df['recall_sensitivity'].std():.4f})")
    print(f"  Specificity:  {spec_mean:.4f}  (std: {df['specificity'].std():.4f})")
    print("---------------------------------------------------\n")

    if save_csv:
        out_f = Path(save_csv)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_f, index=False)
        print(f"[+] Individual image metrics saved to: {out_f.resolve()}")

    return {
        "dice": dice_mean,
        "iou": iou_mean,
        "sensitivity": sens_mean,
        "specificity": spec_mean
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vessel Model on Official Test or External Set")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vessels/best_vessel_model.pth")
    parser.add_argument("--manifest", type=str, default="manifests/vessel/drive_test.csv",
                        help="Manifest path (e.g. manifests/vessel/drive_test.csv or future STARE manifest)")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_csv", type=str, default="outputs/vessel_evaluation_results.csv")

    args = parser.parse_args()
    evaluate_vessels_on_manifest(
        model_checkpoint=args.checkpoint,
        manifest_path=args.manifest,
        image_size=args.image_size,
        threshold=args.threshold,
        save_csv=args.save_csv
    )


if __name__ == "__main__":
    main()

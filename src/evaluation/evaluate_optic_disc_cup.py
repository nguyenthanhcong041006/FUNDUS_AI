"""Optic Disc & Cup Evaluation Script supporting Internal (Drishti-GS1) and External (PAPILA) Validation.

Computes:
  1. Segmentation Metrics (Dice, IoU, Sensitivity, Specificity) for Optic Disc and Cup
  2. Vertical Cup-to-Disc Ratio (vCDR) accuracy against clinical expert ground truth
  3. External Glaucoma Screening evaluation (ROC-AUC, Sensitivity, Specificity) on PAPILA
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch.utils.data import DataLoader

from src.datasets.optic_disc_cup_dataset import DrishtiDataset, PAPILADataset
from src.models.unet import UNet
from src.evaluation.segmentation_metrics import calculate_segmentation_metrics
from src.measurements.optic_disc_metrics import calculate_disc_cup_metrics
from src.datasets.utils import resolve_path


def evaluate_drishti_internal(
    model: torch.nn.Module,
    manifest_path: str,
    device: torch.device,
    image_size: int = 512,
    threshold: float = 0.5,
    limit: int | None = None
) -> pd.DataFrame:
    """Internal validation on official Drishti-GS1 test split."""
    print("-------------------------------------------------------------------")
    print(f"[*] 1. Internal Validation on Drishti-GS1 ({manifest_path})")
    print("-------------------------------------------------------------------")

    dataset = DrishtiDataset(manifest_path=manifest_path, image_size=image_size, limit=limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    records = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            cdr_ref = float(batch["cdr_ref"].item())
            image_id = batch["image_id"][0]

            logits = model(image)
            probs = torch.sigmoid(logits)[0]

            # Channel 0: Optic Disc
            pred_od = (probs[0].cpu().numpy() > threshold).astype(np.uint8) * 255
            gt_od = (target[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
            m_od = calculate_segmentation_metrics(pred_od, gt_od)

            # Channel 1: Optic Cup
            pred_cup = (probs[1].cpu().numpy() > threshold).astype(np.uint8) * 255
            gt_cup = (target[0, 1].cpu().numpy() > 0).astype(np.uint8) * 255
            m_cup = calculate_segmentation_metrics(pred_cup, gt_cup)

            # Morphological vCDR
            morph = calculate_disc_cup_metrics(pred_od, pred_cup)
            pred_vcdr = morph["optic_cup"]["vertical_cdr"]
            cdr_err = abs(pred_vcdr - cdr_ref) if cdr_ref > 0 else None

            records.append({
                "image_id": image_id,
                "dataset": "Drishti-GS1",
                "dice_od": round(m_od["dice"], 4),
                "iou_od": round(m_od["iou"], 4),
                "sens_od": round(m_od["recall_sensitivity"], 4),
                "dice_cup": round(m_cup["dice"], 4),
                "iou_cup": round(m_cup["iou"], 4),
                "sens_cup": round(m_cup["recall_sensitivity"], 4),
                "pred_vcdr": pred_vcdr,
                "ref_cdr": cdr_ref if cdr_ref > 0 else np.nan,
                "cdr_mae": round(cdr_err, 4) if cdr_err is not None else np.nan
            })

    df = pd.DataFrame(records)
    print(f"  [+] Images evaluated: {len(df)}")
    print(f"  [+] Optic Disc - Mean Dice: {df['dice_od'].mean():.4f} | Mean IoU: {df['iou_od'].mean():.4f}")
    print(f"  [+] Optic Cup  - Mean Dice: {df['dice_cup'].mean():.4f} | Mean IoU: {df['iou_cup'].mean():.4f}")
    valid_cdr = df["cdr_mae"].dropna()
    if len(valid_cdr) > 0:
        print(f"  [+] vCDR vs Expert GT - Mean Absolute Error (MAE): {valid_cdr.mean():.4f}")
    print("-------------------------------------------------------------------\n")
    return df


def evaluate_papila_external(
    model: torch.nn.Module,
    manifest_path: str,
    device: torch.device,
    image_size: int = 512,
    threshold: float = 0.5,
    limit: int | None = None
) -> pd.DataFrame:
    """External validation on independent PAPILA dataset (generalization & glaucoma screening)."""
    print("-------------------------------------------------------------------")
    print(f"[*] 2. External Validation on PAPILA ({manifest_path})")
    print("-------------------------------------------------------------------")

    dataset = PAPILADataset(manifest_path=manifest_path, image_size=image_size, limit=limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    records = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            diagnosis = int(batch["diagnosis"].item())
            image_id = batch["image_id"][0]
            patient_id = batch["patient_id"][0]
            eye_side = batch["eye_side"][0]

            logits = model(image)
            probs = torch.sigmoid(logits)[0]

            # Channel 0: Optic Disc
            pred_od = (probs[0].cpu().numpy() > threshold).astype(np.uint8) * 255
            gt_od = (target[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
            m_od = calculate_segmentation_metrics(pred_od, gt_od)

            # Channel 1: Optic Cup
            pred_cup = (probs[1].cpu().numpy() > threshold).astype(np.uint8) * 255
            gt_cup = (target[0, 1].cpu().numpy() > 0).astype(np.uint8) * 255
            m_cup = calculate_segmentation_metrics(pred_cup, gt_cup)

            # Morphological vCDR
            morph = calculate_disc_cup_metrics(pred_od, pred_cup)
            pred_vcdr = morph["optic_cup"]["vertical_cdr"]

            records.append({
                "image_id": image_id,
                "patient_id": patient_id,
                "eye_side": eye_side,
                "dataset": "PAPILA",
                "dice_od": round(m_od["dice"], 4),
                "iou_od": round(m_od["iou"], 4),
                "dice_cup": round(m_cup["dice"], 4),
                "iou_cup": round(m_cup["iou"], 4),
                "pred_vcdr": pred_vcdr,
                "clinical_diagnosis": diagnosis  # 0: Normal, 1: Glaucoma, 2: Suspicious
            })

    df = pd.DataFrame(records)
    print(f"  [+] Images evaluated: {len(df)}")
    print(f"  [+] Optic Disc - External Mean Dice: {df['dice_od'].mean():.4f} | Mean IoU: {df['iou_od'].mean():.4f}")
    print(f"  [+] Optic Cup  - External Mean Dice: {df['dice_cup'].mean():.4f} | Mean IoU: {df['iou_cup'].mean():.4f}")

    # Glaucoma screening evaluation (Normal=0 vs Glaucoma=1, excluding/including suspicious)
    binary_df = df[df["clinical_diagnosis"].isin([0, 1])].copy()
    if len(binary_df) > 0 and len(binary_df["clinical_diagnosis"].unique()) > 1:
        y_true = binary_df["clinical_diagnosis"].values
        y_scores = binary_df["pred_vcdr"].values

        try:
            auc = roc_auc_score(y_true, y_scores)
            # Clinical cutoff at vCDR >= 0.65
            y_pred_binary = (y_scores >= 0.65).astype(int)
            acc = accuracy_score(y_true, y_pred_binary)
            tp = int(((y_pred_binary == 1) & (y_true == 1)).sum())
            fn = int(((y_pred_binary == 0) & (y_true == 1)).sum())
            tn = int(((y_pred_binary == 0) & (y_true == 0)).sum())
            fp = int(((y_pred_binary == 1) & (y_true == 0)).sum())

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)

            print(f"\n  [*] Glaucoma Screening from Morphology (vCDR >= 0.65):")
            print(f"      ROC-AUC:     {auc:.4f}")
            print(f"      Sensitivity: {sens:.4f}")
            print(f"      Specificity: {spec:.4f}")
            print(f"      Accuracy:    {acc:.4f}")
        except Exception as e:
            print(f"  [!] Could not compute ROC-AUC: {e}")

    print("-------------------------------------------------------------------\n")
    return df


def main():
    parser = argparse.ArgumentParser(description="Optic Disc & Cup Comprehensive Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/optic_disc_cup/best_od_oc_model.pth")
    parser.add_argument("--drishti_manifest", type=str, default="manifests/optic_disc_cup/drishti_test.csv")
    parser.add_argument("--papila_manifest", type=str, default="manifests/optic_disc_cup/papila_external_test.csv")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--skip_external", action="store_true")

    args = parser.parse_args()

    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model from {ckpt_path} on {device}...")
    model = UNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Internal Validation (Drishti-GS1)
    df_drishti = evaluate_drishti_internal(
        model=model,
        manifest_path=args.drishti_manifest,
        device=device,
        image_size=args.image_size,
        threshold=args.threshold,
        limit=args.limit
    )
    df_drishti.to_csv(out_dir / "optic_disc_cup_internal_evaluation.csv", index=False)
    print(f"[+] Saved Internal Evaluation CSV: {out_dir / 'optic_disc_cup_internal_evaluation.csv'}")

    # 2. External Validation (PAPILA)
    if not args.skip_external:
        df_papila = evaluate_papila_external(
            model=model,
            manifest_path=args.papila_manifest,
            device=device,
            image_size=args.image_size,
            threshold=args.threshold,
            limit=args.limit
        )
        df_papila.to_csv(out_dir / "optic_disc_cup_external_evaluation.csv", index=False)
        print(f"[+] Saved External Evaluation CSV: {out_dir / 'optic_disc_cup_external_evaluation.csv'}")

    print("\n[OK] Comprehensive Optic Disc/Cup Evaluation Complete!")


if __name__ == "__main__":
    main()


"""Manifest generator script for all FUNDUS AI tasks.

Generates standardized CSV manifest files according to Section 4 (Phase 0)
and Section 7 of FUNDUS_AI_Dataset_Training_Plan_2.md.
Paths are formatted as relative POSIX paths (e.g. datasets/raw/...) for
portability across environments (Windows/Linux/Colab/Kaggle).
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def to_rel_posix(path, project_root: Path = PROJECT_ROOT) -> str:
    """Converts a Path or str to a POSIX relative path from project root."""
    if not path or (isinstance(path, float) and np.isnan(path)):
        return ""
    p = Path(path).resolve()
    try:
        return p.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def save_csv(df: pd.DataFrame, path: Path):
    """Safely saves DataFrame to CSV using standard python file handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)


def build_drive_manifests(raw_root: Path, out_dir: Path):
    """Builds vessel segmentation manifests for DRIVE dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_root = raw_root / "drive"

    for split in ["training", "test"]:
        split_dir = drive_root / split
        img_dir = split_dir / "images"
        mask_dir = split_dir / "1st_manual"
        fov_dir = split_dir / "mask"

        records = []
        if img_dir.exists():
            for img_file in sorted(img_dir.glob("*.tif")):
                base_num = img_file.stem.split("_")[0]
                mask_matches = list(mask_dir.glob(f"{base_num}_*.*"))
                fov_matches = list(fov_dir.glob(f"{base_num}_*.*"))

                records.append({
                    "image_id": img_file.stem,
                    "patient_id": f"drive_{base_num}",
                    "image_path": to_rel_posix(img_file),
                    "mask_path": to_rel_posix(mask_matches[0]) if mask_matches else "",
                    "fov_path": to_rel_posix(fov_matches[0]) if fov_matches else "",
                    "split": "train" if split == "training" else "test",
                    "dataset": "DRIVE",
                    "task": "vessel_segmentation"
                })

        df = pd.DataFrame(records)
        target_name = "drive_train.csv" if split == "training" else "drive_test.csv"
        save_csv(df, out_dir / target_name)
        print(f"[+] DRIVE {split}: {len(df)} records -> {out_dir / target_name}")


def build_rite_manifests(raw_root: Path, out_dir: Path):
    """Builds artery/vein segmentation manifests for RITE dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rite_root = raw_root / "rite"

    for split in ["training", "test"]:
        split_dir = rite_root / split
        img_dir = split_dir / "images"
        av_dir = split_dir / "av"
        vessel_dir = split_dir / "vessel"

        records = []
        if img_dir.exists():
            for img_file in sorted(img_dir.glob("*.tif")):
                base_num = img_file.stem.split("_")[0]
                av_matches = list(av_dir.glob(f"{base_num}_*.png"))
                vessel_matches = list(vessel_dir.glob(f"{base_num}_*.png"))

                records.append({
                    "image_id": img_file.stem,
                    "patient_id": f"rite_{base_num}",
                    "image_path": to_rel_posix(img_file),
                    "av_mask_path": to_rel_posix(av_matches[0]) if av_matches else "",
                    "vessel_mask_path": to_rel_posix(vessel_matches[0]) if vessel_matches else "",
                    "split": "train" if split == "training" else "test",
                    "dataset": "RITE",
                    "task": "artery_vein_segmentation"
                })

        df = pd.DataFrame(records)
        target_name = "rite_train.csv" if split == "training" else "rite_test.csv"
        save_csv(df, out_dir / target_name)
        print(f"[+] RITE {split}: {len(df)} records -> {out_dir / target_name}")


def build_idrid_lesion_manifests(raw_root: Path, out_dir: Path):
    """Builds multi-lesion segmentation manifests for IDRiD dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_root = raw_root / "idrid" / "A. Segmentation"
    img_root = seg_root / "1. Original Images"
    gt_root = seg_root / "2. All Segmentation Groundtruths"

    for split, split_folder in [("train", "a. Training Set"), ("test", "b. Testing Set")]:
        img_dir = img_root / split_folder
        gt_dir = gt_root / split_folder

        records = []
        if img_dir.exists() and gt_dir.exists():
            for img_file in sorted(img_dir.glob("*.jpg")):
                img_name = img_file.stem

                def find_mask(subfolder: str):
                    folder = gt_dir / subfolder
                    if folder.exists():
                        m = list(folder.glob(f"{img_name}*"))
                        return to_rel_posix(m[0]) if m else ""
                    return ""

                records.append({
                    "image_id": img_name,
                    "image_path": to_rel_posix(img_file),
                    "ma_mask_path": find_mask("1. Microaneurysms"),
                    "he_mask_path": find_mask("2. Haemorrhages"),
                    "ex_mask_path": find_mask("3. Hard Exudates"),
                    "se_mask_path": find_mask("4. Soft Exudates"),
                    "od_mask_path": find_mask("5. Optic Disc"),
                    "split": split,
                    "dataset": "IDRiD",
                    "task": "dr_lesion_segmentation"
                })

        df = pd.DataFrame(records)
        save_csv(df, out_dir / f"idrid_lesions_{split}.csv")
        print(f"[+] IDRiD Lesions {split}: {len(df)} records -> {out_dir / f'idrid_lesions_{split}.csv'}")


def build_aptos_manifests(raw_root: Path, out_dir: Path):
    """Builds DR severity grading manifests for APTOS 2019 dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aptos_root = raw_root / "aptos"

    train_csv = aptos_root / "train.csv"
    train_dir = aptos_root / "train_images"

    if train_csv.exists() and train_dir.exists():
        df_train = pd.read_csv(train_csv)
        df_train["image_path"] = df_train["id_code"].apply(lambda x: to_rel_posix(train_dir / f"{x}.png"))
        df_train["split"] = "train"
        df_train["dataset"] = "APTOS_2019"
        df_train["task"] = "dr_grading"
        save_csv(df_train, out_dir / "aptos_train.csv")
        print(f"[+] APTOS Train: {len(df_train)} records -> {out_dir / 'aptos_train.csv'}")

    test_csv = aptos_root / "test.csv"
    test_dir = aptos_root / "test_images"
    if test_csv.exists() and test_dir.exists():
        df_test = pd.read_csv(test_csv)
        df_test["image_path"] = df_test["id_code"].apply(lambda x: to_rel_posix(test_dir / f"{x}.png"))
        df_test["split"] = "test"
        df_test["dataset"] = "APTOS_2019"
        df_test["task"] = "dr_grading"
        save_csv(df_test, out_dir / "aptos_test.csv")
        print(f"[+] APTOS Test: {len(df_test)} records -> {out_dir / 'aptos_test.csv'}")


def build_odir_manifests(raw_root: Path, out_dir: Path, val_ratio: float = 0.15, seed: int = 42):
    """Builds patient-independent multi-disease classification manifests for ODIR-5K."""
    out_dir.mkdir(parents=True, exist_ok=True)
    odir_root = raw_root / "odir"

    ann_file = odir_root / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
    train_img_dir = odir_root / "ODIR-5K_Training_Dataset"

    if ann_file.exists() and train_img_dir.exists():
        df = pd.read_excel(ann_file)

        # Map image relative paths
        df["left_image_path"] = df["Left-Fundus"].apply(lambda x: to_rel_posix(train_img_dir / str(x)))
        df["right_image_path"] = df["Right-Fundus"].apply(lambda x: to_rel_posix(train_img_dir / str(x)))

        # Patient-level split to prevent eye leakage across train and val
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(df))
        val_count = int(len(df) * val_ratio)

        val_idx = set(shuffled_indices[:val_count])
        df["split"] = ["val" if i in val_idx else "train" for i in range(len(df))]
        df["dataset"] = "ODIR-5K"
        df["task"] = "multi_disease_classification"

        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "val"]

        save_csv(df_train, out_dir / "odir_train.csv")
        save_csv(df_val, out_dir / "odir_val.csv")
        print(f"[+] ODIR-5K Train: {len(df_train)} patients | Val: {len(df_val)} patients (patient-level split)")


def build_drishti_manifests(raw_root: Path, out_dir: Path):
    """Builds Optic Disc and Cup segmentation manifests for Drishti-GS1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    drishti_root = raw_root / "Drishti-GS1"
    files_root = drishti_root / "Drishti-GS1_files"
    diag_file = drishti_root / "Drishti-GS1_diagnosis.xlsx"

    diag_map = {}
    if diag_file.exists():
        df_diag = pd.read_excel(diag_file).dropna(how="all")
        for _, row in df_diag.iterrows():
            f_val = str(row.iloc[1]).replace("'", "").strip()
            if "drishti" in f_val:
                diag_map[f_val] = str(row.iloc[8]).strip()

    for split, folder, gt_sub in [("train", "Training", "GT"), ("test", "Test", "Test_GT")]:
        img_dir = files_root / folder / "Images"
        gt_dir = files_root / folder / gt_sub

        records = []
        if img_dir.exists():
            for img_file in sorted(img_dir.glob("*.png")):
                img_id = img_file.stem
                img_gt_dir = gt_dir / img_id
                soft_dir = img_gt_dir / "SoftMap"

                od_soft = soft_dir / f"{img_id}_ODsegSoftmap.png"
                cup_soft = soft_dir / f"{img_id}_cupsegSoftmap.png"
                cdr_file = img_gt_dir / f"{img_id}_cdrValues.txt"

                cdr_ref = ""
                if cdr_file.exists():
                    try:
                        vals = [float(x) for x in cdr_file.read_text().strip().split()]
                        if vals:
                            cdr_ref = round(float(np.mean(vals)), 4)
                    except Exception:
                        pass

                records.append({
                    "image_id": img_id,
                    "patient_id": img_id,
                    "image_path": to_rel_posix(img_file),
                    "od_mask_path": to_rel_posix(od_soft) if od_soft.exists() else "",
                    "cup_mask_path": to_rel_posix(cup_soft) if cup_soft.exists() else "",
                    "cdr_ref": cdr_ref,
                    "diagnosis": diag_map.get(img_id, ""),
                    "split": split,
                    "dataset": "Drishti-GS1",
                    "task": "optic_disc_cup_segmentation"
                })

        df = pd.DataFrame(records)
        target_name = f"drishti_{split}.csv"
        save_csv(df, out_dir / target_name)
        print(f"[+] Drishti-GS1 {split}: {len(df)} records -> {out_dir / target_name}")


def build_papila_manifests(raw_root: Path, out_dir: Path):
    """Builds Optic Disc and Cup External Validation manifest for PAPILA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    papila_root = raw_root / "PAPILA"
    img_dir = papila_root / "FundusImages"
    cont_dir = papila_root / "ExpertsSegmentations" / "Contours"
    clinical_dir = papila_root / "ClinicalData"

    clinical_map = {}
    for side, file_name in [("OD", "patient_data_od.xlsx"), ("OS", "patient_data_os.xlsx")]:
        fpath = clinical_dir / file_name
        if fpath.exists():
            df_c = pd.read_excel(fpath).iloc[2:]
            for _, row in df_c.iterrows():
                raw_id = str(row.iloc[0]).replace("#", "").strip()
                if raw_id:
                    img_stem = f"RET{raw_id}{side}"
                    clinical_map[img_stem] = {
                        "patient_id": f"RET{raw_id}",
                        "eye_side": side,
                        "diagnosis": row.get("Diagnosis", ""),
                        "age": row.get("Age", ""),
                        "gender": row.get("Gender", ""),
                        "iop": row.get("IOP", "")
                    }

    records = []
    if img_dir.exists():
        for img_file in sorted(img_dir.glob("*.jpg")):
            img_id = img_file.stem
            c_info = clinical_map.get(img_id, {
                "patient_id": img_id[:-2] if img_id.endswith(("OD", "OS")) else img_id,
                "eye_side": img_id[-2:] if img_id.endswith(("OD", "OS")) else "",
                "diagnosis": "",
                "age": "",
                "gender": "",
                "iop": ""
            })

            od_c1 = cont_dir / f"{img_id}_disc_exp1.txt"
            od_c2 = cont_dir / f"{img_id}_disc_exp2.txt"
            cup_c1 = cont_dir / f"{img_id}_cup_exp1.txt"
            cup_c2 = cont_dir / f"{img_id}_cup_exp2.txt"

            records.append({
                "image_id": img_id,
                "patient_id": c_info["patient_id"],
                "eye_side": c_info["eye_side"],
                "image_path": to_rel_posix(img_file),
                "od_contour_exp1": to_rel_posix(od_c1) if od_c1.exists() else "",
                "od_contour_exp2": to_rel_posix(od_c2) if od_c2.exists() else "",
                "cup_contour_exp1": to_rel_posix(cup_c1) if cup_c1.exists() else "",
                "cup_contour_exp2": to_rel_posix(cup_c2) if cup_c2.exists() else "",
                "diagnosis": c_info["diagnosis"],
                "age": c_info["age"],
                "gender": c_info["gender"],
                "iop": c_info["iop"],
                "split": "external_test",
                "dataset": "PAPILA",
                "task": "optic_disc_cup_external_validation"
            })

    df = pd.DataFrame(records)
    save_csv(df, out_dir / "papila_external_test.csv")
    print(f"[+] PAPILA External Test: {len(df)} records -> {out_dir / 'papila_external_test.csv'}")


def main():
    raw_root = PROJECT_ROOT / "datasets" / "raw"
    manifests_root = PROJECT_ROOT / "manifests"

    print("\n=======================================================")
    print("[*] Generating Standard Manifests for all FUNDUS AI Tasks...")
    print("=======================================================\n")

    build_drive_manifests(raw_root, manifests_root / "vessel")
    build_rite_manifests(raw_root, manifests_root / "artery_vein")
    build_idrid_lesion_manifests(raw_root, manifests_root / "dr_lesions")
    build_aptos_manifests(raw_root, manifests_root / "dr_grading")
    build_odir_manifests(raw_root, manifests_root / "multi_disease")
    build_drishti_manifests(raw_root, manifests_root / "optic_disc_cup")
    build_papila_manifests(raw_root, manifests_root / "optic_disc_cup")

    print("\n[OK] All manifest files generated successfully in manifests/")


if __name__ == "__main__":
    main()

"""FUNDUS AI - one-click training orchestrator (launched by train.bat).

Pipeline:
  1. Device info      : Python, PyTorch, CUDA, GPU (name / free VRAM / driver), CPU, RAM, disk.
  2. Preflight checks : packages, training scripts, manifests, every image/mask path in the manifests,
                        optional 1-epoch smoke run (--smoke). Takes a few seconds.
  3. Batch size       : fixed per task, tuned once for this machine (HARDWARE_PROFILE). Only a cheap
                        free-VRAM check is done at start; override with --batch-size.
  4. Training         : runs every task sequentially, streams the live terminal output (tqdm bars),
                        and after every epoch appends a detailed record (time, duration, ETA, device,
                        GPU memory/utilization/temperature, RAM, batch size, all metrics) to the logs.
  5. Summary          : SUMMARY.md per run. Best checkpoints are copied to checkpoints/ (the paths that
                        main.py loads); every file that gets replaced is backed up into the run folder.

Every run writes into runs/<timestamp>/ (git-ignored), so earlier results are never overwritten.

Resume: runs/<timestamp>/run_state.json records the status of every task. Running train.bat again
automatically continues the most recent unfinished run: finished tasks are skipped, and the task that was
interrupted continues from the last completed epoch (the training scripts save <checkpoint>_last.pth after
every epoch, see src/training/resume_utils.py). --new starts a fresh run instead.
"""

import argparse
import codecs
import csv
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = sys.executable
REQUIRED_PACKAGES = ["torch", "torchvision", "numpy", "pandas", "cv2", "PIL", "sklearn",
                     "matplotlib", "tqdm", "scipy", "yaml"]
# Batch sizes in TASKS were measured once on this machine (forward+backward+AdamW step @ 512px,
# 07/11/2026, ~6.9 GB VRAM free with desktop apps open). U-Net speed is the same at batch 2 and 3
# (~8.6 img/s, GPU already saturated) so batch 2 is used; Attention U-Net at batch 3 needs 6.8 GB.
# M1 (REFUGE2) uses batch 4 as in the README: it uses BatchNorm, which is unstable at batch 2.
HARDWARE_PROFILE = "RTX 4060 Laptop 8 GB, 512px"

# ---------------------------------------------------------------------------------------------
# Task registry: what each training script needs + fixed batch size (bs) and its measured VRAM peak.
# ---------------------------------------------------------------------------------------------
TASKS = [
    dict(key="optic_disc_cup", name="M1 Optic Disc & Cup (REFUGE2)",
         module="src.training.train_optic_disc_cup", extra=["--norm", "group", "--patience", "10"],   # REFUGE2 only
         manifests=["manifests/optic_disc_cup/refuge2_train.csv", "manifests/optic_disc_cup/refuge2_val.csv",
                    "manifests/optic_disc_cup/refuge2_test.csv"],
         train_manifest="manifests/optic_disc_cup/refuge2_train.csv", train_factor=1.0,
         model="unet:2", epochs=30, bs=4, peak_gb=6.8,
         checkpoint="checkpoints/optic_disc_cup/best_od_oc_model.pth"),
    dict(key="vessels", name="M2 Vessels (DRIVE)",
         module="src.training.train_vessels", extra=[],
         manifests=["manifests/vessel/drive_train.csv", "manifests/vessel/drive_test.csv"],
         train_manifest="manifests/vessel/drive_train.csv", train_factor=0.8,
         model="unet:1", epochs=50, bs=2, peak_gb=3.6,
         checkpoint="checkpoints/vessels/best_vessel_model.pth"),
    dict(key="artery_vein", name="M3 Artery/Vein (RITE)",
         module="src.training.train_artery_vein", extra=[],
         manifests=["manifests/artery_vein/rite_train.csv", "manifests/artery_vein/rite_test.csv"],
         train_manifest="manifests/artery_vein/rite_train.csv", train_factor=0.8,
         model="unet:3", epochs=50, bs=2, peak_gb=3.6,
         checkpoint="checkpoints/artery_vein/best_av_model.pth"),
    dict(key="lesions", name="M4 DR Lesions (IDRiD)",
         module="src.training.train_lesions", extra=[],
         manifests=["manifests/dr_lesions/idrid_lesions_train.csv", "manifests/dr_lesions/idrid_lesions_test.csv"],
         train_manifest="manifests/dr_lesions/idrid_lesions_train.csv", train_factor=0.8,
         model="attunet:4", epochs=35, bs=2, peak_gb=4.84,
         checkpoint="checkpoints/lesions/best_lesion_model.pth"),
    dict(key="patch_lesions", name="M4-SOTA Patch Lesions (IDRiD 2048px)",
         module="src.training.train_patch_lesions", extra=["--stride", "512", "--patches_per_image", "8"],
         manifests=["manifests/dr_lesions/idrid_lesions_train.csv", "manifests/dr_lesions/idrid_lesions_test.csv"],
         train_manifest="manifests/dr_lesions/idrid_lesions_train.csv", train_factor=0.8,
         model="attunet:4", epochs=35, bs=4, peak_gb=5.5,
         checkpoint="checkpoints/lesions/best_patch_lesion_model.pth",
         optional=True),
    dict(key="dr_grading", name="M5 DR Grading (APTOS)",
         module="src.training.train_dr_grading", extra=[],
         manifests=["manifests/dr_grading/aptos_train.csv"],
         train_manifest="manifests/dr_grading/aptos_train.csv", train_factor=0.8,
         model="effnet:5", epochs=30, bs=12, peak_gb=5.86,
         checkpoint="checkpoints/classification/best_dr_classifier.pth"),
    dict(key="multi_disease", name="M6 Multi-Disease (ODIR-5K)",
         module="src.training.train_multi_disease", extra=[],
         manifests=["manifests/multi_disease/odir_train.csv", "manifests/multi_disease/odir_val.csv"],
         train_manifest="manifests/multi_disease/odir_train.csv", train_factor=2.0,  # 2 eyes / patient
         model="densenet:8", epochs=30, bs=8, peak_gb=5.73,
         checkpoint="checkpoints/classification/best_odir_classifier.pth"),
]

TASK_BY_KEY = {t["key"]: t for t in TASKS}
TASK_BY_NAME = {t["name"]: t for t in TASKS}
DEFAULT_KEYS = [t["key"] for t in TASKS if not t.get("optional")]      # what "all" trains: the 6 modules M1..M6
TASK_ALIASES = {"m1": "optic_disc_cup", "m2": "vessels", "m3": "artery_vein", "m4": "lesions",
                "m4s": "patch_lesions", "m5": "dr_grading", "m6": "multi_disease"}


def resolve_keys(text):
    """'M1,m4 vessels' -> (['optic_disc_cup', 'lesions', 'vessels'], []) ; second item = unknown names.
    'all' means the 6 default modules. Result keeps the order of TASKS."""
    keys, bad = set(), []
    for part in re.split(r"[,\s]+", (text or "").strip()):
        if not part:
            continue
        n = part.lower().replace("-", "_")
        if n == "all":
            keys.update(DEFAULT_KEYS)
            continue
        n = TASK_ALIASES.get(n, n)
        if n in TASK_BY_KEY:
            keys.add(n)
        else:
            bad.append(part)
    return [t["key"] for t in TASKS if t["key"] in keys], bad


def parse_task_opts(text):
    """'--epochs 30 --batch-size 4 --norm group' -> dict(epochs=30, batch=4, image_size=None, extra=['--norm','group'])."""
    toks = [a.strip('"') for a in shlex.split(text, posix=False)]
    names = {"--epochs": "epochs", "--batch-size": "batch", "--batch_size": "batch",
             "--image-size": "image_size", "--image_size": "image_size"}
    opts = dict(epochs=None, batch=None, image_size=None, extra=[])
    i = 0
    while i < len(toks):
        name, eq, val = toks[i].partition("=")
        if name in names:
            if not eq:
                i += 1
                val = toks[i] if i < len(toks) else ""
            if not val.isdigit() or int(val) <= 0:
                raise ValueError(f"{name} can mot so nguyen duong (nhan duoc '{val}')")
            opts[names[name]] = int(val)
        else:
            opts["extra"].append(toks[i])
        i += 1
    return opts

EPOCH_RE = re.compile(r"^Epoch \[(\d+)/(\d+)\]\s*-\s*(.*)$")
METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9 ]*?):\s*(-?[\d.]+)%?")
RESUME_RE = re.compile(r"\[RESUME\] Tiep tuc tu epoch (\d+)/(\d+) \(da xong (\d+) epoch\)")

RUN_STATE = "run_state.json"
# Task statuses that need no more work. PENDING / RUNNING / INTERRUPTED / FAILED are continued on resume;
# BLOCKED (data missing at preflight) is not retried, otherwise a run with missing data could never end.
FINAL_STATUSES = ("DONE", "NO_CHECKPOINT", "BLOCKED")


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_dur(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Tee:
    """Writes to the console and to the run's master log at the same time."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def gpu_stats():
    """Live GPU stats via nvidia-smi (does not touch CUDA in this process)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        used, total, util, temp, power = [x.strip() for x in out.splitlines()[0].split(",")]
        return dict(gpu_mem_used_mb=used, gpu_mem_total_mb=total, gpu_util_pct=util, gpu_temp_c=temp,
                    gpu_power_w=power)
    except Exception:
        return dict(gpu_mem_used_mb="", gpu_mem_total_mb="", gpu_util_pct="", gpu_temp_c="", gpu_power_w="")


def ram_stats():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return dict(ram_used_pct=vm.percent, ram_used_gb=round(vm.used / 1e9, 1))
    except Exception:
        return dict(ram_used_pct="", ram_used_gb="")


def run_helper(args, timeout=900):
    """Runs this script in a fresh process (clean CUDA context) and returns its JSON result."""
    p = subprocess.run([PY, "-u", str(Path(__file__)), *args], capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError((p.stderr or p.stdout)[-1500:])


def parse_dur(text):
    m = re.fullmatch(r"(?:(\d+)h)?(\d+)m(\d+)s", text.strip())
    return (int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))) if m else 0


# ---------------------------------------------------------------------------------------------
# Resume: run state, finding the unfinished run, guarding against two trainings at once
# ---------------------------------------------------------------------------------------------
def state_path_for(ckpt):
    """Same rule as src/training/resume_utils.state_path_for (not imported: that package loads torch)."""
    ckpt = Path(ckpt)
    return ckpt.with_name(ckpt.stem + "_last.pth")


def save_run_state(run_dir, st):
    tmp = run_dir / (RUN_STATE + ".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, run_dir / RUN_STATE)


def load_run_state(run_dir):
    p = run_dir / RUN_STATE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return legacy_run_state(run_dir)


def legacy_run_state(run_dir):
    """Rebuilds the state of a run started by an older train_all.py (before run_state.json existed)
    from its train_log.txt. Returns None for runs that never reached the training step."""
    log_path = run_dir / "train_log.txt"
    if not log_path.is_file():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(l.strip() == "[3/4] TRAIN" for l in lines):
        return None
    keys, started = [], ""
    for l in lines:
        if l.startswith(" Bai toan:") and not keys:
            keys = [k.strip() for k in l.split(":", 1)[1].split(",") if k.strip() in TASK_BY_KEY]
        if "TRAIN TOAN BO |" in l and not started:
            started = l.split("|", 1)[1].strip()
    if not keys:
        return None
    try:
        batch = {k: v["bs"] for k, v in json.loads((run_dir / "batch_sizes.json").read_text(encoding="utf-8")).items()}
    except Exception:
        batch = {}
    try:
        pre = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8")).get("tasks", {})
    except Exception:
        pre = {}

    tasks, epochs, promoted, cur = {}, {}, set(), None
    header_re = re.compile(r"^ >>> .+?\|\s+batch (\d+)\s+\|\s+(\d+) epoch")
    for l in lines:
        m = re.match(r"^\[\d+/\d+\] (.+)$", l)
        if m and m.group(1).strip() in TASK_BY_NAME:
            cur = TASK_BY_NAME[m.group(1).strip()]["key"]
            tasks[cur] = dict(status="INTERRUPTED")
            continue
        if not cur:
            continue
        m = header_re.match(l)
        if m:
            batch[cur], epochs[cur] = int(m.group(1)), int(m.group(2))
        elif l.startswith("   => "):
            parts = [p.strip() for p in l[6:].split(" | ", 3)]
            status = parts[0]
            ckpt = run_dir / "checkpoints" / Path(TASK_BY_KEY[cur]["checkpoint"]).name
            try:
                done_ep = int(parts[1].split("/")[0])
            except Exception:
                done_ep = 0
            tasks[cur] = dict(status=status, result=dict(
                status=status, exit_code=0 if status in ("DONE", "NO_CHECKPOINT") else 1, epochs_done=done_ep,
                epochs_planned=epochs.get(cur, TASK_BY_KEY[cur]["epochs"]), batch_size=batch.get(cur),
                duration=parts[2] if len(parts) > 2 else "", best=parts[3] if len(parts) > 3 else "", events=[],
                checkpoint=str(ckpt.relative_to(ROOT)) if ckpt.exists() else "",
                plots=str((run_dir / "plots" / cur).relative_to(ROOT))))
        elif l.startswith("   -> da cap nhat"):
            promoted.add(cur)
    for k in keys:
        if k not in tasks:
            tasks[k] = dict(status="BLOCKED" if pre.get(k, {}).get("status") == "FAIL" else "PENDING")
    # --epochs applies to every task, so any started task with a non-default count reveals the override
    override = next((e for k, e in epochs.items() if e != TASK_BY_KEY[k]["epochs"]), None)
    image_size = 512
    for p in sorted((run_dir / "logs").glob("*.console.log")):
        m = re.search(r"\| image (\d+)px", p.read_text(encoding="utf-8", errors="replace")[:1000])
        if m:
            image_size = int(m.group(1))
            break
    done = [k for k, v in tasks.items() if v["status"] == "DONE"]
    return dict(version=1, run=run_dir.name, started=started, legacy=True, sessions=[],
                config=dict(tasks=keys,
                            epochs={k: epochs.get(k, override or TASK_BY_KEY[k]["epochs"]) for k in keys},
                            batch={k: batch.get(k, TASK_BY_KEY[k]["bs"]) for k in keys},
                            image_size=image_size, no_promote=bool(done) and not promoted),
                tasks=tasks)


def unfinished(st):
    return any(v["status"] not in FINAL_STATUSES for v in st["tasks"].values())


def find_resumable_run(name=None):
    """Most recent training run if it is unfinished (or the run given by name). (None, None) otherwise."""
    runs_root = ROOT / "runs"
    if name:
        d = Path(name) if Path(name).is_absolute() else runs_root / Path(name).name
        st = load_run_state(d) if d.is_dir() else None
        return (d, st) if st and unfinished(st) else (None, None)
    for d in sorted((p for p in runs_root.glob("2*") if p.is_dir()), reverse=True):
        st = load_run_state(d)
        if st is None:
            continue            # --check-only run or stopped at preflight
        return (d, st) if unfinished(st) else (None, None)
    return None, None


def other_trainers():
    """PIDs of other running train_all.py orchestrators or training scripts (they would share the GPU/run)."""
    try:
        import psutil
    except Exception:
        return []
    me, found = os.getpid(), []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if ("train_all.py" in cmd and "--_devinfo" not in cmd and "--check-only" not in cmd) \
                or "src.training.train_" in cmd:
            found.append((p.info["pid"], cmd[-110:]))
    return found


# ---------------------------------------------------------------------------------------------
# Helper modes (run in a subprocess)
# ---------------------------------------------------------------------------------------------
def helper_devinfo():
    import torch
    info = dict(python=platform.python_version(), python_exe=PY, torch=torch.__version__,
                cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.update(gpu_name=prop.name, gpu_total_gb=round(total / 1024**3, 2), gpu_free_gb=round(free / 1024**3, 2),
                    gpu_compute_capability=f"{prop.major}.{prop.minor}", gpu_sm_count=prop.multi_processor_count)
    print(json.dumps(info))


# ---------------------------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------------------------
def check_packages():
    import importlib
    missing = []
    for m in REQUIRED_PACKAGES:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    return missing


def check_manifest_paths(manifest):
    """Returns (rows, missing_count, examples) for every *path column of a manifest."""
    import pandas as pd
    df = pd.read_csv(manifest)
    cols = [c for c in df.columns if c.endswith("_path") or c == "image_path"]
    missing, examples, checked = 0, [], 0
    for c in cols:
        for v in df[c].dropna().astype(str):
            v = v.strip()
            if not v:
                continue
            checked += 1
            if not (ROOT / v).exists():
                missing += 1
                if len(examples) < 3:
                    examples.append(v)
    return len(df), checked, missing, examples


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def preflight(tasks, run_dir, log, args):
    log("\n" + "=" * 78)
    log(" [1/4] KIEM TRA TRUOC KHI TRAIN (PREFLIGHT)")
    log("=" * 78)
    report = dict(global_errors=[], warnings=[], tasks={})

    missing = check_packages()
    if missing:
        report["global_errors"].append(f"Thieu package: {', '.join(missing)} -> pip install {' '.join(missing)}")
    log(f"  [{'FAIL' if missing else ' OK '}] Python packages ({len(REQUIRED_PACKAGES)} package bat buoc)"
        + (f" - thieu: {missing}" if missing else ""))
    if missing:
        return report, None

    # Device information (fresh process so this orchestrator never holds VRAM)
    dev = run_helper(["--_devinfo"], timeout=120)
    try:
        import psutil
        dev["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        dev["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
        dev["cpu_logical_cores"] = psutil.cpu_count()
    except Exception:
        pass
    dev["cpu"] = cpu_name()
    ver = platform.version()
    try:
        win_name = "11" if platform.system() == "Windows" and int(ver.split(".")[2]) >= 22000 else platform.release()
    except Exception:
        win_name = platform.release()
    dev["os"] = f"{platform.system()} {win_name} ({ver})"
    dev["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    try:
        dev["gpu_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        dev["gpu_driver"] = ""
    try:
        dev["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                           text=True, timeout=10).stdout.strip()
    except Exception:
        dev["git_commit"] = ""
    (run_dir / "device_info.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")

    log("\n  THONG TIN THIET BI")
    log(f"    OS        : {dev['os']}")
    log(f"    CPU       : {dev.get('cpu')} ({dev.get('cpu_logical_cores', '?')} luong)")
    log(f"    RAM       : {dev.get('ram_total_gb', '?')} GB (con trong {dev.get('ram_available_gb', '?')} GB)")
    log(f"    Python    : {dev['python']} ({dev['python_exe']})")
    log(f"    PyTorch   : {dev['torch']} | CUDA {dev['cuda_version']} | cuDNN {dev['cudnn']}")
    if dev["cuda_available"]:
        log(f"    GPU       : {dev['gpu_name']} | VRAM {dev['gpu_total_gb']} GB (trong {dev['gpu_free_gb']} GB)"
            f" | driver {dev['gpu_driver']} | CC {dev['gpu_compute_capability']}")
    else:
        log("    GPU       : KHONG CO CUDA -> se train bang CPU (rat cham)")
    log(f"    O dia     : trong {dev['disk_free_gb']} GB | git commit {dev['git_commit']}")

    log("\n  KIEM TRA HE THONG")
    if not dev["cuda_available"]:
        msg = "Khong tim thay GPU CUDA. Train tren CPU co the mat nhieu ngay."
        if args.require_gpu:
            report["global_errors"].append(msg)
        else:
            report["warnings"].append(msg)
        log(f"  [{'FAIL' if args.require_gpu else 'WARN'}] {msg}")
    else:
        log(f"  [ OK ] CUDA GPU: {dev['gpu_name']}")
        if dev["gpu_free_gb"] < 2:
            report["warnings"].append(f"VRAM trong chi {dev['gpu_free_gb']} GB - hay dong bot ung dung (trinh duyet, game...).")
            log(f"  [WARN] VRAM trong chi {dev['gpu_free_gb']} GB")
    if dev["disk_free_gb"] < 5:
        report["warnings"].append(f"O dia chi con {dev['disk_free_gb']} GB")
        log(f"  [WARN] O dia chi con {dev['disk_free_gb']} GB (moi lan train can ~0.5-1 GB)")
    else:
        log(f"  [ OK ] O dia con {dev['disk_free_gb']} GB")

    log("\n  KIEM TRA DU LIEU TUNG BAI TOAN")
    for t in tasks:
        res = dict(status="OK", problems=[], warnings=[])
        script = ROOT / (t["module"].replace(".", "/") + ".py")
        if not script.exists():
            res["problems"].append(f"khong thay script {script.relative_to(ROOT)}")
        for m in t["manifests"]:
            mp = ROOT / m
            if not mp.exists():
                res["problems"].append(f"thieu manifest {m} (chay: python src/datasets/build_manifests.py)")
                continue
            rows, checked, miss, ex = check_manifest_paths(mp)
            if rows == 0:
                res["problems"].append(f"{m} rong")
            elif miss:
                ratio = miss / max(checked, 1)
                text = f"{m}: thieu {miss}/{checked} file anh/mask (vd: {ex[0]})"
                (res["problems"] if ratio > 0.05 else res["warnings"]).append(text)
        if res["problems"]:
            res["status"] = "FAIL"
        tag = "FAIL" if res["problems"] else ("WARN" if res["warnings"] else " OK ")
        log(f"  [{tag}] {t['name']}")
        for p in res["problems"] + res["warnings"]:
            log(f"         - {p}")
        report["tasks"][t["key"]] = res

    # Smoke test: 1 epoch on 8 samples, outputs into the run folder (never into checkpoints/)
    if args.smoke:
        log("\n  CHAY THU (SMOKE TEST: 1 epoch, 8 anh, batch 2) - phat hien loi code truoc khi train that")
        smoke_dir = run_dir / "smoke_test"
        smoke_dir.mkdir(exist_ok=True)
        for t in tasks:
            res = report["tasks"][t["key"]]
            if res["status"] == "FAIL":
                log(f"  [SKIP] {t['name']} (da loi o buoc kiem tra du lieu)")
                continue
            cmd = [PY, "-u", "-m", t["module"], *t["extra"], *args.task_cfg[t["key"]]["extra"],
                   "--epochs", "1", "--limit", "8", "--batch_size", "2",
                   "--image_size", str(min(args.task_cfg[t["key"]]["image_size"], 256)),
                   "--output_checkpoint", str(smoke_dir / f"{t['key']}.pth"), "--plot_dir", str(smoke_dir / t["key"])]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, encoding="utf-8",
                                   errors="replace", env=child_env())
                (smoke_dir / f"{t['key']}.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
                ok = p.returncode == 0
                err = "" if ok else (p.stderr.strip().splitlines() or ["(khong co stderr)"])[-1]
            except subprocess.TimeoutExpired:
                ok, err = False, "qua 20 phut"
            log(f"  [{' OK ' if ok else 'FAIL'}] {t['name']} ({time.time() - t0:.0f}s)"
                + ("" if ok else f" - {err}  -> xem {smoke_dir.relative_to(ROOT) / (t['key'] + '.log')}"))
            if not ok:
                res["status"] = "FAIL"
                res["problems"].append(f"smoke test loi: {err}")

    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report, dev


# ---------------------------------------------------------------------------------------------
# Batch size selection
# ---------------------------------------------------------------------------------------------
def choose_batch_sizes(tasks, dev, run_dir, log, args, previous=None):
    """Fixed batch sizes tuned once for RTX 4060 Laptop 8 GB @ 512px (see HARDWARE_PROFILE).
    previous: {task: bs} of the run being resumed (kept unless --batch-size is given)."""
    log("\n" + "=" * 78)
    log(f" [2/4] BATCH SIZE CO DINH ({HARDWARE_PROFILE})")
    log("=" * 78)
    free_gb = dev.get("gpu_free_gb") if dev else None
    chosen = {}
    for t in tasks:
        tc = args.task_cfg[t["key"]]
        img, ov = tc["image_size"], tc.get("batch")
        if ov:
            bs, reason = ov, "nguoi dung chi dinh"
        elif previous and t["key"] in previous:
            bs, reason = previous[t["key"]], "giu nguyen nhu luc bat dau run nay"
        else:
            bs, reason = t["bs"], f"can ~{t['peak_gb']} GB VRAM"
            if img != 512:
                reason += f" (do o 512px; dang dung {img}px)"
        if free_gb is not None and not ov and img == 512 and t["peak_gb"] > free_gb * 0.95:
            reason += f" | CANH BAO: VRAM trong chi {free_gb} GB -> dong bot app hoac dung --batch-size nho hon"
        chosen[t["key"]] = dict(bs=bs, reason=reason)
        log(f"  {t['name']:<34} batch = {bs:<3} ({reason})")
    if previous is None:   # resumed runs keep the batch_sizes.json written when the run started
        (run_dir / "batch_sizes.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
    return chosen


# ---------------------------------------------------------------------------------------------
# Training with live output + per-epoch logs
# ---------------------------------------------------------------------------------------------
def child_env():
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return env


def parse_metrics(text):
    return {k.strip(): float(v) for k, v in METRIC_RE.findall(text)}


def train_task(t, bs, epochs, run_dir, dev, log, args):
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    ckpt = run_dir / "checkpoints" / Path(t["checkpoint"]).name
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots" / t["key"]
    state_file = state_path_for(ckpt)
    resuming = state_file.exists()      # saved by the training script after every finished epoch
    img = args.task_cfg[t["key"]]["image_size"]
    cmd = [PY, "-u", "-m", t["module"], *t["extra"], *getattr(args, "extra_args", []), *args.task_cfg[t["key"]]["extra"],
           "--epochs", str(epochs), "--batch_size", str(bs),
           "--image_size", str(img), "--output_checkpoint", str(ckpt), "--plot_dir", str(plot_dir)]
    if resuming:
        cmd.append("--resume")
    device_str = f"cuda:0 {dev.get('gpu_name')}" if dev and dev.get("cuda_available") else "cpu"

    console_path = logs_dir / f"{t['key']}.console.log"
    epoch_csv_path = logs_dir / f"{t['key']}.epochs.csv"
    jsonl_path = logs_dir / f"{t['key']}.epochs.jsonl"
    csv_writer, csv_file, prev_seconds, prev_best = None, None, 0.0, ""
    if resuming:
        # continue the existing logs; elapsed_seconds in the CSV is cumulative over all sessions
        if epoch_csv_path.exists():
            with open(epoch_csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                try:
                    prev_seconds = float(rows[-1]["elapsed_seconds"])
                except (KeyError, ValueError):
                    pass
                csv_file = open(epoch_csv_path, "a", newline="", encoding="utf-8")
                csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        if console_path.exists():   # best checkpoint so far (a later session may not improve on it)
            saved = [l for l in console_path.read_text(encoding="utf-8", errors="replace").splitlines()
                     if "Saved new best" in l]
            prev_best = saved[-1].split("] ", 1)[-1].strip() if saved else ""
    else:
        # an earlier attempt of this task stopped before saving any epoch state: keep its logs aside
        old = [p for p in (console_path, epoch_csv_path, jsonl_path) if p.exists()]
        if old:
            aside = logs_dir / f"aborted_{t['key']}_{dt.datetime.now():%Y%m%d_%H%M%S}"
            aside.mkdir()
            for p in old:
                shutil.move(str(p), str(aside / p.name))
            log(f"   (log lan chay dang do truoc do cua bai nay -> {aside.relative_to(ROOT)})")
    mode = "a" if resuming else "w"
    console_log = open(console_path, mode, encoding="utf-8")
    epoch_jsonl = open(jsonl_path, mode, encoding="utf-8")

    header = (f"# {t['name']} | {'resume' if resuming else 'start'} {now()} | device {device_str} | batch {bs} | "
              f"epochs {epochs} | image {img}px | git {dev.get('git_commit', '') if dev else ''}\n"
              f"# cmd: {' '.join(cmd)}\n")
    console_log.write(header)
    log(f"\n{'-' * 78}\n >>> {t['name']}  |  batch {bs}  |  {epochs} epoch  |  {device_str}"
        + (f"\n     TIEP TUC tu trang thai da luu ({state_file.name})" if resuming else "") + f"\n{'-' * 78}")

    start = time.time()
    last_epoch_t = start
    epochs_done, best_line, events, n_session = 0, prev_best, [], 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env(), cwd=ROOT)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf, pending = "", ""
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            text = pending + decoder.decode(chunk)
            pending = ""
            if text.endswith("\r"):          # a "\r\n" may be split across two chunks
                pending, text = "\r", text[:-1]
            text = text.replace("\r\n", "\n")  # Windows child processes emit CRLF
            sys.stdout.write(text)          # live terminal view incl. tqdm progress bars
            sys.stdout.flush()
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.split("\r")[-1].rstrip()   # keep only the final state of a progress bar
                if not line:
                    continue
                console_log.write(f"[{now()}] {line}\n")
                console_log.flush()
                if "Saved new best" in line or ("[+] Saved" in line and "checkpoint" in line.lower()):
                    best_line = line.strip()
                    events.append(f"ep{epochs_done}: {line.strip()}")
                if "Early stopping triggered" in line:
                    events.append(line.strip())
                if "Test Benchmark Result" in line or "Official Test Result" in line:
                    events.append(line.strip())
                rm = RESUME_RE.search(line)
                if rm:
                    epochs_done = int(rm.group(3))
                    events.append(line.strip().split(" <- ")[0])
                m = EPOCH_RE.match(line.strip())
                if m:
                    ep, total = int(m.group(1)), int(m.group(2))
                    tnow = time.time()
                    n_session += 1
                    ep_sec, elapsed = tnow - last_epoch_t, prev_seconds + (tnow - start)
                    last_epoch_t = tnow
                    epochs_done = ep
                    eta = (tnow - start) / n_session * (total - ep)
                    metrics = parse_metrics(m.group(3))
                    rec = dict(timestamp=now(), task=t["key"], epoch=ep, total_epochs=total,
                               epoch_seconds=round(ep_sec, 1), elapsed_seconds=round(elapsed, 1),
                               eta_seconds=round(eta, 1), batch_size=bs, image_size=img,
                               device=device_str, **gpu_stats(), **ram_stats(), **metrics)
                    epoch_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    epoch_jsonl.flush()
                    if csv_writer is None:
                        csv_file = open(epoch_csv_path, "w", newline="", encoding="utf-8")
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(rec.keys()), extrasaction="ignore")
                        csv_writer.writeheader()
                    csv_writer.writerow(rec)
                    csv_file.flush()
                    g = rec
                    log(f"   [{rec['timestamp']}] {t['key']} ep {ep:02d}/{total:02d} | {fmt_dur(ep_sec)} "
                        f"(da chay {fmt_dur(elapsed)}, con ~{fmt_dur(eta)}) | GPU {g['gpu_mem_used_mb']}/"
                        f"{g['gpu_mem_total_mb']}MB {g['gpu_util_pct']}% {g['gpu_temp_c']}C | RAM {g['ram_used_pct']}%")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        console_log.close()
        epoch_jsonl.close()
        if csv_file:
            csv_file.close()
        raise
    console_log.write(f"# end {now()} | exit {proc.returncode} | duration {fmt_dur(time.time() - start)}\n")
    console_log.close()
    epoch_jsonl.close()
    if csv_file:
        csv_file.close()

    if proc.returncode == 0 and ckpt.exists():
        status = "DONE"
    elif proc.returncode == 0:
        status = "NO_CHECKPOINT"   # script finished but the metric never improved, so nothing was saved
        events.append("Script chay xong nhung khong luu checkpoint (metric val khong bao gio > 0). "
                      "Thuong do qua it epoch.")
    else:
        status = "FAILED"
        events.append(f"Loi exit {proc.returncode} - xem {(logs_dir / (t['key'] + '.console.log')).relative_to(ROOT)}"
                      " (chay lai train.bat de thu tiep tu epoch da luu)")
    if status != "FAILED":
        # task finished: the resume state (model + optimizer, up to ~380 MB) is no longer needed
        for p in (state_file, state_file.with_name(state_file.name + ".tmp")):
            p.unlink(missing_ok=True)
    return dict(status=status, exit_code=proc.returncode, epochs_done=epochs_done,
                epochs_planned=epochs, batch_size=bs, duration=fmt_dur(prev_seconds + time.time() - start),
                best=best_line, events=events, checkpoint=str(ckpt.relative_to(ROOT)) if ckpt.exists() else "",
                plots=str(plot_dir.relative_to(ROOT)), resumed=resuming)


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def approve_run(run_name, keys, log, assume_yes=False):
    """Approval step: copies the checkpoints and result plots of the approved tasks of a finished run into the
    project (checkpoints/ = what main.py loads, outputs/plots/<task>/ = the published plots).
    Everything that gets replaced is first backed up to runs/<run>/approved_backup/<timestamp>/.
    Returns 0 on success."""
    run_dir = ROOT / "runs" / Path(run_name).name
    st = load_run_state(run_dir) if run_dir.is_dir() else None
    if not st:
        print(f"[X] Khong tim thay run '{run_name}' trong runs/ (hoac run chua train bai nao).")
        return 2
    done = [k for k, v in st["tasks"].items() if v["status"] == "DONE"]
    sel = done if not keys else keys
    bad = [k for k in sel if k not in done]
    if bad:
        print(f"[X] Bai khong duyet duoc (chua DONE trong run nay): {bad}. Bai DONE: {done}")
        return 2
    if not sel:
        print("[X] Run nay khong co bai nao DONE de duyet.")
        return 2

    plan = []
    for k in sel:
        t = TASK_BY_KEY[k]
        ckpt = run_dir / "checkpoints" / Path(t["checkpoint"]).name
        plots = run_dir / "plots" / k
        if not ckpt.is_file():
            print(f"[X] {t['name']}: thieu checkpoint {ckpt.relative_to(ROOT)}")
            return 2
        plan.append((t, ckpt, plots, ROOT / t["checkpoint"], ROOT / "outputs" / "plots" / k))

    print(f"\n=== DUYET KET QUA RUN {run_dir.name} ===")
    for t, ckpt, plots, dst, pdst in plan:
        same = dst.is_file() and file_md5(ckpt) == file_md5(dst)
        r = st["tasks"][t["key"]].get("result", {})
        print(f"  {t['name']:<34} {r.get('epochs_done', '?')}/{r.get('epochs_planned', '?')} epoch | {r.get('duration', '')}")
        print(f"      checkpoint : {ckpt.relative_to(ROOT)} -> {t['checkpoint']}" + ("  (giong het ban dang dung)" if same else ""))
        print(f"      bieu do    : {plots.relative_to(ROOT) if plots.is_dir() else '(khong co)'} -> outputs/plots/{t['key']}/")
        print(f"      log        : logs/{t['key']}.* -> outputs/logs/train/{run_dir.name}/")
    if not assume_yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("\n[X] Can xac nhan: chay lai voi --yes (hoac chay train.bat --approve trong cua so terminal).")
            return 2
        try:
            ans = input("\nDuyet va sao chep cac muc tren ra checkpoints/ va outputs/ cua project? (ban cu duoc sao luu) [y/N]: ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("[i] Khong sao chep gi.")
            return 1

    tlog = Tee(run_dir / "approve_log.txt")
    backup_root = run_dir / "approved_backup" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    entries = []
    for t, ckpt, plots, dst, pdst in plan:
        if dst.exists():
            b = backup_root / t["checkpoint"]
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, b)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, dst)
        copied_plots = False
        if plots.is_dir() and any(plots.iterdir()):
            if pdst.exists():
                shutil.copytree(pdst, backup_root / "outputs" / "plots" / t["key"])
                shutil.rmtree(pdst)
            shutil.copytree(plots, pdst)
            copied_plots = True
        entries.append(dict(task=t["key"], checkpoint=t["checkpoint"], md5=file_md5(dst), plots=copied_plots))
        tlog(f"[{now()}] {t['name']}: {t['checkpoint']} <- {ckpt.relative_to(ROOT)}"
             + (f" | outputs/plots/{t['key']}/ cap nhat" if copied_plots else ""))
    out_logs = ROOT / "outputs" / "logs" / "train" / run_dir.name
    out_logs.mkdir(parents=True, exist_ok=True)
    for t, _c, _p, _d, _pd in plan:
        for f in (run_dir / "logs").glob(f"{t['key']}.*"):
            shutil.copy2(f, out_logs / f.name)
    if (run_dir / "SUMMARY.md").is_file():
        shutil.copy2(run_dir / "SUMMARY.md", out_logs / "SUMMARY.md")
    tlog(f"  Log tung epoch + SUMMARY.md: {out_logs.relative_to(ROOT)}/")
    hist_path = run_dir / "approved.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.is_file() else []
    hist.append(dict(time=now(), items=entries, backup=str(backup_root.relative_to(ROOT))))
    hist_path.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    tlog(f"  Ban cu da sao luu tai: {backup_root.relative_to(ROOT)}")
    return 0


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------
def write_summary(run_dir, dev, report, chosen, results, tasks, started, log):
    lines = [f"# FUNDUS AI - Bao cao train ({run_dir.name})", "",
             f"- Bat dau: {started} | Ket thuc: {now()}",
             f"- Git commit: {dev.get('git_commit', '') if dev else ''}"]
    if dev:
        lines += [f"- GPU: {dev.get('gpu_name', 'CPU')} ({dev.get('gpu_total_gb', '-')} GB, trong {dev.get('gpu_free_gb', '-')} GB "
                  f"luc bat dau) | driver {dev.get('gpu_driver', '')}",
                  f"- CPU: {dev.get('cpu')} | RAM {dev.get('ram_total_gb')} GB | PyTorch {dev.get('torch')} CUDA {dev.get('cuda_version')}",
                  f"- Python: {dev.get('python_exe')}"]
    if report["global_errors"] or report["warnings"]:
        lines += ["", "## Canh bao / loi he thong"] + [f"- {x}" for x in report["global_errors"] + report["warnings"]]
    lines += ["", "## Ket qua tung bai toan", "",
              "| Bai toan | Trang thai | Batch | Epoch | Thoi gian | Checkpoint tot nhat |", "|---|---|---|---|---|---|"]
    for t in tasks:
        r = results.get(t["key"])
        pre = report["tasks"].get(t["key"], {})
        if r is None:
            if pre.get("problems"):
                status = "CHUA TRAIN DUOC: " + "; ".join(pre["problems"])
            elif t["key"] in chosen:
                status = "SAN SANG (chua train)"
            else:
                status = "chua train"
            bs = chosen.get(t["key"], {}).get("bs", "-")
            lines.append(f"| {t['name']} | {status} | {bs} | - | - | - |")
        else:
            lines.append(f"| {t['name']} | {r['status']} (exit {r['exit_code']}) | {r['batch_size']} | "
                         f"{r['epochs_done']}/{r['epochs_planned']} | {r['duration']} | {r['best'].replace('|', '/')} |")
    lines += ["", "## Chi tiet", ""]
    for t in tasks:
        r = results.get(t["key"])
        if not r:
            continue
        lines += [f"### {t['name']}", f"- Batch size: {r['batch_size']} ({chosen.get(t['key'], {}).get('reason', '')})",
                  f"- Checkpoint: `{r['checkpoint'] or '(khong co)'}` | Bieu do: `{r['plots']}`",
                  f"- Log tung epoch: `{(run_dir / 'logs' / (t['key'] + '.epochs.csv')).relative_to(ROOT)}` | "
                  f"log day du: `{(run_dir / 'logs' / (t['key'] + '.console.log')).relative_to(ROOT)}`"]
        lines += [f"- {e}" for e in r["events"][-6:]] + [""]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines[lines.index("## Ket qua tung bai toan"):lines.index("## Chi tiet")]))
    log(f"\n  Bao cao: {(run_dir / 'SUMMARY.md').relative_to(ROOT)}")


# ---------------------------------------------------------------------------------------------
# Status & Interactive Menu
# ---------------------------------------------------------------------------------------------
def show_training_status():
    runs_root = ROOT / "runs"
    if not runs_root.is_dir():
        print("\n[i] Chua co thu muc runs/ (chua co lan train nao chay qua train.bat).")
        return 0
    all_runs = sorted((p for p in runs_root.glob("2*") if p.is_dir()), reverse=True)
    if not all_runs:
        print("\n[i] Chua co lan train nao trong runs/.")
        return 0
    print("\n" + "=" * 80)
    print(" FUNDUS AI - LICH SU & TRANG THAI HUAN LUYEN (TRAIN STATUS)")
    print("=" * 80)
    for rd in all_runs[:10]:
        st = load_run_state(rd)
        if not st:
            print(f"\n[Run {rd.name}] (chua co state hoac check-only)")
            continue
        cfg = st.get("config", {})
        started = st.get("started", "?")
        sessions = st.get("sessions", [])
        last_end = sessions[-1].get("end") if sessions else None
        print(f"\n[Run {rd.name}]  Bat dau: {started}" + (f" | Gan nhat: {last_end}" if last_end else ""))
        print(f"  Cau hinh: {len(cfg.get('tasks', []))} tasks | img: {cfg.get('image_size', 512)}px")
        for k, v in st.get("tasks", {}).items():
            t = TASK_BY_KEY.get(k, {})
            t_name = t.get("name", k)
            status = v.get("status", "UNKNOWN")
            r = v.get("result", {})
            done_ep = r.get("epochs_done", "?")
            plan_ep = r.get("epochs_planned", cfg.get("epochs", {}).get(k, "?"))
            dur = r.get("duration", "")
            best = r.get("best", "")

            ckpt = rd / "checkpoints" / Path(t.get("checkpoint", f"{k}.pth")).name
            last_ckpt = state_path_for(ckpt)

            res_info = ""
            if last_ckpt.exists():
                res_info = f" -> CO STATE RESUME ({last_ckpt.name})"

            print(f"    * {t_name:<34} : {status:<12} (ep {done_ep}/{plan_ep}) {dur}{res_info}")
            if best:
                print(f"      Best: {best[:70]}")
    print("=" * 80 + "\n")
    return 0


def latest_unapproved_run():
    """Newest run that has DONE tasks which were not approved yet (None if there is no such run)."""
    for d in sorted((p for p in (ROOT / "runs").glob("2*") if p.is_dir()), reverse=True):
        st = load_run_state(d)
        if not st:
            continue
        ap_file = d / "approved.json"
        approved = {e["task"] for h in (json.loads(ap_file.read_text(encoding="utf-8")) if ap_file.is_file() else [])
                    for e in h["items"]}
        if [k for k, v in st["tasks"].items() if v["status"] == "DONE" and k not in approved]:
            return d.name
    return None


def interactive_menu(args):
    """Interactive CLI menu when train.bat is run with no arguments."""
    resumable_dir, resumable_st = find_resumable_run()

    print("\n" + "=" * 78)
    print("           FUNDUS AI - BANG DIEU KHIEN HUAN LUYEN (TRAIN MANAGER)")
    print("=" * 78)

    if resumable_st:
        cfg = resumable_st.get("config", {})
        print(f"\n  [!] PHAT HIEN RUN DANG DO CO THE TIEP TUC: runs/{resumable_dir.name}")
        print(f"      Bat dau luc: {resumable_st.get('started', '?')}")
        for k, v in resumable_st.get("tasks", {}).items():
            t_name = TASK_BY_KEY.get(k, {}).get("name", k)
            st_text = v.get("status", "UNKNOWN")
            r = v.get("result", {})
            ep_info = f"ep {r.get('epochs_done', 0)}/{cfg.get('epochs', {}).get(k, '?')}" if "epochs_done" in r else ""
            print(f"        * {t_name:<34} : {st_text:<12} {ep_info}")
        print(f"      => Ban co the tiep tuc ngay tu diem da dung (Ctrl+C) ma khong mat du lieu.")

    print("\n  LUA CHON HANH DONG:")
    if resumable_st:
        print("    [1] (Enter) Tiep tuc (RESUME) run dang do tren")
    else:
        print("    [1] Train TAT CA 6 bai toan co ban (M1 -> M6)")
    print("    [2] Chon module cu the de train (M1, M2, M3, M4, M4-SOTA, M5, M6)")
    print("    [3] Train moi tat ca tu dau (--new)")
    print("    [4] Kiem tra du lieu va thiet bi truoc (--check-only)")
    print("    [5] Chay thu 1 epoch/8 anh moi bai (--smoke)")
    print("    [6] Xem lich su & tien do cac run (train toi dau)")
    print("    [7] DUYET ket qua mot run (copy checkpoint + bieu do + log ra project)")
    print("    [0] Thoat")

    prompt = "\n  Nhap lua chon [1]: " if resumable_st else "\n  Nhap lua chon [1-6, 0]: "
    try:
        choice = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[i] Da huy.")
        sys.exit(0)

    if choice == "" or choice == "1":
        if resumable_st:
            args.resume = resumable_dir.name
        else:
            args.tasks = "all"
    elif choice == "2":
        print("\n  DANH SACH MODULE CO THE TRAIN:")
        for idx, t in enumerate(TASKS, 1):
            opt = " [SOTA Phase 3]" if t.get("optional") else ""
            print(f"    [{idx}] {t['key']:<15} : {t['name']}{opt} ({t['epochs']} ep, batch {t['bs']})")
        print("    [A] Tat ca cac module tren")

        try:
            sel_input = input("\n  Nhap so thu tu module (vd: 5 hoac 2,4,5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[i] Da huy.")
            sys.exit(0)

        if sel_input.lower() == "a":
            args.tasks = "all"
        else:
            chosen_keys = []
            for part in sel_input.replace(",", " ").split():
                if part.isdigit():
                    num = int(part)
                    if 1 <= num <= len(TASKS):
                        chosen_keys.append(TASKS[num - 1]["key"])
                elif part in TASK_BY_KEY:
                    chosen_keys.append(part)
            if not chosen_keys:
                print("[!] Lua chon khong hop le.")
                sys.exit(1)
            args.tasks = ",".join(chosen_keys)

        try:
            ep_input = input("  So epoch muon train (Enter de giu mac dinh): ").strip()
            if ep_input.isdigit() and int(ep_input) > 0:
                args.epochs = int(ep_input)
        except (EOFError, KeyboardInterrupt):
            pass

        try:
            bs_input = input("  Batch size (Enter de giu mac dinh toi uu cho 8GB VRAM): ").strip()
            if bs_input.isdigit() and int(bs_input) > 0:
                args.batch_size = int(bs_input)
        except (EOFError, KeyboardInterrupt):
            pass

    elif choice == "3":
        args.new = True
        args.tasks = "all"
    elif choice == "4":
        args.check_only = True
    elif choice == "5":
        args.smoke = True
    elif choice == "6":
        show_training_status()
        sys.exit(0)
    elif choice == "7":
        try:
            run_in = input(f"  Ten run trong runs/ (Enter = run moi nhat chua duyet: {latest_unapproved_run()}): ").strip()
            task_in = input("  Bai can duyet, vd M1,M4 (Enter = tat ca bai DONE cua run): ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        args.approve = run_in or "latest"
        args.tasks = task_in or "all"
    elif choice == "0":
        print("[i] Thoat.")
        sys.exit(0)
    else:
        print("[!] Lua chon khong hop le.")
        sys.exit(1)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FUNDUS AI - train all tasks")
    ap.add_argument("--tasks", default="all", help="vd: M1,M4 hoac vessels,lesions (mac dinh: all = 6 bai M1..M6). "
                    "Khoa: " + ",".join(t["key"] for t in TASKS))
    ap.add_argument("--epochs", type=int, default=None, help="ghi de so epoch cho moi bai toan")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="ghi de batch co dinh cho moi bai")
    ap.add_argument("--image-size", dest="image_size", type=int, default=None, help="mac dinh 512")
    ap.add_argument("--check-only", action="store_true", help="chi kiem tra thiet bi + du lieu, khong train")
    ap.add_argument("--smoke", action="store_true", help="chay thu 1 epoch/8 anh moi bai truoc khi train (~1 phut)")
    ap.add_argument("--extra", default=None, metavar='"ARGS"',
                    help='tham so them cho script train cua cac bai da chon, vd: --extra "--lr 0.0001 --patience 8" '
                         '(de ghi de mac dinh; dung --batch-size/--epochs/--image-size cho 3 tham so do)')
    ap.add_argument("--approve", metavar="RUN", nargs="?", const="latest", default=None,
                    help="DUYET ket qua mot run: copy checkpoint -> checkpoints/, bieu do -> outputs/plots/<bai>/, "
                         "log -> outputs/logs/train/<run>/. Khong ghi RUN = run moi nhat chua duyet. "
                         "Dung --tasks M1,M4 de chi duyet mot so bai (mac dinh: moi bai da DONE), --yes de khong hoi")
    ap.add_argument("--yes", action="store_true", help="dung voi --approve: khong hoi xac nhan")
    ap.add_argument("--promote", action="store_true",
                    help="tu dong duyet: chep ngay checkpoint + bieu do moi vao checkpoints/ khi tung bai train xong")
    ap.add_argument("--no-promote", action="store_true", help=argparse.SUPPRESS)  # cu: khong con tu chep mac dinh
    ap.add_argument("--require-gpu", action="store_true", help="dung lai neu khong co GPU")
    ap.add_argument("--new", action="store_true", help="bat dau run moi, KHONG tiep tuc run dang do")
    ap.add_argument("--resume", metavar="RUN", nargs="?", const="latest", default=None,
                    help="tiep tuc run dang do (khong ghi RUN = run dang do moi nhat). Chay train.bat khong tham so "
                         "cung tu dong tiep tuc run dang do")
    ap.add_argument("--set", action="append", metavar='TASK="ARGS"',
                    help='cau hinh rieng tung bai, vd: --set M4="--epochs 35 --batch-size 2 --lr 0.0002" '
                         '(co the lap lai; TASK = M1..M6, M4S hoac ten bai)')
    for _alias, _key in TASK_ALIASES.items():
        ap.add_argument(f"--{_alias}", dest=f"opt_{_alias}", default=None, metavar='"ARGS"',
                        help=f"chon bai {_alias.upper()} ({_key}) va cau hinh rieng, vd: --{_alias} \"--epochs 30\"")
    ap.add_argument("--status", action="store_true", help="xem lich su va tien do cac run (train toi dau)")
    ap.add_argument("--menu", action="store_true", help="hien thi bang dieu khien interactive de chon module")
    ap.add_argument("--_devinfo", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._devinfo:
        return helper_devinfo()
    if args.status:
        return show_training_status()
    if args.menu and sys.stdin and sys.stdin.isatty():
        interactive_menu(args)
    if args.new and args.resume:
        print("[!] Khong dung --new cung --resume.")
        return 2
    if args.approve:
        keys, bad = resolve_keys(args.tasks) if args.tasks != "all" else ([], [])
        if bad:
            print(f"[!] Khong biet bai toan: {bad}. Hop le: M1..M6, M4S hoac {list(TASK_BY_KEY)}")
            return 2
        run_name = latest_unapproved_run() if args.approve == "latest" else args.approve
        if not run_name:
            print("[i] Khong co run nao co bai DONE chua duyet.")
            return 0
        return approve_run(run_name, keys, print, assume_yes=args.yes)
    try:
        extra_args = shlex.split(args.extra, posix=False) if args.extra else []
    except ValueError as exc:
        print(f"[!] --extra khong hop le: {exc}")
        return 2
    extra_args = [a.strip('"') for a in extra_args]
    task_opts = {}
    try:
        for spec in args.set or []:
            name, _, val = spec.partition("=")
            ks, bad = resolve_keys(name)
            if bad or len(ks) != 1 or name.strip().lower() == "all":
                print(f'[!] --set can dang TASK="ARGS" voi TASK la 1 bai (M1..M6, M4S): nhan duoc "{spec}"')
                return 2
            task_opts[ks[0]] = parse_task_opts(val)
        for alias, key in TASK_ALIASES.items():
            val = getattr(args, f"opt_{alias}")
            if val is not None:
                task_opts[key] = parse_task_opts(val)
    except ValueError as exc:
        print(f"[!] Cau hinh bai khong hop le: {exc}")
        return 2
    config_flags = [f for f, v in (("--tasks", args.tasks != "all"), ("--epochs", args.epochs),
                                   ("--batch-size", args.batch_size), ("--image-size", args.image_size),
                                   ("--extra", args.extra), ("--set/--m1..--m6", task_opts)) if v]

    # ---- continue an unfinished run? -------------------------------------------------------
    run_dir, rstate = None, None
    if not args.check_only:
        busy = other_trainers()
        if busy:
            print("[X] Dang co tien trinh train khac chay - chay song song se tranh GPU va ghi de log:")
            for pid, cmd in busy:
                print(f"      PID {pid}: ...{cmd}")
            print("    Dong cua so train do (hoac Ctrl+C trong cua so do) roi chay lai train.bat.")
            return 3
        # train.bat with no settings (or --resume) continues the unfinished run; any setting starts a new run
        if not args.new and (args.resume or not config_flags):
            run_dir, rstate = find_resumable_run(None if args.resume in (None, "latest") else args.resume)
            if args.resume and not run_dir:
                print(f"[X] Khong tiep tuc duoc '{args.resume}': khong co trong runs/, chua tung train, hoac da xong het.")
                return 2

    started = now()
    if rstate:
        cfg = rstate["config"]
        tasks = [TASK_BY_KEY[k] for k in cfg["tasks"]]
        ignored = config_flags
        args.image_size = cfg["image_size"]
        extra_args = cfg.get("extra", [])
        log = Tee(run_dir / "train_log.txt")
        log("\n" + "=" * 78)
        log(f" FUNDUS AI - TIEP TUC (RESUME) | {started}")
        log(f" Run dang do: {run_dir.relative_to(ROOT)} (bat dau {rstate.get('started', '?')})")
        for t in tasks:
            log(f"   {t['name']:<34} {rstate['tasks'][t['key']]['status']}")
        if ignored:
            log(f" [i] Bo qua {', '.join(ignored)}: run cu giu nguyen cau hinh. Muon train moi: train.bat --new ...")
        log("=" * 78)
        todo = [t for t in tasks if rstate["tasks"][t["key"]]["status"] not in FINAL_STATUSES]
    else:
        if args.tasks != "all":
            sel, unknown = resolve_keys(args.tasks)
        else:                                  # default: all 6 modules, or the modules given with --m1..--m6 / --set
            sel, unknown = (list(task_opts) if task_opts else DEFAULT_KEYS), []
        if unknown or not sel:
            print(f"[!] Khong biet bai toan: {unknown}. Hop le: M1..M6, M4S hoac {list(TASK_BY_KEY)}")
            return 2
        stray = [k for k in task_opts if k not in sel]
        if stray:
            print(f"[!] Cau hinh cho {stray} bi bo qua vi bai do khong nam trong --tasks")
        tasks = todo = [t for t in TASKS if t["key"] in sel]
        args.image_size = args.image_size or 512
        run_dir = ROOT / "runs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        log = Tee(run_dir / "train_log.txt")
        log("=" * 78)
        log(f" FUNDUS AI - TRAIN TOAN BO | {started}")
        log(f" Thu muc ket qua: {run_dir.relative_to(ROOT)}  (khong ghi de lan train truoc)")
        log(f" Bai toan: {', '.join(t['key'] for t in tasks)}")
        if extra_args:
            log(f" Tham so them (--extra): {' '.join(extra_args)}")
        log("=" * 78)
    args.extra_args = extra_args
    args.task_cfg = {}                     # resolved per-task settings: epochs, batch (None = fixed default), image, extra
    for t in tasks:
        k = t["key"]
        if rstate:
            args.task_cfg[k] = dict(epochs=cfg["epochs"].get(k, t["epochs"]), batch=None,
                                    image_size=cfg.get("image_sizes", {}).get(k, cfg["image_size"]),
                                    extra=cfg.get("extra_by_task", {}).get(k, []))
        else:
            o = task_opts.get(k, {})
            args.task_cfg[k] = dict(epochs=o.get("epochs") or args.epochs or t["epochs"],
                                    batch=o.get("batch") or args.batch_size,
                                    image_size=o.get("image_size") or args.image_size,
                                    extra=o.get("extra", []))
    run_started = rstate.get("started", started) if rstate else started

    # ---- preflight + batch sizes (only for the tasks that still need training) ----------------
    report, dev = preflight(todo, run_dir, log, args)
    if report["global_errors"]:
        log("\n[X] DUNG: loi he thong:\n" + "\n".join(f"    - {e}" for e in report["global_errors"]))
        write_summary(run_dir, dev, report, {}, {}, tasks, run_started, log)
        return 1
    runnable = [t for t in todo if report["tasks"][t["key"]]["status"] != "FAIL"]
    blocked = [t for t in todo if t not in runnable]
    if blocked:
        log("\n[!] CHUA TRAIN DUOC (se bo qua):")
        for t in blocked:
            log(f"    - {t['name']}: {'; '.join(report['tasks'][t['key']]['problems'])}")
    if not runnable:
        log("\n[X] Khong con bai toan nao chay duoc.")
        write_summary(run_dir, dev, report, {}, {}, tasks, run_started, log)
        return 1

    chosen = choose_batch_sizes(runnable, dev, run_dir, log, args,
                                previous=rstate["config"]["batch"] if rstate else None)
    log("\n  CAU HINH TUNG BAI")
    for t in runnable:
        k, tc = t["key"], args.task_cfg[t["key"]]
        ep = rstate["config"]["epochs"][k] if rstate else tc["epochs"]
        log(f"    {t['name']:<34} epoch {ep:<3} batch {chosen[k]['bs']:<3} anh {tc['image_size']}px"
            + (f" | them: {' '.join(extra_args + tc['extra'])}" if extra_args or tc["extra"] else ""))
    if args.check_only:
        log("\n[i] --check-only: dung truoc khi train.")
        write_summary(run_dir, dev, report, chosen, {}, tasks, run_started, log)
        return 0

    if not rstate:
        rstate = dict(version=1, run=run_dir.name, started=started, sessions=[],
                      config=dict(tasks=[t["key"] for t in tasks],
                                  epochs={t["key"]: args.task_cfg[t["key"]]["epochs"] for t in tasks},
                                  batch={t["key"]: chosen.get(t["key"], {}).get("bs", t["bs"]) for t in tasks},
                                  image_size=args.image_size, extra=extra_args,
                                  image_sizes={t["key"]: args.task_cfg[t["key"]]["image_size"] for t in tasks},
                                  extra_by_task={t["key"]: args.task_cfg[t["key"]]["extra"] for t in tasks}),
                      tasks={t["key"]: dict(status="PENDING") for t in tasks})
    for t in blocked:
        rstate["tasks"][t["key"]] = dict(status="BLOCKED", problems=report["tasks"][t["key"]]["problems"])
    rstate.setdefault("sessions", []).append(dict(start=started, end=None))
    save_run_state(run_dir, rstate)

    # ---- train ----------------------------------------------------------------------------------
    log("\n" + "=" * 78)
    log(" [3/4] TRAIN" + (f"  ({len(runnable)} bai con lai)" if len(runnable) < len(tasks) else ""))
    log("=" * 78)
    results = {k: v["result"] for k, v in rstate["tasks"].items() if v.get("result") and v["status"] in FINAL_STATUSES}
    current = None
    try:
        for i, t in enumerate(runnable, 1):
            current = t["key"]
            log(f"\n[{i}/{len(runnable)}] {t['name']}")
            rstate["tasks"][current] = dict(status="RUNNING", since=now())
            save_run_state(run_dir, rstate)
            r = train_task(t, chosen[current]["bs"], rstate["config"]["epochs"][current], run_dir, dev, log, args)
            results[current] = r
            rstate["tasks"][current] = dict(status=r["status"], result=r)
            save_run_state(run_dir, rstate)
            log(f"   => {r['status']} | {r['epochs_done']}/{r['epochs_planned']} epoch | {r['duration']} | {r['best']}")
            if r["status"] == "DONE" and args.promote:
                approve_run(run_dir.name, [current], log, assume_yes=True)
            current = None
    except KeyboardInterrupt:
        if current:
            rstate["tasks"][current] = dict(status="INTERRUPTED", since=now())
        log("\n[!] Da dung (Ctrl+C). Cac bai da xong van duoc giu; bai dang chay luu den epoch xong gan nhat.")
    rstate["sessions"][-1]["end"] = now()
    save_run_state(run_dir, rstate)

    log("\n" + "=" * 78)
    log(" [4/4] TONG KET")
    log("=" * 78)
    write_summary(run_dir, dev, report, chosen, results, tasks, run_started, log)
    if unfinished(rstate):
        log("\n  [i] Run chua xong. Chay lai train.bat de TIEP TUC (bai da xong se bo qua, bai dang do train tiep"
            " tu epoch da luu). Muon train lai tu dau: train.bat --new")
    done_keys = [k for k, v in rstate["tasks"].items() if v["status"] == "DONE"]
    approved = {e["task"] for h in (json.loads((run_dir / "approved.json").read_text(encoding="utf-8"))
                                    if (run_dir / "approved.json").is_file() else []) for e in h["items"]}
    pending = [k for k in done_keys if k not in approved]
    if pending and not args.promote:
        hint = f"train.bat --approve {run_dir.name} --tasks {','.join(pending)}"
        log(f"\n  [i] Ket qua CHUA duoc chep vao checkpoints/ va outputs/plots/. Sau khi xem SUMMARY.md, duyet bang:\n      {hint}")
        if sys.stdin and sys.stdin.isatty():
            try:
                if input("\n  Duyet va sao chep ngay bay gio? [y/N]: ").strip().lower() in ("y", "yes"):
                    approve_run(run_dir.name, pending, log, assume_yes=True)
            except (EOFError, KeyboardInterrupt):
                pass
    return 0 if all(v["status"] == "DONE" for v in rstate["tasks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)

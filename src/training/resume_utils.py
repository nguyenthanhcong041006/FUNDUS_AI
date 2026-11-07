"""Save / restore the full training state so an interrupted training run can continue (--resume).

After every epoch the training scripts write `<checkpoint stem>_last.pth` next to the best checkpoint,
e.g. best_dr_classifier.pth -> best_dr_classifier_last.pth. It holds the model, optimizer, LR scheduler,
RNG states, the epoch number and the loop variables (best metric, early-stopping counter, history).
With --resume the script reloads it and continues from the next epoch.

The file is written atomically (tmp file + rename), so stopping the process (Ctrl+C, closing the window,
power loss) at any moment leaves either the previous or the new state on disk, never a broken one.
"""

import os
import random
from pathlib import Path


def state_path_for(output_checkpoint) -> Path:
    """best_x.pth -> best_x_last.pth (same rule is duplicated in scripts/train_all.py)."""
    p = Path(output_checkpoint)
    return p.with_name(p.stem + "_last.pth")


def save_training_state(output_checkpoint, epoch, model, optimizer, scheduler, finished=False, **loop_vars):
    """Saves everything needed to continue after `epoch`. finished=True marks an early stop."""
    import numpy as np
    import torch

    path = state_path_for(output_checkpoint)
    state = {
        "epoch": epoch,
        "finished": finished,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loop_vars": loop_vars,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, str(tmp))
    os.replace(tmp, path)


def load_training_state(output_checkpoint, model, optimizer, scheduler, device, epochs):
    """Restores a saved state. Returns (start_epoch, loop_vars); (1, {}) if there is nothing to resume."""
    import numpy as np
    import torch

    path = state_path_for(output_checkpoint)
    if not path.is_file():
        print(f"[RESUME] Khong co trang thai {path} -> train tu epoch 1")
        return 1, {}

    state = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state.get("rng") or {}
    try:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    except Exception as exc:  # RNG restore is best effort (e.g. different GPU count)
        print(f"[RESUME] Khong khoi phuc duoc RNG ({exc}) - van tiep tuc")

    done = int(state["epoch"])
    start = epochs + 1 if state.get("finished") else done + 1
    note = " (da early stop, chi chay buoc danh gia cuoi)" if state.get("finished") else ""
    print(f"[RESUME] Tiep tuc tu epoch {start}/{epochs} (da xong {done} epoch){note} <- {path}")
    return start, state.get("loop_vars", {})

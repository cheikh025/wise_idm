"""Print the LR-probe comparison: train-loss trace and per-epoch validation."""
from __future__ import annotations

import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/wise_idm/lr_probe"

print(f"{'lr':>8} {'steps':>6} {'loss@25%':>9} {'loss@50%':>9} {'loss_last':>10}   val_mean_joint_mae / gripper_acc per epoch")
for directory in sorted(glob.glob(os.path.join(root, "tb_lr*"))):
    lr = os.path.basename(directory).removeprefix("tb_lr")
    accumulator = EventAccumulator(directory)
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    if "train/loss" not in tags:
        print(f"{lr:>8}  (no scalars yet)")
        continue
    losses = accumulator.Scalars("train/loss")
    quarter = losses[len(losses) // 4].value
    half = losses[len(losses) // 2].value

    history_path = os.path.join(root, f"lr{lr}", "history.json")
    epochs = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as handle:
            for record in json.load(handle):
                epochs.append(
                    f"e{record['epoch']}: {record['mean_joint_mae']:.5f}/{record['gripper_accuracy']:.4f}"
                )
    print(
        f"{lr:>8} {losses[-1].step:>6} {quarter:>9.4f} {half:>9.4f} {losses[-1].value:>10.4f}   "
        + "  ".join(epochs)
    )

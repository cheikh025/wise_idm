"""M4 training ladder stage 3: checkpoint save/reload verification.

Loads a saved checkpoint fresh into a new model instance and re-evaluates on
the val split, confirming the reloaded metrics match what was recorded
during training (not a new/different model, and no state lost in the
save/load round trip).
"""
import argparse

import torch
from torch.utils.data import DataLoader

from droid_dataset import DroidIDMDataset
from model import DroidIDM
from train import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-episodes", type=int, nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    print(f"checkpoint epoch={ckpt['epoch']} recorded val_metrics={ckpt['val_metrics']}")

    model = DroidIDM(image_size=cfg["image_size"], num_frames=cfg["num_frames"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = DroidIDMDataset(episode_indices=a.val_episodes, image_size=cfg["image_size"])
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False)

    metrics = evaluate(model, val_loader, ckpt["joint_stats"], DEVICE)
    print(f"reloaded, re-evaluated val_metrics={metrics}")

    recorded = ckpt["val_metrics"]
    mae_diff = abs(metrics["mean_joint_mae"] - recorded["mean_joint_mae"])
    acc_diff = abs(metrics["gripper_accuracy"] - recorded["gripper_accuracy"])
    print(f"\ndiff: joint_mae={mae_diff:.6f}, gripper_acc={acc_diff:.6f}")
    ok = mae_diff < 1e-4 and acc_diff < 1e-4
    print("MATCH" if ok else "MISMATCH -- investigate")


if __name__ == "__main__":
    main()

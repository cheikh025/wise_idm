"""M4 training ladder for the DROID IDM.

Stages (per research/bootstrap conventions and build-droid-idm skill):
  1. overfit a tiny batch/window to prove model+labels can fit
  2. train on the debug subset with a held-out val split
  3. checkpoint/reload test
  4. log per-joint error, gripper accuracy, normalized aggregate error

No VAE roundtrip anywhere (hard constraint) -- model.py's backbone is a
plain CNN on raw RGB pixels.

Gripper target is heavily bimodal (measured on the DROID debug subset: 59%
near 0, 29% near 1, 12% mid-transition, mean step-to-step change 0.012) --
trained as binary classification (BCEWithLogitsLoss against the raw [0,1]
target), not SmoothL1 regression in standardized space. Regression on this
signal plateaued (gripper_loss stuck ~0.45 across 300 overfit steps while
joint_loss kept dropping) even on a trivially-memorizable 8-window batch.
Joint channels (continuous) keep standardized SmoothL1 regression.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from droid_dataset import DroidIDMDataset
from model import DroidIDM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_joint_stats(ds: DroidIDMDataset) -> dict:
    """Per-channel mean/std over the 7 joint-position targets only."""
    actions = []
    for i in range(len(ds)):
        actions.append(ds[i]["action"][:, :7].numpy())
    actions = np.concatenate(actions, axis=0)  # (N*chunk_len, 7)
    mean = actions.mean(axis=0)
    std = actions.std(axis=0) + 1e-6
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_joints(joints: torch.Tensor, stats: dict, device) -> torch.Tensor:
    mean = torch.tensor(stats["mean"], device=device)
    std = torch.tensor(stats["std"], device=device)
    return (joints - mean) / std


def denormalize_joints(joints: torch.Tensor, stats: dict, device) -> torch.Tensor:
    mean = torch.tensor(stats["mean"], device=device)
    std = torch.tensor(stats["std"], device=device)
    return joints * std + mean


def loss_fn(joints_pred_norm, gripper_logit_pred, joints_target_norm, gripper_target, gripper_pos_weight=None):
    joint_loss = F.smooth_l1_loss(joints_pred_norm, joints_target_norm)
    gripper_loss = F.binary_cross_entropy_with_logits(
        gripper_logit_pred.squeeze(-1), gripper_target, pos_weight=gripper_pos_weight)
    return joint_loss + gripper_loss, joint_loss.item(), gripper_loss.item()


def compute_gripper_pos_weight(ds: DroidIDMDataset, device) -> torch.Tensor:
    """neg/pos ratio for BCEWithLogitsLoss's pos_weight, to counter the
    measured 59%/29%/12% open/closed/mid class imbalance."""
    vals = []
    for i in range(len(ds)):
        vals.append(ds[i]["action"][:, 7].numpy())
    vals = np.concatenate(vals)
    pos = (vals > 0.5).sum()
    neg = (vals <= 0.5).sum()
    w = neg / max(pos, 1)
    print(f"gripper pos_weight: {w:.3f} (pos={pos}, neg={neg})")
    return torch.tensor(w, device=device)


def evaluate(model, loader, stats, device):
    model.eval()
    per_joint_abs_err = torch.zeros(7, device=device)
    n = 0
    gripper_correct = 0
    gripper_total = 0
    gripper_abs_err = 0.0
    with torch.no_grad():
        for batch in loader:
            wrist, left, right = batch["wrist"].to(device), batch["left"].to(device), batch["right"].to(device)
            proprio = batch["proprio"].to(device)
            target = batch["action"].to(device)
            joints_target, gripper_target = target[..., :7], target[..., 7]

            out = model(wrist, left, right, proprio)
            joints_pred = denormalize_joints(out["joints"], stats, device)
            gripper_prob = torch.sigmoid(out["gripper_logit"].squeeze(-1))

            err = (joints_pred - joints_target).abs()
            per_joint_abs_err += err.mean(dim=(0, 1)) * wrist.shape[0]
            n += wrist.shape[0]

            gripper_pred_bin = (gripper_prob > 0.5).float()
            gripper_tgt_bin = (gripper_target > 0.5).float()
            gripper_correct += (gripper_pred_bin == gripper_tgt_bin).sum().item()
            gripper_total += gripper_tgt_bin.numel()
            gripper_abs_err += (gripper_prob - gripper_target).abs().sum().item()
    model.train()
    return {
        "per_joint_mae": (per_joint_abs_err / n).tolist(),
        "mean_joint_mae": (per_joint_abs_err / n).mean().item(),
        "gripper_mae": gripper_abs_err / gripper_total,
        "gripper_accuracy": gripper_correct / gripper_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["overfit", "train"], default="train")
    p.add_argument("--train-episodes", type=int, nargs="+", default=list(range(22)))
    p.add_argument("--val-episodes", type=int, nargs="+", default=list(range(22, 27)))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", default="/workspace/wise_idm/checkpoints")
    p.add_argument("--log-dir", default="/workspace/wise_idm/tb_logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cnn-width", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-encoder-layers", type=int, default=4)
    p.add_argument("--n-decoder-layers", type=int, default=4)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    writer = SummaryWriter(a.log_dir)

    train_ds = DroidIDMDataset(episode_indices=a.train_episodes, image_size=a.image_size)
    val_ds = DroidIDMDataset(episode_indices=a.val_episodes, image_size=a.image_size)
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    print("computing joint normalization stats from train split ...")
    stats = compute_joint_stats(train_ds)
    print("mean:", np.round(stats["mean"], 3))
    print("std:", np.round(stats["std"], 3))

    model = DroidIDM(image_size=a.image_size, num_frames=train_ds.num_frames,
                      cnn_width=a.cnn_width, d_model=a.d_model, n_heads=a.n_heads,
                      n_encoder_layers=a.n_encoder_layers, n_decoder_layers=a.n_decoder_layers).to(DEVICE)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M, spatial tokens: {model.num_spatial_tokens}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    gripper_pos_weight = compute_gripper_pos_weight(train_ds, DEVICE) if a.mode == "train" else None

    if a.mode == "overfit":
        loader = DataLoader(train_ds, batch_size=min(a.batch_size, len(train_ds)), shuffle=False)
        batch = next(iter(loader))
        wrist, left, right = batch["wrist"].to(DEVICE), batch["left"].to(DEVICE), batch["right"].to(DEVICE)
        proprio = batch["proprio"].to(DEVICE)
        target = batch["action"].to(DEVICE)
        joints_target, gripper_target = target[..., :7], target[..., 7]
        joints_target_norm = normalize_joints(joints_target, stats, DEVICE)

        t0 = time.time()
        for step in range(300):
            opt.zero_grad()
            out = model(wrist, left, right, proprio)
            joints_pred_norm = out["joints"]
            loss, jl, gl = loss_fn(joints_pred_norm, out["gripper_logit"], joints_target_norm, gripper_target)
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == 299:
                joints_pred = denormalize_joints(joints_pred_norm, stats, DEVICE)
                joint_mae = (joints_pred - joints_target).abs().mean().item()
                gripper_prob = torch.sigmoid(out["gripper_logit"].squeeze(-1))
                gripper_acc = ((gripper_prob > 0.5).float() == (gripper_target > 0.5).float()).float().mean().item()
                print(f"step {step:4d}  loss={loss.item():.5f}  joint_loss={jl:.5f}  gripper_bce={gl:.5f}  "
                      f"joint_mae={joint_mae:.5f}  gripper_acc={gripper_acc:.3f}  ({time.time()-t0:.1f}s)")
                writer.add_scalar("overfit/loss", loss.item(), step)
                writer.add_scalar("overfit/joint_mae", joint_mae, step)
                writer.add_scalar("overfit/gripper_acc", gripper_acc, step)
        print(f"overfit test done in {time.time()-t0:.1f}s")
        return

    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    total_steps = a.epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps, anneal_strategy="cos")

    global_step = 0
    best_val_mae = float("inf")
    history = []
    for epoch in range(a.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            wrist, left, right = batch["wrist"].to(DEVICE), batch["left"].to(DEVICE), batch["right"].to(DEVICE)
            proprio = batch["proprio"].to(DEVICE)
            target = batch["action"].to(DEVICE)
            joints_target, gripper_target = target[..., :7], target[..., 7]
            joints_target_norm = normalize_joints(joints_target, stats, DEVICE)

            opt.zero_grad()
            out = model(wrist, left, right, proprio)
            loss, jl, gl = loss_fn(out["joints"], out["gripper_logit"], joints_target_norm, gripper_target,
                                    gripper_pos_weight=gripper_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            epoch_loss += loss.item()
            n_batches += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/joint_loss", jl, global_step)
            writer.add_scalar("train/gripper_bce", gl, global_step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], global_step)
            global_step += 1

        val_metrics = evaluate(model, val_loader, stats, DEVICE)
        epoch_time = time.time() - t0
        print(f"epoch {epoch:3d}  train_loss={epoch_loss/n_batches:.5f}  "
              f"val_mean_joint_mae={val_metrics['mean_joint_mae']:.5f}  "
              f"val_gripper_acc={val_metrics['gripper_accuracy']:.3f}  ({epoch_time:.1f}s)")
        writer.add_scalar("val/mean_joint_mae", val_metrics["mean_joint_mae"], epoch)
        writer.add_scalar("val/gripper_accuracy", val_metrics["gripper_accuracy"], epoch)
        history.append({"epoch": epoch, "train_loss": epoch_loss / n_batches, **val_metrics})

        if val_metrics["mean_joint_mae"] < best_val_mae:
            best_val_mae = val_metrics["mean_joint_mae"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "joint_stats": stats,
                "config": {
                    "image_size": a.image_size, "num_frames": train_ds.num_frames,
                    "chunk_len": train_ds.chunk_len, "cameras": ["wrist", "left", "right"],
                },
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(a.out_dir, "best.pt"))
            print(f"  -> saved new best checkpoint (val_mean_joint_mae={best_val_mae:.5f})")

    with open(os.path.join(a.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("training complete")


if __name__ == "__main__":
    main()

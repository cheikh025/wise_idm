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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from droid_dataset import DroidIDMDataset
from model import DroidIDM
from model_v2 import DroidIDMv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_stats_fast(ds: DroidIDMDataset) -> tuple[dict, float, int, int]:
    """Joint mean/std + gripper pos/neg counts, read directly from the
    already-loaded parquet action columns (ds.data / ds.windows) instead of
    through ds[i] -- __getitem__ always decodes all 3 cameras' video for a
    window even though only the 8-dim action column is needed here, which
    made this pass over the full dataset (~40k+ windows at 5000-episode
    scale) dominated by needless video I/O. Group-by-episode once, then
    slice each window's action rows with numpy -- no video touched."""
    per_episode = {}
    for ep, g in ds.data.groupby("episode_index"):
        g = g.sort_values("frame_index")
        joints = np.stack(g["action.joint_position"].to_numpy())
        gripper = g["action.gripper_position"].to_numpy().astype(np.float32)
        per_episode[ep] = (joints, gripper)

    all_joints, all_gripper = [], []
    for w in ds.windows:
        joints, gripper = per_episode[w.episode_index]
        all_joints.append(joints[w.chunk_start:w.chunk_start + ds.chunk_len])
        all_gripper.append(gripper[w.chunk_start:w.chunk_start + ds.chunk_len])
    all_joints = np.concatenate(all_joints, axis=0)
    all_gripper = np.concatenate(all_gripper, axis=0)

    mean = all_joints.mean(axis=0)
    std = all_joints.std(axis=0) + 1e-6
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    pos = int((all_gripper > 0.5).sum())
    neg = int((all_gripper <= 0.5).sum())
    pos_weight = neg / max(pos, 1)
    return stats, pos_weight, pos, neg


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


def call_model(model, arch: str, batch, device, cameras: list[str]):
    views = [batch[cam].to(device) for cam in cameras]
    if arch == "v2":
        return model(*views)  # model_v2.DroidIDMv2 always takes positional wrist,left,right -- 3-view only
    return model(views)


def evaluate(model, loader, stats, device, arch: str = "v1", cameras: list[str] = ("wrist", "left", "right")):
    model.eval()
    per_joint_abs_err = torch.zeros(7, device=device)
    n = 0
    gripper_correct = 0
    gripper_total = 0
    gripper_abs_err = 0.0
    with torch.no_grad():
        for batch in loader:
            target = batch["action"].to(device)
            joints_target, gripper_target = target[..., :7], target[..., 7]

            out = call_model(model, arch, batch, device, cameras)
            joints_pred = denormalize_joints(out["joints"], stats, device)
            gripper_prob = torch.sigmoid(out["gripper_logit"].squeeze(-1))

            err = (joints_pred - joints_target).abs()
            n_batch = batch["action"].shape[0]
            per_joint_abs_err += err.mean(dim=(0, 1)) * n_batch
            n += n_batch

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
    p.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="/workspace/wise_idm/checkpoints")
    p.add_argument("--log-dir", default="/workspace/wise_idm/tb_logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cnn-width", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-encoder-layers", type=int, default=4)
    p.add_argument("--n-decoder-layers", type=int, default=4)
    p.add_argument("--arch", choices=["v1", "v2"], default="v1",
                    help="v1: flatten-to-grid tokens (original). v2: spatial-softmax token compression (RUN_0015 redesign, 3-view only).")
    p.add_argument("--num-keypoints", type=int, default=48, help="v2 only: spatial-softmax keypoints per camera-pair")
    p.add_argument("--cameras", nargs="+", default=["wrist", "left", "right"], choices=["wrist", "left", "right"],
                    help="v1 only: which camera views to feed the model (v2 is always all 3). "
                         "e.g. --cameras left  for a single-exterior-view ablation.")
    a = p.parse_args()
    if a.arch == "v2" and a.cameras != ["wrist", "left", "right"]:
        raise ValueError("--cameras is not supported with --arch v2 (fixed 3-view forward signature)")

    # Multi-GPU via torchrun: LOCAL_RANK/WORLD_SIZE are set by the launcher,
    # unset (defaulting to single-process) under a plain `python3 train.py`.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    is_main = (not distributed) or dist.get_rank() == 0

    torch.manual_seed(a.seed)
    if is_main:
        os.makedirs(a.out_dir, exist_ok=True)
    writer = SummaryWriter(a.log_dir) if is_main else None

    train_ds = DroidIDMDataset(episode_indices=a.train_episodes, image_size=a.image_size)
    val_ds = DroidIDMDataset(episode_indices=a.val_episodes, image_size=a.image_size)
    if is_main:
        print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}, world_size={world_size}")
        print("computing joint normalization stats + gripper pos_weight (parquet-only, no video decode) ...")
    stats, pos_weight_val, pos, neg = compute_stats_fast(train_ds)
    if is_main:
        print("mean:", np.round(stats["mean"], 3))
        print("std:", np.round(stats["std"], 3))
        print(f"gripper pos_weight: {pos_weight_val:.3f} (pos={pos}, neg={neg})")

    if a.arch == "v2":
        model = DroidIDMv2(image_size=a.image_size, num_frames=train_ds.num_frames,
                            cnn_width=a.cnn_width, num_keypoints=a.num_keypoints, d_model=a.d_model,
                            n_heads=a.n_heads, n_encoder_layers=a.n_encoder_layers,
                            n_decoder_layers=a.n_decoder_layers).to(device)
        n_tokens_desc = f"{3 * model.num_pairs} tokens (vs v1's {3 * model.num_pairs * 16})"
    else:
        model = DroidIDM(image_size=a.image_size, num_frames=train_ds.num_frames, num_cameras=len(a.cameras),
                          cnn_width=a.cnn_width, d_model=a.d_model, n_heads=a.n_heads,
                          n_encoder_layers=a.n_encoder_layers, n_decoder_layers=a.n_decoder_layers).to(device)
        n_tokens_desc = f"{model.num_spatial_tokens} spatial tokens/pair x {len(a.cameras)} camera(s) ({a.cameras})"
    n_params = sum(p_.numel() for p_ in model.parameters())
    if is_main:
        print(f"arch={a.arch}  model params: {n_params/1e6:.2f}M  {n_tokens_desc}")
    if distributed:
        # Broadcasts rank 0's initial weights to all ranks at construction,
        # then all-reduces gradients each backward() so every rank's
        # optimizer applies an identical update -- no manual param sync needed.
        model = DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    gripper_pos_weight = torch.tensor(pos_weight_val, device=device) if a.mode == "train" else None

    if a.mode == "overfit":
        loader = DataLoader(train_ds, batch_size=min(a.batch_size, len(train_ds)), shuffle=False)
        batch = next(iter(loader))
        target = batch["action"].to(device)
        joints_target, gripper_target = target[..., :7], target[..., 7]
        joints_target_norm = normalize_joints(joints_target, stats, device)

        t0 = time.time()
        for step in range(300):
            opt.zero_grad()
            out = call_model(model, a.arch, batch, device, a.cameras)
            joints_pred_norm = out["joints"]
            loss, jl, gl = loss_fn(joints_pred_norm, out["gripper_logit"], joints_target_norm, gripper_target)
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == 299:
                joints_pred = denormalize_joints(joints_pred_norm, stats, device)
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

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=a.seed)
        train_loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=train_sampler,
                                   num_workers=a.num_workers, persistent_workers=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                                   num_workers=a.num_workers, persistent_workers=True)
    val_loader = None
    if is_main:
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
        if distributed:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            target = batch["action"].to(device)
            joints_target, gripper_target = target[..., :7], target[..., 7]
            joints_target_norm = normalize_joints(joints_target, stats, device)

            opt.zero_grad()
            out = call_model(model, a.arch, batch, device, a.cameras)
            loss, jl, gl = loss_fn(out["joints"], out["gripper_logit"], joints_target_norm, gripper_target,
                                    gripper_pos_weight=gripper_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()

            epoch_loss += loss.item()
            n_batches += 1
            if is_main:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/joint_loss", jl, global_step)
                writer.add_scalar("train/gripper_bce", gl, global_step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], global_step)
            global_step += 1

        epoch_time = time.time() - t0
        if is_main:
            eval_model = model.module if distributed else model
            val_metrics = evaluate(eval_model, val_loader, stats, device, arch=a.arch, cameras=a.cameras)
            print(f"epoch {epoch:3d}  train_loss={epoch_loss/n_batches:.5f}  "
                  f"val_mean_joint_mae={val_metrics['mean_joint_mae']:.5f}  "
                  f"val_gripper_acc={val_metrics['gripper_accuracy']:.3f}  ({epoch_time:.1f}s)")
            writer.add_scalar("val/mean_joint_mae", val_metrics["mean_joint_mae"], epoch)
            writer.add_scalar("val/gripper_accuracy", val_metrics["gripper_accuracy"], epoch)
            history.append({"epoch": epoch, "train_loss": epoch_loss / n_batches, **val_metrics})

            if val_metrics["mean_joint_mae"] < best_val_mae:
                best_val_mae = val_metrics["mean_joint_mae"]
                ckpt = {
                    "model_state_dict": eval_model.state_dict(),
                    "joint_stats": stats,
                    "config": {
                        "arch": a.arch,
                        "image_size": a.image_size, "num_frames": train_ds.num_frames,
                        "chunk_len": train_ds.chunk_len, "cameras": a.cameras,
                        "cnn_width": a.cnn_width, "d_model": a.d_model, "n_heads": a.n_heads,
                        "n_encoder_layers": a.n_encoder_layers, "n_decoder_layers": a.n_decoder_layers,
                        "num_keypoints": a.num_keypoints if a.arch == "v2" else None,
                    },
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }
                torch.save(ckpt, os.path.join(a.out_dir, "best.pt"))
                print(f"  -> saved new best checkpoint (val_mean_joint_mae={best_val_mae:.5f})")
        if distributed:
            dist.barrier()

    if is_main:
        with open(os.path.join(a.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print("training complete")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

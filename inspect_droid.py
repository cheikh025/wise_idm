#!/usr/bin/env python3
"""M4 step 1: inspect the DROID debug subset (chunk-000/file-000, success split).

Verifies: episode structure, correct commanded-action fields (joint_position,
NOT joint_velocity/cartesian), frame/state/action temporal alignment, and
video-vs-parquet frame count agreement.
"""
import cv2
import numpy as np
import pandas as pd

BASE = "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/5c11a20accb11497270a5247a7f1e66ad04c956c/success"

df = pd.read_parquet(f"{BASE}/data/chunk-000/file-000.parquet")
print("columns:", list(df.columns))
print("rows:", len(df))
print("dtypes:\n", df.dtypes)

episodes = df["episode_index"].unique()
print(f"\nepisodes in this file: {len(episodes)}")
print("episode indices (first 10):", sorted(episodes)[:10])

ep0 = sorted(episodes)[0]
ep_df = df[df["episode_index"] == ep0].sort_values("frame_index")
print(f"\n=== episode {ep0}: {len(ep_df)} frames ===")
print("frame_index range:", ep_df["frame_index"].min(), "-", ep_df["frame_index"].max())
print("timestamp range:", ep_df["timestamp"].min(), "-", ep_df["timestamp"].max())
ts = ep_df["timestamp"].to_numpy()
dts = np.diff(ts)
print(f"timestamp deltas: mean={dts.mean():.5f} std={dts.std():.5f} (expect ~1/15={1/15:.5f})")

jp = np.stack(ep_df["action.joint_position"].to_numpy())
jv = np.stack(ep_df["action.joint_velocity"].to_numpy())
gp = np.stack(ep_df["action.gripper_position"].to_numpy())
cp = np.stack(ep_df["action.cartesian_position"].to_numpy())
sp = np.stack(ep_df["observation.state.joint_positions"].to_numpy())
sg = np.stack(ep_df["observation.state.gripper_position"].to_numpy())

print(f"\naction.joint_position shape={jp.shape} range=[{jp.min():.3f},{jp.max():.3f}]")
print(f"action.joint_velocity shape={jv.shape} range=[{jv.min():.3f},{jv.max():.3f}]  <- must differ from joint_position")
print(f"action.gripper_position shape={gp.shape} range=[{gp.min():.3f},{gp.max():.3f}]")
print(f"action.cartesian_position shape={cp.shape} range=[{cp.min():.3f},{cp.max():.3f}]  <- different quantity entirely (6-D pose)")
print(f"observation.state.joint_positions shape={sp.shape} range=[{sp.min():.3f},{sp.max():.3f}]")
print(f"observation.state.gripper_position shape={sg.shape} range=[{sg.min():.3f},{sg.max():.3f}]")

# Sanity: commanded joint_position should track close to the *next* observed
# state.joint_positions (it's a target for the following step), not be
# identical to the *current* one, and should NOT numerically equal joint_velocity.
diff_immediate = np.abs(jp - sp).mean()
diff_next = np.abs(jp[:-1] - sp[1:]).mean()
print(f"\nmean |action.joint_position - CURRENT state.joint_positions| = {diff_immediate:.4f}")
print(f"mean |action.joint_position - NEXT state.joint_positions|    = {diff_next:.4f}  (expect smaller if action leads state by 1 step)")

same_as_velocity = np.allclose(jp, jv, atol=1e-3)
print(f"\naction.joint_position == action.joint_velocity (should be False): {same_as_velocity}")

# Video frame count check
wrist_path = f"{BASE}/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4"
cap = cv2.VideoCapture(wrist_path)
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"\nwrist video (whole file, all episodes concatenated): frames={total_video_frames} fps={fps} size={w}x{h}")
print(f"parquet total rows in this file: {len(df)} (expect == video frame count if 1 file = 1 concatenated video)")

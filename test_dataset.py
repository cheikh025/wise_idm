from droid_dataset import DroidIDMDataset
import time

t0 = time.time()
ds = DroidIDMDataset(
    episode_indices=list(range(27)),
    input_height=128,
    input_width=224,
    stride=16,
)
print(f"dataset built in {time.time()-t0:.2f}s, {len(ds)} windows from 27 episodes")

t0 = time.time()
sample = ds[0]
print(f"sample 0 loaded in {time.time()-t0:.2f}s")
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} range=[{v.min().item():.3f},{v.max().item():.3f}]")
    else:
        print(f"  {k}: {v}")

# spot check a window from a later episode
sample2 = ds[len(ds) // 2]
print(f"\nmid-dataset sample: episode={sample2['episode_index']} chunk_start={sample2['chunk_start']}")
print(f"  action shape: {tuple(sample2['action'].shape)}")
print(f"  dataset split: {sample2['dataset_split']}")

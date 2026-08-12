import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE = "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/5c11a20accb11497270a5247a7f1e66ad04c956c/success"
meta = pd.read_parquet(f"{BASE}/meta/episodes/chunk-000/file-000.parquet")

cols = ["episode_index", "length", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index",
        "videos/observation.image.wrist_image_left/chunk_index",
        "videos/observation.image.wrist_image_left/file_index",
        "videos/observation.image.wrist_image_left/from_timestamp",
        "videos/observation.image.wrist_image_left/to_timestamp"]
print(meta[cols].head(10).to_string())
print()
print("unique wrist video file_index values in this meta chunk:",
      sorted(meta["videos/observation.image.wrist_image_left/file_index"].unique())[:20])
print("how many episodes point at wrist video chunk=0,file=0:",
      ((meta["videos/observation.image.wrist_image_left/chunk_index"] == 0) &
       (meta["videos/observation.image.wrist_image_left/file_index"] == 0)).sum())

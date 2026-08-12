import pandas as pd

BASE = "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/5c11a20accb11497270a5247a7f1e66ad04c956c/success"

meta = pd.read_parquet(f"{BASE}/meta/episodes/chunk-000/file-000.parquet")
print("columns:", list(meta.columns))
print("rows:", len(meta))
print(meta.iloc[0].to_dict())
print()
print(meta.head(5).to_string())

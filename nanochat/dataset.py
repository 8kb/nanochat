"""
The base/pretraining dataset is a set of parquet files. This file owns this fork's dataset
*identity* (which URL, which shard count, which local directory) -- the download mechanism lives
in datacore.download, which knows nothing about ClimbMix and never will. See
datacore/docs/architecture.md.

`parquets_iter_batched` stays here rather than moving to datacore: it iterates row GROUPS with a
DDP start/step stride restarting at every file (unlike datacore.sources.ParquetDirectorySource,
whose one-batch-per-file granularity exists specifically to give DataManager.prepare a volume
flush boundary) and both its callers -- scripts/tok_train.py, scripts/tok_eval.py -- run before
any tokenizer (and therefore any DataManager) exists at all.

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import argparse
import os

import pyarrow.parquet as pq

from datacore.download import download_shards

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames


def data_dir():
    """Not a module-level constant -- computed on every call so NANOCHAT_BASE_DIR can be set
    (e.g. by a test) without needing to import this module in a particular order (the previous
    version computed this at import time, which was exactly this footgun)."""
    return os.path.join(get_base_dir(), "base_data_climbmix")


_default_data_dir = data_dir  # alias so list_parquet_files's `data_dir=` param can shadow the name


# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    dir_path = _default_data_dir() if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if not os.path.exists(dir_path):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {dir_path}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        dir_path = os.path.join(get_base_dir(), "base_data")

    parquet_files = sorted([
        f for f in os.listdir(dir_path)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(dir_path, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    dest_dir = data_dir()
    os.makedirs(dest_dir, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # always download the validation shard

    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {dest_dir}")
    print()
    result = download_shards(
        BASE_URL + "/shard_{index:05d}.parquet", ids_to_download, dest_dir,
        filename_fn=index_to_filename, num_workers=args.num_workers,
        log=print,
    )
    print(f"Done! Downloaded: {result['successful']}/{result['total']} shards to {dest_dir}")

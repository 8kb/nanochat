"""
download.py: a generic resumable HTTP shard downloader -- the mechanism nanochat/dataset.py's
download_single_file used to own inline, generalized to any URL-template + index-range corpus.
Uses stdlib urllib.request, not the `requests` package (nanochat's own tasks/common.py already
does the same; `requests` is imported-but-undeclared debt elsewhere in this repo -- see
docs/roadmap.md's "Explicitly deferred" -- and a brand-new standalone package shouldn't add a
dependency it doesn't need).
"""
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def download_file(url: str, dest_path: str, *, max_attempts: int = 5, chunk_size: int = 1024 * 1024) -> bool:
    """Downloads one file, skipping if it already exists, retrying with exponential backoff, and
    writing through a `.tmp` sibling + rename so a failed/interrupted download can never be
    mistaken for a complete one."""
    if os.path.exists(dest_path):
        return True
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp_path = dest_path + ".tmp"
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response, open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp_path, dest_path)
            return True
        except (urllib.error.URLError, OSError):
            for path in (tmp_path, dest_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_shards(url_template: str, indices, dest_dir: str, *, filename_fn=None,
                     num_workers: int = 4, log=lambda msg: None) -> dict:
    """Downloads `url_template.format(index=i)` for every i in `indices` into `dest_dir`, in
    parallel. `filename_fn(index) -> str` names the local file (default: the URL's basename).
    Returns {"successful": n, "total": n}."""
    os.makedirs(dest_dir, exist_ok=True)
    indices = list(indices)

    def _one(index):
        url = url_template.format(index=index)
        filename = filename_fn(index) if filename_fn else url.rsplit("/", 1)[-1]
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            log(f"Skipping {dest_path} (already exists)")
            return True
        log(f"Downloading {filename}...")
        ok = download_file(url, dest_path)
        log(f"{'Successfully downloaded' if ok else 'Failed to download'} {filename}")
        return ok

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(_one, indices))
    return {"successful": sum(results), "total": len(indices)}

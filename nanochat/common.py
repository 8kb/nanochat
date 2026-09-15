"""
Common utilities for nanochat.

Device/DDP/seed bring-up (compute_init/compute_cleanup/is_ddp_requested/is_ddp_initialized/
get_dist_info/autodetect_device_type) and the GPU peak-FLOPs/bandwidth tables (get_peak_flops/
get_peak_bandwidth) now live in modelcore.runtime -- moved there because tinylab carried an
identical copy of the former, and dropped the latter entirely wanting it back (see modelcore's own
docs/architecture.md). Re-exported here so every existing `from nanochat.common import ...` call
site keeps working unchanged, same shim pattern as nanochat/optim.py and nanochat/flash_attention.py.
"""

import os
import re
import logging
import urllib.request
import torch
from filelock import FileLock
from modelcore.runtime import DEFAULT_RUNTIME
from modelcore.runtime import compute_cleanup, get_dist_info, is_ddp_initialized, is_ddp_requested  # noqa: F401
from modelcore.runtime import compute_init as _compute_init
from modelcore.runtime import autodetect_device_type as _autodetect_device_type
from modelcore.runtime import peak_bandwidth as _peak_bandwidth, peak_flops as _peak_flops

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
# Detection lives in modelcore.runtime (modelcore has no nanochat dependencies at all) -- this is
# just a convenience re-export of the process-wide default runtime's value.
COMPUTE_DTYPE = DEFAULT_RUNTIME.compute_dtype
COMPUTE_DTYPE_REASON = DEFAULT_RUNTIME.compute_dtype_reason

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def autodetect_device_type():
    device_type = _autodetect_device_type(log=print0)
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common. Mechanism lives in
    modelcore.runtime.compute_init (see this module's own docstring); this wrapper only adds the
    two things that were nanochat-specific about the original: routing autodetection's message
    through print0 (rank-0-only), and logging the distributed world size on rank 0."""
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
    is_ddp_req, ddp_rank, ddp_local_rank, ddp_world_size, device = _compute_init(device_type, log=print0)
    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")
    return is_ddp_req, ddp_rank, ddp_local_rank, ddp_world_size, device

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def get_peak_flops(device_name: str) -> float:
    return _peak_flops(device_name, log=logger.warning)

def get_peak_bandwidth(device_name: str) -> float:
    return _peak_bandwidth(device_name, log=logger.warning)

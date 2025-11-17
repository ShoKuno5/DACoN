import logging
import os
from pathlib import Path

from .image_process import *
from .data_process import *
from .metrics import *
from .pooling import *

import yaml

# Allow GeoAware-SC's `utils` directory to be discovered under the same package
_geo_candidates = []
_base_path = Path(__file__).resolve()
for parent_offset in (2, 3, 4):
    try:
        candidate = _base_path.parents[parent_offset] / "GeoAware-SC" / "utils"
    except IndexError:
        continue
    _geo_candidates.append(candidate)

geo_env = os.environ.get("GEOAWARE_REPO")
if geo_env:
    _geo_candidates.insert(0, Path(geo_env).expanduser().resolve() / "utils")

for _geo_utils_dir in _geo_candidates:
    if _geo_utils_dir.is_dir():
        _geo_utils_str = str(_geo_utils_dir)
        if _geo_utils_str not in __path__:
            __path__.append(_geo_utils_str)

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(save_path, current_time, log_name):
    log_filename = os.path.join(save_path, f"{log_name}_{current_time}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

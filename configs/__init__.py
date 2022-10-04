from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / Path('out')
LOG_CONFIG = ROOT_DIR / Path('configs/logger_conf.json')
MATLAB_CONFIG = ROOT_DIR / Path('configs/matplotlib_conf.json')

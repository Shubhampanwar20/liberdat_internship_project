import os
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TS_FILE = os.path.join(DATA_DIR, "timeseries-data.json")
CLIN_FILE = os.path.join(DATA_DIR, "categorical-data.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
HOLDOUT_DAYS = 7
FORECAST_DAYS = 365

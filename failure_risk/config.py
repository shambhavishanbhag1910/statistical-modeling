from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "synthetic"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
REPORT_DIR = ROOT_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"
TABLE_DIR = REPORT_DIR / "tables"

DEFAULT_DATA_PATH = DATA_DIR / "fleet_survival.csv"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "model_bundle.pkl"
DEFAULT_METRICS_PATH = ARTIFACT_DIR / "metrics.json"
DEFAULT_METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

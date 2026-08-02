from pathlib import Path
import os
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path("/tmp/stock-market-forecast") if os.environ.get("VERCEL") else BASE_DIR
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
DB_PATH = RUNTIME_DIR / "market_forecast.db"
RAW_DATA_PATH = RUNTIME_DIR / "market_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
MODEL_PATH = SAVED_MODELS_DIR / "cnn_bilstm_model.keras"
SCALER_PATH = SAVED_MODELS_DIR / "scaler.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SUMMARY_PATH = REPORTS_DIR / "summary.txt"

DEFAULT_SYMBOL = "^NSEI"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
SEQUENCE_LENGTH = 60
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

for directory in [DATA_DIR, MODELS_DIR, SAVED_MODELS_DIR, PLOTS_DIR, REPORTS_DIR, NOTEBOOKS_DIR, APP_DIR, TEMPLATES_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

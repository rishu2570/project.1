from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from config import DEFAULT_SYMBOL, PLOTS_DIR, REPORTS_DIR
from predict import predict_movement
from utils.data_utils import get_history, init_db

app = FastAPI(title="Stock Market Forecast API")
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "app" / "templates" / "index.html"


@app.on_event("startup")
def initialize_database() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/predict")
def predict(symbol: str = Query(DEFAULT_SYMBOL)) -> dict:
    return predict_movement(symbol=symbol, source="web")


@app.get("/history")
def history() -> list[dict]:
    return get_history()


@app.get("/plots")
def plots() -> list[str]:
    return sorted(path.name for path in PLOTS_DIR.glob("*.png"))


@app.get("/plots/{filename}")
def plot_file(filename: str) -> FileResponse:
    path = PLOTS_DIR / filename
    if path.suffix != ".png" or not path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path)


@app.get("/reports")
def reports() -> list[str]:
    return sorted(path.name for path in REPORTS_DIR.glob("*.*") if path.suffix in {".json", ".txt"})


@app.get("/reports/{filename}")
def report_file(filename: str) -> FileResponse:
    path = REPORTS_DIR / filename
    if path.suffix not in {".json", ".txt"} or not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)

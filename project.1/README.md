# Stock Market Index Movement Forecasting Using Multiple Indicator and Deep Learning

This project builds a production-ready Python application for forecasting whether the next day movement of a stock market index will be UP or DOWN using the NIFTY 50 index, technical indicators, and a hybrid CNN + BiLSTM model.

## Features
- Downloads market data via yfinance
- Computes 10 technical indicators
- Preprocesses and normalizes data
- Trains a hybrid CNN + BiLSTM model
- Evaluates using accuracy, precision, recall, F1, ROC-AUC, and confusion metrics
- Provides a FastAPI dashboard and REST APIs
- Stores predictions and requests in SQLite
- Generates plots and reports

## Project Structure
- data/ - input and processed datasets
- notebooks/ - analysis notebooks
- models/ - source model definitions
- saved_models/ - trained Keras and scaler artifacts
- reports/ - evaluation reports
- plots/ - visualization assets
- utils/ - shared helpers and database utilities
- app/templates/ - dashboard template

## Installation
```bash
pip install -r requirements.txt
```

## Run locally
```bash
uvicorn main:app --reload
```

The app will be available at http://127.0.0.1:5000.

## Deploy on Render
1. Push this project to GitHub.
2. Create a new Web Service on Render.
3. Connect the repository.
4. Use these settings:
   - Build Command: pip install -r requirements.txt
   - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
5. Deploy.

The project includes [Procfile](Procfile) for process-based hosts. Vercel detects the FastAPI `app` in `main.py` automatically.

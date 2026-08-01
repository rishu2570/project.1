from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request, send_from_directory

from config import DEFAULT_SYMBOL, PLOTS_DIR, REPORTS_DIR
from predict import predict_movement
from utils.data_utils import get_history


def register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["GET", "POST"])
    def predict_route():
        symbol = request.args.get("symbol", DEFAULT_SYMBOL) if request.method == "GET" else request.form.get("symbol", DEFAULT_SYMBOL)
        result = predict_movement(symbol=symbol, source="web")
        return jsonify(result)

    @app.route("/history")
    def history():
        return jsonify(get_history())

    @app.route("/plots")
    def plots():
        files = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
        return jsonify(files)

    @app.route("/plots/<path:filename>")
    def plot_file(filename: str):
        return send_from_directory(PLOTS_DIR, filename)

    @app.route("/reports")
    def reports():
        files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith((".json", ".txt"))])
        return jsonify(files)

    @app.route("/reports/<path:filename>")
    def report_file(filename: str):
        return send_from_directory(REPORTS_DIR, filename)

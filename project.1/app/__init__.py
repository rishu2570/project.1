from flask import Flask
from config import TEMPLATES_DIR, STATIC_DIR


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
    app.config["SECRET_KEY"] = "stock-market-forecast"
    from app.routes import register_routes
    register_routes(app)
    return app

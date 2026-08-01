from __future__ import annotations

import os

from app import create_app
from train import train_pipeline
from utils.data_utils import init_db

app = create_app()


def main() -> None:
    init_db()
    train_pipeline("^NSEI")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)


if __name__ == "__main__":
    main()

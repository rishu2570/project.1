from pathlib import Path


def test_project_files_exist():
    base = Path(__file__).resolve().parents[1]
    required = [
        "main.py",
        "train.py",
        "predict.py",
        "model.py",
        "preprocess.py",
        "indicators.py",
        "requirements.txt",
        "README.md",
    ]
    for item in required:
        assert (base / item).exists()

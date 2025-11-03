# agent/agent.py
from pathlib import Path
from agent.tools import (
    load_csv_from_path,
    basic_eda,
    detect_task,
    train_baseline,
)


def run_pipeline(csv_path: str, target: str | None = None) -> dict:
    """
    Главный сценарий:
    - загрузить csv
    - сделать EDA
    - определить задачу
    - обучить бейзлайн (если возможно)
    - вернуть всё одним json
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = load_csv_from_path(csv_path)
    eda = basic_eda(df)
    task_info = detect_task(df, target=target)

    model_report = None
    if task_info["task"] != "eda" and task_info["target"]:
        model_report = train_baseline(
            df,
            target=task_info["target"],
            task=task_info["task"],
        )

    return {
        "source": str(csv_path),
        "eda": eda,
        "task": task_info,
        "model": model_report,
    }


if __name__ == "__main__":
    # для локального теста: подставь сюда свой csv
    report = run_pipeline("data/raw/sample.csv")
    print(report)

# api/app.py

# api/app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO

from agent.tools import basic_eda, detect_task, train_baseline

app = FastAPI(title="Nikita DS Agent")


# ---------- вспомогалки ----------

def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    """
    Читаем CSV максимально терпеливо.
    Пробуем разные разделители и кодировки.
    """
    bio = BytesIO(file_bytes)

    variants = [
        {},  # стандарт: utf-8, ','
        {"sep": ";"},
        {"encoding": "utf-8-sig"},
        {"encoding": "windows-1251"},
        {"sep": ";", "encoding": "windows-1251"},
        {"encoding": "latin-1"},          # <-- добавили
        {"sep": ";", "encoding": "latin-1"},  # <-- добавили
    ]

    for kwargs in variants:
        try:
            bio.seek(0)
            return pd.read_csv(bio, on_bad_lines="skip", **kwargs)
        except Exception:
            continue

    # крайний вариант: читаем, игнорируя битые символы
    bio.seek(0)
    return pd.read_csv(
        bio,
        on_bad_lines="skip",
        encoding="latin-1",
        errors="ignore",
    )


    # 3) если всё равно не получилось — пусть pandas кинет нормальную ошибку
    bio.seek(0)
    return pd.read_csv(bio)


def build_report(eda: dict, task: dict, model: dict | None) -> str:
    """
    Собираем человекочитаемый отчёт из того, что мы насчитали.
    """
    rows, cols = eda["shape"]
    lines: list[str] = []

    lines.append(f"📊 В датасете {rows} строк и {cols} колонок.")
    lines.append("Типы колонок:")
    for name, dt in eda["dtypes"].items():
        lines.append(f"  • {name}: {dt}")

    nulls = eda["nulls"]
    has_nulls = any(v > 0 for v in nulls.values())
    if has_nulls:
        lines.append("Пропуски обнаружены:")
        for name, v in nulls.items():
            if v > 0:
                lines.append(f"  • {name}: {v}")
    else:
        lines.append("Пропусков нет.")

    # про задачу
    if task["task"] == "eda":
        lines.append("🤖 Подходящую целевую колонку не нашёл, сделал только EDA.")
    else:
        lines.append(
            f'🧠 Определена задача: {task["task"]} по колонке "{task["target"]}".'
        )

    # про модель
    if model:
        if "accuracy" in model:
            lines.append(
                f'📈 Базовая модель: {model["model_type"]}, accuracy={model["accuracy"]:.3f}, f1={model["f1"]:.3f}'
            )
        elif "rmse" in model:
            lines.append(
                f'📈 Базовая модель: {model["model_type"]}, RMSE={model["rmse"]:.3f}'
            )
    else:
        lines.append("📦 Модель не обучалась (нечего было предсказывать).")

    return "\n".join(lines)


# ---------- эндпоинты ----------

@app.get("/")
def root():
    return {"msg": "Nikita DS Agent is running"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    try:
        contents = await file.read()

        # читаем CSV
        df = read_csv_safely(contents)

        # считаем EDA
        eda = basic_eda(df)

        # определяем задачу
        task = detect_task(df, target=target)

        # пробуем обучить, если есть что
        model_res = None
        if task["task"] != "eda" and task["target"]:
            model_res = train_baseline(df, task["target"], task["task"])

        # собираем красивый текст
        report_text = build_report(eda, task, model_res)

        return JSONResponse(
            {
                "filename": file.filename,
                "eda": eda,
                "task": task,
                "model": model_res,
                "report": report_text,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "failed_to_process_file",
                "details": str(e),
                "hint": "проверь разделитель (',' или ';'), названия колонок и target",
            },
        )

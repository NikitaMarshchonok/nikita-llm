# api/app.py

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO

from agent.tools import basic_eda, detect_task, train_baseline, build_report


app = FastAPI(
    title="Nikita DS Agent",
    description="Загрузи CSV → получи EDA и базовую модель",
    version="0.1.0",
)


# ---------- вспомогалки ----------

def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    """
    Пытаемся прочитать CSV с разными кодировками и разделителями.
    """
    bio = BytesIO(file_bytes)

    variants = [
        {},  # по умолчанию: utf-8 и запятая
        {"sep": ";"},
        {"encoding": "utf-8-sig"},
        {"encoding": "cp1251"},
        {"sep": ";", "encoding": "cp1251"},
        {"encoding": "latin-1"},
        {"sep": ";", "encoding": "latin-1"},
    ]

    for kwargs in variants:
        try:
            bio.seek(0)
            df = pd.read_csv(bio, on_bad_lines="skip", **kwargs)
            if df.shape[1] > 0:
                return df
        except Exception:
            continue

    # если вообще ничего не зашло — отдаём понятную ошибку
    raise ValueError("Не удалось прочитать CSV ни с одной комбинацией кодировка/разделитель")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает имена колонок удобными: без пробелов и точек.
    """
    df = df.copy()
    df.columns = [
        (
            col.strip()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        for col in df.columns
    ]
    return df


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

        # 1) читаем CSV
        df = read_csv_safely(contents)

        # 2) нормализуем имена колонок, чтобы они совпадали с тем, что пользователь пишет в target
        df = normalize_columns(df)

        # если пользователь прислал target — тоже нормализуем так же
        if target is not None:
            target = (
                target.strip()
                .replace(" ", "_")
                .replace(".", "_")
                .replace("-", "_")
                .replace("/", "_")
            )

        # 3) EDA
        eda = basic_eda(df)

        # 4) определяем задачу
        task = detect_task(df, target=target)

        # 5) пробуем обучить, если есть что
        model_res = None
        if task["task"] != "eda" and task["target"]:
            model_res = train_baseline(df, task["target"], task["task"])

        # 6) собираем человекочитаемый отчёт (функция у нас теперь в agent.tools)
        report_text = build_report(df, eda, task, model_res)

        return JSONResponse(
            {
                "filename": file.filename,
                "eda": eda,
                "task": task,
                "model": model_res,
                "report": report_text,
            }
        )

    except ValueError as e:
        # это наши осознанные ошибки чтения
        raise HTTPException(
            status_code=400,
            detail={
                "error": "failed_to_process_file",
                "details": str(e),
                "hint": "проверь разделитель (',' или ';'), названия колонок и target",
            },
        )
    except Exception as e:
        # всё остальное
        raise HTTPException(
            status_code=400,
            detail={
                "error": "failed_to_process_file",
                "details": str(e),
                "hint": "проверь разделитель (',' или ';'), названия колонок и target",
            },
        )


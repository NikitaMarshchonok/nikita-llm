# api/app.py
from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from agent.tools import basic_eda, detect_task, train_baseline, build_report

# 1. СОЗДАЁМ ПРИЛОЖЕНИЕ
app = FastAPI(
    title="Nikita DS Agent",
    description="Загрузи CSV → получи EDA и базовую модель",
    version="0.1.0",
)


# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    """Пробуем разные кодировки и разделители."""
    bio = BytesIO(file_bytes)

    variants = [
        {},  # по умолчанию: utf-8, ','
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

    raise ValueError("Не удалось прочитать CSV ни с одной комбинацией кодировка/разделитель")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Делаем имена колонок удобными и без пробелов."""
    df = df.copy()
    df.columns = [
        col.strip()
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
        for col in df.columns
    ]
    return df


# 3. ЭНДПОИНТЫ

@app.get("/")
def root():
    return {"msg": "Nikita DS Agent is running"}


@app.get("/ui")
def ui():
    """Отдаём простой фронт, если он у тебя есть в api/frontend.html"""
    frontend_path = os.path.join("api", "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"msg": "frontend.html не найден, пользуйся /docs"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    try:
        contents = await file.read()

        # 1) читаем и нормализуем
        df = read_csv_safely(contents)
        df = normalize_columns(df)

        # 2) нормализуем target, если прислали
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

        # 4) задача
        task = detect_task(df, target=target)

        # 5) модель (если есть что предсказывать)
        model_res = None
        if task["task"] != "eda" and task["target"]:
            model_res = train_baseline(df, task["target"], task["task"])

        # 6) отчёт
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
        # это наши понятные ошибки
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
                "hint": "проверь CSV и target",
            },
        )

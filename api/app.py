# api/app.py

from __future__ import annotations

import os
from io import BytesIO
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from agent.tools import (
    basic_eda,
    detect_task,
    train_baseline,
    build_report,
    make_plots_base64,
    analyze_dataset,       # 👈 добавили
    build_recommendations, # 👈 уже было
)

app = FastAPI(
    title="Nikita DS Agent",
    description="Загрузи CSV → получи EDA и базовую модель",
    version="0.1.0",
)

# простое хранилище в памяти
RUNS: dict[str, dict] = {}


# ---------- вспомогалки ----------

def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    bio = BytesIO(file_bytes)

    variants = [
        {},
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


# ---------- UI ----------

@app.get("/ui")
def ui():
    return FileResponse(os.path.join("api", "static", "frontend.html"))


# ---------- API ----------

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

        # 4) определяем задачу
        task = detect_task(df, target=target)

        # 5) анализ проблем датасета (константы, корреляции, дисбаланс и т.п.)
        problems = analyze_dataset(df, eda, task)

        # 6) пробуем обучить модель
        model_res = None
        if task["task"] != "eda" and task["target"]:
            model_res = train_baseline(
                df,
                task["target"],
                task["task"],
            )

        # 7) отчёт и графики
        report_text = build_report(df, eda, task, model_res)
        plots = make_plots_base64(df)

        # 8) рекомендации — теперь с правильной сигнатурой
        recs = build_recommendations(
            df=df,
            eda=eda,
            task=task,
            problems=problems,
            model=model_res,
        )

        # 9) сохраняем запуск
        run_id = f"run_{uuid4().hex[:8]}"
        RUNS[run_id] = {
            "filename": file.filename,
            "eda": eda,
            "task": task,
            "problems": problems,
            "model": model_res,
            "report": report_text,
            "plots": plots,
            "recommendations": recs,
            "columns": list(df.columns),
        }

        return JSONResponse(
            {
                "run_id": run_id,
                "filename": file.filename,
                "eda": eda,
                "task": task,
                "problems": problems,
                "model": model_res,
                "report": report_text,
                "plots": plots,
                "recommendations": recs,
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "failed_to_process_file",
                "details": str(e),
                "hint": "проверь разделитель (',' или ';'), названия колонок и target",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "failed_to_process_file",
                "details": str(e),
                "hint": "проверь CSV и target",
            },
        )


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")
    return RUNS[run_id]


@app.get("/runs")
def list_runs():
    items = []
    for run_id, data in RUNS.items():
        items.append({
            "run_id": run_id,
            "filename": data.get("filename"),
            "task": data.get("task"),
            "has_model": data.get("model") is not None,
        })
    return items

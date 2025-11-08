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
    return FileResponse(os.path.join("api", "frontend.html"))


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

        df = read_csv_safely(contents)
        df = normalize_columns(df)

        if target is not None:
            target = (
                target.strip()
                .replace(" ", "_")
                .replace(".", "_")
                .replace("-", "_")
                .replace("/", "_")
            )

        eda = basic_eda(df)
        task = detect_task(df, target=target)

        model_res = None
        pipeline = None
        if task["task"] != "eda" and task["target"]:
            model_res = train_baseline(
                df,
                task["target"],
                task["task"],
                return_model=True,
            )
            # model_res может быть None, поэтому аккуратно
            if model_res is not None and "pipeline" in model_res:
                pipeline = model_res.pop("pipeline")

        report_text = build_report(df, eda, task, model_res)

        # ---- сохраняем запуск ----
        run_id = f"run_{uuid4().hex[:8]}"
        RUNS[run_id] = {
            "filename": file.filename,
            "eda": eda,
            "task": task,
            "model": model_res,
            "report": report_text,
            "pipeline": pipeline,
            "columns": list(df.columns),
        }

        return JSONResponse(
            {
                "run_id": run_id,
                "filename": file.filename,
                "eda": eda,
                "task": task,
                "model": model_res,
                "report": report_text,
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
    # pipeline не отдаём наружу (его всё равно не сериализовать)
    data = RUNS[run_id].copy()
    data.pop("pipeline", None)
    return data

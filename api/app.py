# api/app.py
from __future__ import annotations

import os
from io import BytesIO
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

from agent.tools import (
    basic_eda,
    detect_task,
    train_baseline,
    build_report,
    make_plots_base64,
    analyze_dataset,
    build_recommendations,
    evaluate_dataset_health,
    build_analysis_status,
    build_next_actions,
    summarize_target,        # ← у тебя уже было
    rank_problems,           # ← у тебя уже было
    auto_feature_suggestions,    # ← добавили
    extract_feature_importance,  # ← добавили
)

app = FastAPI(
    title="Nikita DS Agent",
    description="Загрузи CSV → получи EDA и базовую модель",
    version="0.1.0",
)

# простое хранилище в памяти
RUNS: dict[str, dict] = {}


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


@app.get("/ui")
def ui():
    return FileResponse(os.path.join("api", "static", "frontend.html"))


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

        # 2) нормализуем target, если пользователь его указал
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

        # 5) диагностика
        problems = analyze_dataset(df, task)

        # 5.1) короткая сводка по таргету
        target_summary = summarize_target(df, task)

        # 5.2) ранжированный список проблем (high/medium/low)
        problem_list = rank_problems(problems)

        # 5.3) общая “здоровость” датасета
        dataset_health = evaluate_dataset_health(eda, problems)

        # 5.4) идеи по новым фичам (это чистый список строк)
        feature_suggestions = auto_feature_suggestions(df)

        # 6) базовая модель
        model_res = None
        pipeline = None
        feature_importance: list[dict] = []
        if task["task"] != "eda" and task["target"]:
            # просим вернуть пайплайн
            model_res = train_baseline(
                df,
                task["target"],
                task["task"],
                problems=problems,
                return_model=True,
            )

            # вытаскиваем pipeline и не даём ему попасть в JSON
            if model_res and "pipeline" in model_res:
                pipeline = model_res.pop("pipeline")
                feature_importance = extract_feature_importance(pipeline)

        # 7) человекочитаемый отчёт
        report_text = build_report(df, eda, task, model_res, problems)

        # 8) графики
        plots = make_plots_base64(df)

        # 9) рекомендации
        recs = build_recommendations(df, eda, task, problems, model_res)

        # 10) короткий статус + next actions
        status = build_analysis_status(task, problems, model_res)
        next_actions = build_next_actions(task, problems, model_res)

        # 11) сохраняем в память
        run_id = f"run_{uuid4().hex[:8]}"
        RUNS[run_id] = {
            "run_id": run_id,
            "filename": file.filename,
            "eda": eda,
            "task": task,
            "problems": problems,
            "problem_list": problem_list,
            "target_summary": target_summary,
            "dataset_health": dataset_health,
            "model": model_res,
            "report": report_text,
            "plots": plots,
            "recommendations": recs,
            "status": status,
            "next_actions": next_actions,
            "feature_suggestions": feature_suggestions,
            "feature_importance": feature_importance,
            "columns": list(df.columns),
            "pipeline": pipeline,   # храним только в RAM, не в JSON
        }

        # 12) отдаём на фронт (без pipeline!)
        return JSONResponse(
            {
                "run_id": run_id,
                "filename": file.filename,
                "eda": eda,
                "task": task,
                "problems": problems,
                "problem_list": problem_list,
                "target_summary": target_summary,
                "dataset_health": dataset_health,
                "model": model_res,
                "report": report_text,
                "plots": plots,
                "recommendations": recs,
                "status": status,
                "next_actions": next_actions,
                "feature_suggestions": feature_suggestions,
                "feature_importance": feature_importance,
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


@app.get("/runs/{run_id}/report")
def get_run_report(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")
    return RUNS[run_id]


@app.get("/runs/{run_id}/download")
def download_run_markdown(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")

    data = RUNS[run_id]

    lines = []
    lines.append(f"# Run {run_id}")
    if data.get("filename"):
        lines.append(f"**Файл:** {data['filename']}")
    if data.get("task"):
        t = data["task"]
        lines.append(f"**Задача:** {t.get('task')}  target: `{t.get('target')}`")
    lines.append("")

    if data.get("report"):
        lines.append("## Отчёт")
        lines.append("```")
        lines.append(data["report"])
        lines.append("```")
        lines.append("")

    if data.get("problems"):
        lines.append("## Найденные проблемы")
        for key, val in data["problems"].items():
            lines.append(f"- **{key}**: {val}")
        lines.append("")

    if data.get("recommendations"):
        lines.append("## Что сделать дальше")
        for r in data["recommendations"]:
            lines.append(f"- {r}")
        lines.append("")

    md = "\n".join(lines)
    return PlainTextResponse(md)

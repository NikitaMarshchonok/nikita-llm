# api/app.py
from __future__ import annotations

import os
from io import BytesIO
from uuid import uuid4
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

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
    summarize_target,
    rank_problems,
    auto_feature_suggestions,
    extract_feature_importance,
    build_code_hints,
    detect_column_roles,
    build_experiment_plan,
)

# ---------------------------
# App init + CORS
# ---------------------------
app = FastAPI(
    title="Nikita DS Agent",
    description="Загрузи файл с данными → получи EDA, базовую модель и умный план действий",
    version="0.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JSON-safe данные отдельно от несериализуемых объектов (pipelines)
RUNS: Dict[str, Dict[str, Any]] = {}
PIPELINES: Dict[str, Any] = {}


# ---------------------------
# Utils
# ---------------------------
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


def read_any_table(file_bytes: bytes, filename: str | None) -> pd.DataFrame:
    """
    Универсальный загрузчик табличных данных:
    CSV / TXT / Excel / JSON / Parquet.
    При любой ошибке откатываемся к read_csv_safely.
    """
    ext = (os.path.splitext(filename or "")[1] or "").lower()
    bio = BytesIO(file_bytes)

    # 1) CSV / TXT
    if ext in [".csv", ".txt"]:
        return read_csv_safely(file_bytes)

    # 2) Excel
    if ext in [".xlsx", ".xls"]:
        try:
            bio.seek(0)
            df = pd.read_excel(bio)
            if df.shape[1] > 0:
                return df
        except Exception:
            pass  # упадём в fallback ниже

    # 3) Parquet
    if ext in [".parquet", ".pq"]:
        try:
            bio.seek(0)
            df = pd.read_parquet(bio)
            if df.shape[1] > 0:
                return df
        except Exception:
            pass

    # 4) JSON / JSONL
    if ext in [".json"]:
        try:
            bio.seek(0)
            # сначала пытаемся jsonl
            df = pd.read_json(bio, lines=True)
            if df.shape[1] > 0:
                return df
        except Exception:
            try:
                bio.seek(0)
                df = pd.read_json(bio)
                if df.shape[1] > 0:
                    return df
            except Exception:
                pass

    # 5) fallback — как CSV
    return read_csv_safely(file_bytes)


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


def align_features_for_inference(
    df: pd.DataFrame, train_cols: List[str], target: Optional[str]
) -> pd.DataFrame:
    """
    Делает так, чтобы у инференса были те же фичи, что и при обучении:
    - выкидывает target, если он есть в df
    - создаёт отсутствующие колонки с NaN
    - упорядочивает колонки как в train_cols (без target)
    """
    df = df.copy()
    feat_cols = [c for c in train_cols if (target is None or c != target)]
    if target and target in df.columns:
        df = df.drop(columns=[target])
    for c in feat_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[feat_cols]


def build_llm_context(run: Dict[str, Any]) -> str:
    """
    Собираем текстовый контекст по одному run для LLM:
    задача, здоровье датасета, метрики, важные фичи, проблемы,
    рекомендации и план экспериментов.
    """
    parts: list[str] = []

    task = run.get("task") or {}
    eda = run.get("eda") or {}
    dataset_health = run.get("dataset_health") or {}
    target_summary = run.get("target_summary") or {}
    model = run.get("model") or {}
    feature_importance = run.get("feature_importance") or []
    problem_list = run.get("problem_list") or []
    recommendations = run.get("recommendations") or []
    next_actions = run.get("next_actions") or []
    experiment_plan = run.get("experiment_plan") or []

    # Задача
    parts.append("=== Задача и таргет ===")
    parts.append(f"Тип задачи: {task.get('task')}")
    parts.append(f"Таргет: {task.get('target')}")
    parts.append("")

    # Размер данных
    shape = eda.get("shape")
    if shape and len(shape) == 2:
        parts.append("=== Датасет ===")
        parts.append(f"Размер датасета: {shape[0]} строк × {shape[1]} колонок.")
        parts.append("")

    # Здоровье датасета
    if dataset_health:
        parts.append("=== Здоровье датасета ===")
        parts.append(
            f"Оценка: {dataset_health.get('score')} / уровень: {dataset_health.get('level')}"
        )
        reasons = dataset_health.get("reasons") or []
        if reasons:
            parts.append("Проблемы: " + "; ".join(reasons))
        parts.append("")

    # Сводка по таргету
    if target_summary and (target_summary.get("target") or target_summary.get("has_target")):
        parts.append("=== Таргет ===")
        parts.append(f"Имя таргета: {target_summary.get('target')}")
        parts.append(
            f"Строк: {target_summary.get('n_rows')}, "
            f"пропусков: {target_summary.get('n_missing')} "
            f"(доля={target_summary.get('missing_frac')})"
        )
        if target_summary.get("task") == "classification":
            parts.append(f"Число классов: {target_summary.get('n_classes')}")
            top_classes = target_summary.get("top_classes") or []
            short = []
            for cls in top_classes[:5]:
                label = cls.get("label")
                share = cls.get("share")
                short.append(f"{label} (~{share:.2f})")
            if short:
                parts.append("Топ классы: " + ", ".join(short))
        elif target_summary.get("task") == "regression":
            parts.append(
                "Статистика таргета: "
                f"min={target_summary.get('min')}, "
                f"max={target_summary.get('max')}, "
                f"mean={target_summary.get('mean')}, "
                f"std={target_summary.get('std')}"
            )
        parts.append("")

    # Модель и метрики
    parts.append("=== Модель ===")
    if not model:
        parts.append("Модель не обучалась или не вернулась.")
    elif model.get("model_type") == "skipped":
        parts.append(f"Модель пропущена: {model.get('reason')}")
    else:
        parts.append(f"Тип модели: {model.get('model_type')}")
        if "accuracy" in model:
            parts.append(f"accuracy = {model['accuracy']:.4f}")
        if "f1" in model:
            parts.append(f"f1 = {model['f1']:.4f}")
        if "roc_auc" in model:
            parts.append(f"roc_auc = {model['roc_auc']:.4f}")
        if "rmse" in model:
            parts.append(f"rmse = {model['rmse']:.4f}")
    parts.append("")

    # Топ важные фичи
    if feature_importance:
        parts.append("=== Топ важные признаки ===")
        top_feats = feature_importance[:5]
        for fi in top_feats:
            parts.append(f"- {fi.get('feature')} (importance={fi.get('importance'):.3f})")
        parts.append("")

    # Проблемы (ранжированные)
    if problem_list:
        parts.append("=== Ключевые проблемы в данных ===")
        for p in problem_list[:6]:
            msg = p.get("message") or p.get("code") or p.get("key")
            sev = p.get("severity") or p.get("level")
            parts.append(f"- [{sev}] {msg}")
        parts.append("")

    # Рекомендации
    if recommendations:
        parts.append("=== Рекомендации ===")
        for r in recommendations[:6]:
            parts.append(f"- {r}")
        parts.append("")

    # Next actions
    if next_actions:
        parts.append("=== Следующие шаги ===")
        for a in next_actions[:6]:
            parts.append(f"- {a}")
        parts.append("")

    # План экспериментов
    if experiment_plan:
        parts.append("=== План экспериментов (mid+ DS) ===")
        for step in experiment_plan[:10]:
            prio = step.get("priority")
            title = step.get("title")
            descr = step.get("description")
            parts.append(f"- [{prio}] {title}: {descr}")
        parts.append("")

    return "\n".join(parts)


def call_llm(prompt: str) -> str:
    """
    Заглушка для LLM.
    ⚠️ Сейчас НИЧЕГО не вызывает извне, просто форматирует ответ.
    Потом сюда можно подставить:
      - твой create-llm чат-сервис,
      - или OpenAI / HuggingFace.
    """
    # Попробуем аккуратно вытащить только вопрос пользователя
    marker = "=== Вопрос пользователя ==="
    if marker in prompt:
        user_question = prompt.split(marker, 1)[1].strip()
    else:
        user_question = prompt.strip()

    return (
        "⚠️ LLM ещё не подключён (stub).\n\n"
        "Как DS-ассистент я вижу это так:\n\n"
        f"Вопрос: {user_question}\n\n"
        "• У тебя уже есть EDA, бейзлайн-модель, проблемы, рекомендации и план экспериментов.\n"
        "• Последовательно закрывай шаги из блоков 'Следующие шаги' и 'План экспериментов (mid+ DS)'.\n"
        "• Когда подключим реальную LLM, здесь будет детальный, адаптивный ответ с кодом и примерами."
    )


# ---------------------------
# Static / health
# ---------------------------
@app.get("/ui")
def ui():
    return FileResponse(os.path.join("api", "static", "frontend.html"))


@app.get("/")
def root():
    return {"msg": "Nikita DS Agent is running"}


# ---------------------------
# Upload & Train
# ---------------------------
@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    try:
        contents = await file.read()

        # 1) читаем и нормализуем
        df = read_any_table(contents, file.filename)
        df = normalize_columns(df)

        # 2) нормализуем target; пустую строку считаем отсутствующим target
        if target is not None:
            target = target.strip()
            if target == "":
                target = None
            else:
                target = (
                    target.replace(" ", "_")
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

        # 5.1.1) роли колонок (id / datetime / text / ...)
        column_roles = detect_column_roles(df, task)

        # 5.2) ранжированный список проблем (high/medium/low)
        problem_list = rank_problems(problems)

        # 5.3) здоровость
        dataset_health = evaluate_dataset_health(eda, problems)

        # 5.4) идеи фич
        feature_suggestions = auto_feature_suggestions(df)

        # 6) модель
        model_res = None
        pipeline = None
        feature_importance: List[Dict[str, Any]] = []
        if task["task"] != "eda" and task.get("target"):
            model_res = train_baseline(
                df,
                task["target"],
                task["task"],
                problems=problems,
                return_model=True,
            )
            if model_res and "pipeline" in model_res:
                pipeline = model_res.pop("pipeline")
                try:
                    feature_importance = extract_feature_importance(pipeline)
                except Exception:
                    feature_importance = []

        # 6.1) код-подсказки (сигнатура build_code_hints(problems, task))
        code_hints: List[Any] = []
        try:
            code_hints = build_code_hints(problems, task)
        except Exception:
            code_hints = []

        # 6.2) план экспериментов (mid+ DS)
        experiment_plan = build_experiment_plan(
            task=task,
            problems=problems,
            dataset_health=dataset_health,
            model=model_res,
            column_roles=column_roles,
        )

        # 7) отчёт
        report_text = build_report(df, eda, task, model_res, problems)

        # 8) графики
        plots = make_plots_base64(df)

        # 9) рекомендации
        recs = build_recommendations(df, eda, task, problems, model_res)

        # 10) статус
        status = build_analysis_status(task, problems, model_res)
        next_actions = build_next_actions(task, problems, model_res)

        # 11) сохраняем run
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
            "column_roles": column_roles,
            "model": model_res,
            "report": report_text,
            "plots": plots,
            "recommendations": recs,
            "status": status,
            "next_actions": next_actions,
            "feature_suggestions": feature_suggestions,
            "feature_importance": feature_importance,
            "code_hints": code_hints,
            "columns": list(df.columns),
            "experiment_plan": experiment_plan,
        }
        if pipeline is not None:
            PIPELINES[run_id] = pipeline

        payload = {
            key: RUNS[run_id][key]
            for key in [
                "run_id",
                "filename",
                "eda",
                "task",
                "problems",
                "problem_list",
                "target_summary",
                "dataset_health",
                "model",
                "report",
                "plots",
                "recommendations",
                "status",
                "next_actions",
                "feature_suggestions",
                "feature_importance",
                "code_hints",
                "experiment_plan",
            ]
        }
        return JSONResponse(content=jsonable_encoder(payload))

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
                "hint": "проверь файл и target",
            },
        )


# ---------------------------
# Inference
# ---------------------------
@app.post("/runs/{run_id}/predict")
async def predict_on_run(
    run_id: str,
    file: UploadFile = File(..., description="Файл с новыми объектами (может содержать target)"),
    include_proba: bool = Form(default=True),
    return_rows: int = Form(default=30),
):
    """
    Прогоняет новый файл через pipeline выбранного run_id.
    Если target в файле присутствует, вернём быстрые метрики по этому файлу.
    """
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")
    if run_id not in PIPELINES:
        raise HTTPException(
            status_code=400,
            detail={"error": "no_pipeline", "details": "Для этого run нет сохранённого pipeline. Переобучи /upload."},
        )

    try:
        contents = await file.read()
        df_new = read_any_table(contents, file.filename)
        df_new = normalize_columns(df_new)

        task = RUNS[run_id]["task"]
        train_cols: List[str] = RUNS[run_id]["columns"]
        target = task.get("target")
        task_type = task.get("task")

        # y_true (если есть)
        y_true = None
        if target and target in df_new.columns:
            y_true = df_new[target].copy()

        # выравниваем признаки
        X_inf = align_features_for_inference(df_new, train_cols, target)
        pipe = PIPELINES[run_id]

        # предсказания
        y_pred = pipe.predict(X_inf)
        y_pred_list = np.asarray(y_pred).tolist()

        proba_list = None
        if include_proba and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_inf)
                # если многоклассовая классификация — вернём max prob
                proba_list = np.max(proba, axis=1).tolist()
            except Exception:
                proba_list = None

        # метрики, если есть y_true
        metrics_out = None
        if y_true is not None:
            try:
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    r2_score,
                    mean_squared_error,
                    mean_absolute_error,
                )

                if task_type == "classification":
                    acc = float(accuracy_score(y_true, y_pred))
                    f1m = float(f1_score(y_true, y_pred, average="macro"))
                    metrics_out = {"accuracy": acc, "f1_macro": f1m}
                elif task_type == "regression":
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    mae = float(mean_absolute_error(y_true, y_pred))
                    r2 = float(r2_score(y_true, y_pred))
                    metrics_out = {"rmse": rmse, "mae": mae, "r2": r2}
            except Exception:
                metrics_out = None

        # превью
        preview = pd.DataFrame({"prediction": y_pred_list})
        if proba_list is not None:
            preview["proba"] = proba_list
        preview_full = pd.concat(
            [df_new.reset_index(drop=True), preview],
            axis=1,
        ).head(max(1, return_rows))

        return JSONResponse(
            content=jsonable_encoder(
                {
                    "run_id": run_id,
                    "n_rows": int(len(X_inf)),
                    "task": task,
                    "metrics": metrics_out,
                    "predictions": y_pred_list[:1000],
                    "probabilities": proba_list[:1000] if proba_list is not None else None,
                    "preview": preview_full.to_dict(orient="records"),
                }
            )
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "bad_file", "details": str(e)})
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "predict_failed", "details": str(e)})


# ---------------------------
# Runs utils
# ---------------------------
@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")
    return RUNS[run_id]


@app.get("/runs")
def list_runs():
    items = []
    for run_id, data in RUNS.items():
        items.append(
            {
                "run_id": run_id,
                "filename": data.get("filename"),
                "task": data.get("task"),
                "has_model": data.get("model") is not None,
            }
        )
    return items


@app.get("/runs/{run_id}/report")
def get_run_report(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")
    return RUNS[run_id]


# ---------------------------
# LLM-слой поверх агента
# ---------------------------
@app.post("/ask")
async def ask_agent(
    run_id: str = Form(..., description="ID запуска (run_id), по которому спрашиваем"),
    question: str = Form(..., description="Вопрос к DS-агенту"),
):
    """
    LLM-слой поверх DS-агента:
    - по run_id вытаскиваем весь контекст (EDA, модель, проблемы, рекомендации, план),
    - собираем текстовый контекст,
    - формируем промпт и отправляем в call_llm (пока заглушка),
    - возвращаем текстовый ответ.
    """
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")

    run = RUNS[run_id]
    context_text = build_llm_context(run)

    prompt = (
        "Ты — опытный Data Science ассистент. "
        "У тебя есть результаты анализа датасета, базовой модели, план экспериментов.\n\n"
        "=== Контекст анализа ===\n"
        f"{context_text}\n\n"
        "=== Вопрос пользователя ===\n"
        f"{question}\n\n"
        "Ответь по-русски или на английском, структурировано, с конкретными шагами, "
        "без лишней воды. Если уместно — предложи 2–3 следующих шага."
    )

    answer = call_llm(prompt)

    return {
        "run_id": run_id,
        "question": question,
        "answer": answer,
    }


# ---------------------------
# Export отчёта в Markdown
# ---------------------------
@app.get("/runs/{run_id}/download")
def download_run_markdown(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")

    data = RUNS[run_id]
    lines: List[str] = []
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

    if data.get("experiment_plan"):
        lines.append("## План экспериментов (mid+ DS)")
        for step in data["experiment_plan"]:
            lines.append(f"- [{step.get('priority')}] {step.get('title')}: {step.get('description')}")
        lines.append("")

    md = "\n".join(lines)
    return PlainTextResponse(md)

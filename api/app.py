# api/app.py
from __future__ import annotations
from dotenv import load_dotenv
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
import json
import joblib
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
    auto_model_search,
    suggest_targets,
    build_data_quality,
    # авто-фиксы + сохранение run'ов
    apply_auto_fixes_for_training,
    apply_auto_fixes_for_inference,
    save_run,
)
load_dotenv()
# ---------------------------
# App init + CORS
# ---------------------------
app = FastAPI(
    title="Nikita DS Agent",
    description="Загрузи CSV/Excel/JSON/Parquet → получи EDA, базовую модель и инференс",
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

# Ограничение размера датасета для анализа/обучения
MAX_ROWS: int = int(os.getenv("DS_AGENT_MAX_ROWS", "25000"))

# Папка для сохранения run'ов
RUNS_DIR = os.path.join("runs")


def load_run_from_disk(run_id: str) -> tuple[Dict[str, Any] | None, Any | None]:
    """
    Пытаемся загрузить run с диска: runs/<run_id>/report.json (+ model.joblib).
    Возвращаем (run_data, pipeline) или (None, None), если не нашли/не получилось.
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    report_path = os.path.join(run_dir, "report.json")
    if not os.path.exists(report_path):
        return None, None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            run_data: Dict[str, Any] = json.load(f)
    except Exception:
        return None, None

    model_path = os.path.join(run_dir, "model.joblib")
    pipeline = None
    if os.path.exists(model_path):
        try:
            pipeline = joblib.load(model_path)
        except Exception:
            pipeline = None

    run_data.setdefault("run_id", run_id)
    return run_data, pipeline


def get_run_and_pipeline(run_id: str) -> tuple[Dict[str, Any], Any | None]:
    """
    Берём run из памяти или подгружаем с диска при первом обращении.
    Если run нет ни в памяти, ни на диске — кидаем 404.
    """
    if run_id in RUNS:
        return RUNS[run_id], PIPELINES.get(run_id)

    run_data, pipeline = load_run_from_disk(run_id)
    if run_data is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    RUNS[run_id] = run_data
    if pipeline is not None:
        PIPELINES[run_id] = pipeline

    return run_data, pipeline


# ---------------------------
# Utils: чтение файлов
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
    """Выравниваем фичи инференса под тренировочные фичи."""
    df = df.copy()
    feat_cols = [c for c in train_cols if (target is None or c != target)]
    if target and target in df.columns:
        df = df.drop(columns=[target])
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[feat_cols]


# ---------------------------
# Сбор контекста для LLM
# ---------------------------
def build_llm_context(run: Dict[str, Any]) -> str:
    """
    Собираем текстовый контекст по одному run для LLM:
    задача, здоровье датасета, метрики, важные фичи, проблемы,
    рекомендации, next actions и план экспериментов.
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
            f"пропусков: {target_summary.get('n_missing')}"
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
        parts.append(f"Тип модели: {model.get('model_type')}")
    else:
        parts.append(f"Тип модели: {model.get('model_type')}")
        if "accuracy" in model:
            parts.append(f"accuracy = {model['accuracy']:.4f}")
        if "f1" in model:
            parts.append(f"f1 = {model['f1']:.4f}")
        if "precision" in model:
            parts.append(f"precision = {model['precision']:.4f}")
        if "recall" in model:
            parts.append(f"recall = {model['recall']:.4f}")
        if "roc_auc" in model and model.get("roc_auc") is not None:
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

    # Следующие шаги
    if next_actions:
        parts.append("=== Следующие шаги (кратко) ===")
        for a in next_actions[:6]:
            parts.append(f"- {a}")
        parts.append("")

    # План экспериментов
    if experiment_plan:
        parts.append("=== План экспериментов (now / next / later) ===")
        for step in experiment_plan[:10]:
            parts.append(
                f"- [{step.get('priority')}] "
                f"{step.get('title')}: {step.get('description')}"
            )
        parts.append("")

    return "\n".join(parts)


# ---------------------------
# LLM: реальный вызов + fallback
# ---------------------------
def call_llm(prompt: str) -> str:
    """
    Вызов LLM через OpenAI, с безопасным fallback, если:
    - нет OPENAI_API_KEY,
    - библиотека не установлена,
    - произошла ошибка при запросе.
    """
    def _fallback(reason: str) -> str:
        question_marker = "=== Вопрос пользователя ==="
        if question_marker in prompt:
            user_question = prompt.split(question_marker, 1)[1].strip()
        else:
            user_question = prompt.strip()

        return (
            f"⚠️ LLM пока не подключён или недоступен ({reason}).\n\n"
            "Но вот как я бы интерпретировал ситуацию как DS-ассистент:\n\n"
            f"Вопрос: {user_question}\n\n"
            "• У тебя уже есть EDA, бейзлайн-модель, список проблем и план экспериментов.\n"
            "• Следующий шаг — последовательно закрывать пункты из блоков "
            "«Что сделать дальше», «Следующие шаги» и из секции now/next в плане экспериментов.\n"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        return _fallback("нет переменной окружения OPENAI_API_KEY")

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _fallback("библиотека openai не установлена (pip install -U openai)")

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты опытный Data Science ассистент уровня strong middle/senior. "
                        "Отвечай по-делу, структурировано, без воды. "
                        "Давай конкретные шаги по работе с данными, фичами, моделями и метриками."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        if not content:
            return _fallback("пустой ответ от модели")
        return content
    except Exception as e:
        return _fallback(f"ошибка вызова API: {e}")


# ---------------------------
# LLM: Gemini + fallback
# ---------------------------
def call_gemini(prompt: str) -> str:
    """
    Вызов Gemini (google-generativeai) с безопасным fallback.
    Использует GEMINI_API_KEY и GEMINI_MODEL (по умолчанию gemini-2.5-flash).
    """
    def _fallback(reason: str) -> str:
        question_marker = "=== Вопрос пользователя ==="
        if question_marker in prompt:
            user_question = prompt.split(question_marker, 1)[1].strip()
        else:
            user_question = prompt.strip()

        return (
            f"⚠️ Gemini пока не подключён или недоступен ({reason}).\n\n"
            "Но вот как я бы подсказал как DS-комментатор:\n\n"
            f"Вопрос: {user_question}\n\n"
            "• Проверь качество данных (пропуски, кардинальность, дисбаланс).\n"
            "• Сравни текущую модель с простым бейзлайном.\n"
            "• Используй план экспериментов (now / next / later), чтобы шаг за шагом "
            "улучшать качество и готовить отчёт.\n"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        return _fallback("нет переменной окружения GEMINI_API_KEY")

    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return _fallback("библиотека google-generativeai не установлена (pip install -U google-generativeai)")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            return _fallback("пустой ответ от Gemini")
        return text
    except Exception as e:
        return _fallback(f"ошибка вызова Gemini API: {e}")



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
        df_raw = read_any_table(contents, file.filename)
        df_raw = normalize_columns(df_raw)

        # 1.1) ограничение по размеру датасета (по сэмплу делаем EDA и обучение)
        original_rows = int(len(df_raw))
        sampling_info: Dict[str, Any] = {
            "applied": False,
            "original_rows": original_rows,
            "used_rows": original_rows,
        }
        if original_rows > MAX_ROWS:
            df = df_raw.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
            sampling_info.update(
                {
                    "applied": True,
                    "used_rows": MAX_ROWS,
                    "strategy": "random_sample",
                    "random_state": 42,
                    "note": (
                        f"Датасет был случайно подсэмплирован до {MAX_ROWS} строк "
                        "для ускорения анализа и обучения."
                    ),
                }
            )
        else:
            df = df_raw.copy()

        # 2) нормализуем target; пустую строку считаем отсутствующим
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

        # 3) EDA (по подсэмплированному df)
        eda = basic_eda(df)
        # добавим метаданные по сэмплированию в EDA
        eda["original_rows"] = original_rows
        eda["used_rows"] = sampling_info["used_rows"]
        eda["sampling_applied"] = sampling_info["applied"]

        # 4) задача
        task = detect_task(df, target=target)

        # 5) диагностика
        problems = analyze_dataset(df, task)

        # 5.0) кандидаты таргета (на основе текущего анализа)
        target_suggestions = suggest_targets(
            df,
            problems=problems,
            current_target=task.get("target"),
        )

        # 5.1) короткая сводка по таргету
        target_summary = summarize_target(df, task)

        # 5.1.1) роли колонок (id / datetime / text / ...)
        column_roles = detect_column_roles(df, task)

        # 5.2) ранжированный список проблем (high/medium/low)
        problem_list = rank_problems(problems)

        # 5.3) здоровье датасета
        dataset_health = evaluate_dataset_health(eda, problems)

        # 5.4) идеи фич
        feature_suggestions = auto_feature_suggestions(df)

        # 5.5) расширенная диагностика качества данных
        data_quality = build_data_quality(df, task)


        # === Шаг 1: авто-фиксы под "Сделать сейчас" ===
        df_model, auto_fixes = apply_auto_fixes_for_training(
            df,
            problems=problems,
            task=task,
            high_null_threshold=30.0,
            drop_threshold=80.0,
        )

        # 6) авто-поиск лучшей модели (уже на df_model)
        model_res: Optional[Dict[str, Any]] = None
        pipeline = None
        model_leaderboard: Optional[list] = None
        feature_importance: List[Dict[str, Any]] = []

        if task["task"] != "eda" and task.get("target"):
            try:
                auto_res = auto_model_search(df_model, task, problems)
            except Exception:
                auto_res = None

            if auto_res is not None:
                model_res = auto_res.get("best_model")
                pipeline = auto_res.get("pipeline")
                model_leaderboard = auto_res.get("leaderboard")

            # fallback: если всё упало — старый train_baseline
            if model_res is None or model_res.get("model_type") == "skipped":
                baseline = train_baseline(
                    df_model,
                    task["target"],
                    task["task"],
                    problems=problems,
                    return_model=True,
                )
                if baseline:
                    if "pipeline" in baseline:
                        pipeline = baseline.pop("pipeline")
                    model_res = baseline

        # 6.1) feature importance, если смогли обучить pipeline
        if pipeline is not None:
            try:
                feature_importance = extract_feature_importance(pipeline)
            except Exception:
                feature_importance = []

        # 6.2) код-подсказки под проблемы
        try:
            code_hints = build_code_hints(problems, task)
        except Exception:
            code_hints = []

        # 7) текстовый отчёт (по исходному df, чтобы описывать "сырые" данные)
        report_text = build_report(df, eda, task, model_res, problems)

        # 8) графики по данным (EDA)
        plots = make_plots_base64(df)

        # 8.1) ML-графики из модели (confusion matrix, ROC), если они есть
        ml_plots = []
        if model_res and isinstance(model_res, dict):
            ml_plots = model_res.get("ml_plots") or []
        if ml_plots:
            plots.extend(ml_plots)


        # 9) рекомендации
        recs = build_recommendations(df, eda, task, problems, model_res)

        # 10) статус + next actions
        status = build_analysis_status(task, problems, model_res)
        next_actions = build_next_actions(task, problems, model_res)

        # 11) план экспериментов (now/next/later)
        experiment_plan = build_experiment_plan(
            task=task,
            problems=problems,
            dataset_health=dataset_health,
            model=model_res,
            column_roles=column_roles,
        )

        # 12) собираем структуру run
        run_id = f"run_{uuid4().hex[:8]}"
        run_record: Dict[str, Any] = {
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
            "model_leaderboard": model_leaderboard,
            "report": report_text,
            "plots": plots,
            "recommendations": recs,
            "status": status,
            "next_actions": next_actions,
            "feature_suggestions": feature_suggestions,
            "feature_importance": feature_importance,
            "code_hints": code_hints,
            "experiment_plan": experiment_plan,
            "target_suggestions": target_suggestions,
            "columns": list(df.columns),             # сырые колонки
            "model_columns": list(df_model.columns), # колонки для модели
            "sampling": sampling_info,
            "auto_fixes": auto_fixes,                # метаданные авто-фиксов
            "data_quality": data_quality,
        }

        # 12.1) сохраняем run на диск (JSON + model.joblib)
        save_run(run_record, pipeline, run_id=run_id)

        # 12.2) кладём в память для быстрого доступа
        RUNS[run_id] = run_record
        if pipeline is not None:
            PIPELINES[run_id] = pipeline

        payload_keys = [
            "run_id",
            "filename",
            "eda",
            "task",
            "problems",
            "problem_list",
            "target_summary",
            "dataset_health",
            "model",
            "model_leaderboard",
            "report",
            "plots",
            "recommendations",
            "status",
            "next_actions",
            "experiment_plan",
            "feature_suggestions",
            "feature_importance",
            "code_hints",
            "target_suggestions",
            "sampling",
            "auto_fixes",
            "data_quality",
        ]
        payload = {key: run_record.get(key) for key in payload_keys}
        return JSONResponse(content=jsonable_encoder(payload))

    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "bad_file", "details": str(e)})
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "upload_failed", "details": str(e)})


# ---------------------------
# Inference
# ---------------------------
@app.post("/runs/{run_id}/predict")
async def predict_on_run(
    run_id: str,
    file: UploadFile = File(..., description="CSV/Excel/JSON/Parquet c новыми объектами (может содержать target)"),
    include_proba: bool = Form(default=True),
    return_rows: int = Form(default=30),
):
    """Прогоняет новый файл через pipeline выбранного run_id.
    Если target в файле присутствует, вернём быстрые метрики по этому файлу.
    """
    # ✅ Берём run и pipeline (из памяти или с диска)
    run, pipeline = get_run_and_pipeline(run_id)

    if pipeline is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_pipeline",
                "details": "Для этого run нет сохранённого pipeline. Переобучи /upload.",
            },
        )

    try:
        contents = await file.read()
        df_new_raw = read_any_table(contents, file.filename)
        df_new_raw = normalize_columns(df_new_raw)

        task = run["task"]
        target = task.get("target")
        task_type = task.get("task")

        # какие колонки ждёт модель (те, на которых она реально училась)
        train_cols: List[str] = run.get("model_columns") or run.get("columns")

        # авто-фиксы, которые применялись при обучении
        auto_fixes = run.get("auto_fixes")

        # применяем те же авто-фиксы к новому датасету
        df_new_model = apply_auto_fixes_for_inference(df_new_raw, auto_fixes)

        # y_true (если есть таргет в новом файле)
        y_true = None
        if target and target in df_new_model.columns:
            y_true = df_new_model[target].copy()

        # выравниваем признаки под train-колонки
        X_inf = align_features_for_inference(df_new_model, train_cols, target)

        # ✅ используем pipeline, который вернул helper
        pipe = pipeline

        # предсказания
        y_pred = pipe.predict(X_inf)
        y_pred_list = np.asarray(y_pred).tolist()

        proba_list = None
        if include_proba and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_inf)
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

        # превью — показываем сырые колонки + prediction (+ proba)
        preview = df_new_raw.reset_index(drop=True).copy()
        preview["prediction"] = y_pred_list
        if proba_list is not None:
            preview["proba"] = proba_list
        preview_full = preview.head(max(1, return_rows))

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
    """Возвращает сохранённый run (из памяти или с диска)."""
    run, _ = get_run_and_pipeline(run_id)
    return run


@app.get("/runs")
def list_runs():
    """
    Список запусков:
    - сначала те, что уже есть в памяти,
    - затем поднимаем то, что лежит в папке runs/ (если сервер перезапускался).
    """
    items: list[dict[str, Any]] = []

    # из памяти
    for run_id, data in RUNS.items():
        items.append(
            {
                "run_id": run_id,
                "filename": data.get("filename"),
                "task": data.get("task"),
                "has_model": data.get("model") is not None,
            }
        )

    # с диска
    if os.path.isdir(RUNS_DIR):
        for name in sorted(os.listdir(RUNS_DIR)):
            run_id = name
            # не дублируем то, что уже в памяти
            if any(it["run_id"] == run_id for it in items):
                continue

            run_data, _ = load_run_from_disk(run_id)
            if not run_data:
                continue

            items.append(
                {
                    "run_id": run_id,
                    "filename": run_data.get("filename"),
                    "task": run_data.get("task"),
                    "has_model": run_data.get("model") is not None,
                }
            )

    return items


@app.get("/runs/{run_id}/report")
def get_run_report(run_id: str):
    """
    Возвращает полный run (тот же, что /runs/{run_id}),
    оставлено для совместимости.
    """
    run, _ = get_run_and_pipeline(run_id)
    return run

@app.get("/runs/{run_id}/report")
def get_run_report(run_id: str):
    run, _ = get_run_and_pipeline(run_id)
    return run



# LLM-слой: /ask
# ---------------------------
@app.post("/ask")
async def ask_agent(
    run_id: str = Form(..., description="ID запуска (run_id), по которому спрашиваем"),
    question: str = Form(..., description="Вопрос к DS-агенту"),
    engine: str = Form(
        default="gemini",
        description="LLM движок: 'gemini' или 'openai'",
    ),
):
    """
    LLM-слой поверх DS-агента:
    - по run_id вытаскиваем контекст (EDA, модель, проблемы, рекомендации),
    - собираем текстовый контекст,
    - формируем промпт,
    - отправляем в выбранный движок (по умолчанию Gemini).
    """
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run_id not found")

    run = RUNS[run_id]
    context_text = build_llm_context(run)

    prompt = (
        "Ты — опытный Data Science ассистент уровня strong middle/senior.\n"
        "У тебя есть результаты анализа датасета, бейзлайн-модель, список проблем, "
        "оценка здоровья датасета и план экспериментов.\n\n"
        "=== Контекст анализа ===\n"
        f"{context_text}\n\n"
        "=== Вопрос пользователя ===\n"
        f"{question}\n\n"
        "Ответь по-русски или на английском (как удобнее), структурировано и по шагам. "
        "Без лишней воды, дай 2–5 конкретных следующих шагов."
    )

    if engine == "openai":
        answer = call_llm(prompt)
    else:
        answer = call_gemini(prompt)

    return {
        "run_id": run_id,
        "engine": engine,
        "question": question,
        "answer": answer,
    }



# ---------------------------
# Экспорт отчёта в Markdown
# ---------------------------
@app.get("/runs/{run_id}/download")
def download_run_markdown(run_id: str):
    """
    Генерирует человекочитаемый Markdown-отчёт по одному run:
    данные, задача, здоровье, модель, метрики, проблемы,
    рекомендации, план экспериментов, идеи фич и подсказки по коду.
    """
    run, _ = get_run_and_pipeline(run_id)

    eda = run.get("eda") or {}
    task = run.get("task") or {}
    dataset_health = run.get("dataset_health") or {}
    model = run.get("model") or {}
    sampling = run.get("sampling") or {}
    problem_list = run.get("problem_list") or []
    recommendations = run.get("recommendations") or []
    experiment_plan = run.get("experiment_plan") or []
    feature_suggestions = run.get("feature_suggestions") or []
    code_hints = run.get("code_hints") or []
    target_summary = run.get("target_summary") or {}
    status = run.get("status") or {}
    next_actions = run.get("next_actions") or []

    lines: list[str] = []

    # Заголовок
    lines.append(f"# DS-отчёт по запуску `{run_id}`")
    filename = run.get("filename")
    if filename:
        lines.append(f"**Файл:** `{filename}`")

    shape = eda.get("shape")
    if shape and len(shape) == 2:
        n_rows, n_cols = shape
        lines.append(f"**Размер таблицы:** {n_rows} строк × {n_cols} колонок.")

    if sampling.get("applied"):
        lines.append(
            f"**Сэмплирование:** использовано {sampling.get('used_rows')} строк "
            f"из {sampling.get('original_rows')} (стратегия: {sampling.get('strategy')})."
        )
    lines.append("")

    # 1. Задача и таргет
    lines.append("## 1. Задача и таргет")
    task_type = task.get("task")
    target = task.get("target")
    lines.append(f"- Тип задачи: `{task_type}`")
    lines.append(f"- Таргет: `{target}`")
    lines.append("")

    # Краткая сводка по таргету
    if target_summary.get("target") or target_summary.get("has_target"):
        lines.append("### 1.1. Сводка по таргету")
        lines.append(
            f"- Строк: {target_summary.get('n_rows')}, "
            f"пропусков: {target_summary.get('n_missing')}"
        )
        if target_summary.get("task") == "classification":
            lines.append(f"- Число классов: {target_summary.get('n_classes')}")
            top_classes = target_summary.get("top_classes") or []
            if top_classes:
                lines.append("- Топ классы:")
                for cls in top_classes[:5]:
                    label = cls.get("label")
                    share = cls.get("share")
                    lines.append(f"  - {label}: ~{share:.2f}")
        elif target_summary.get("task") == "regression":
            lines.append(
                "- Статистика: "
                f"min={target_summary.get('min')}, "
                f"max={target_summary.get('max')}, "
                f"mean={target_summary.get('mean')}, "
                f"std={target_summary.get('std')}"
            )
        lines.append("")

    # 2. Здоровье датасета
    lines.append("## 2. Здоровье датасета")
    score = dataset_health.get("score")
    level = dataset_health.get("level")
    if score is not None or level:
        lines.append(f"- Оценка: {score}/100 · уровень: {level}")
    reasons = dataset_health.get("reasons") or []
    if reasons:
        lines.append("- Основные проблемы:")
        for r in reasons:
            lines.append(f"  - {r}")
    lines.append("")

    # 3. Модель и метрики
    lines.append("## 3. Модель и метрики")
    if not model:
        lines.append("- Модель не обучалась или не была сохранена.")
    elif model.get("model_type") == "skipped":
        lines.append("- Модель пропущена (model_type = `skipped`).")
    else:
        lines.append(f"- Тип модели: `{model.get('model_type')}`")
        # Ключевые метрики
        if task_type == "classification":
            acc = model.get("accuracy")
            f1m = model.get("f1")
            roc_auc = model.get("roc_auc")
            if acc is not None:
                lines.append(f"- accuracy: {acc:.4f}")
            if f1m is not None:
                lines.append(f"- f1_macro: {f1m:.4f}")
            if roc_auc is not None:
                lines.append(f"- roc_auc: {roc_auc:.4f}")
        elif task_type == "regression":
            rmse = model.get("rmse")
            mae = model.get("mae")
            r2 = model.get("r2")
            if rmse is not None:
                lines.append(f"- RMSE: {rmse:.4f}")
            if mae is not None:
                lines.append(f"- MAE: {mae:.4f}")
            if r2 is not None:
                lines.append(f"- R²: {r2:.4f}")
    lines.append("")

    # 4. Проблемы в данных
    lines.append("## 4. Ключевые проблемы в данных")
    if problem_list:
        for p in problem_list:
            sev = p.get("severity") or p.get("level")
            msg = p.get("message") or p.get("code") or p.get("key")
            lines.append(f"- [{sev}] {msg}")
    else:
        lines.append("- Явных проблем не найдено или они не были сохранены.")
    lines.append("")

    # 5. Рекомендации и следующие шаги
    lines.append("## 5. Рекомендации")
    if recommendations:
        for r in recommendations:
            lines.append(f"- {r}")
    else:
        lines.append("- Рекомендации отсутствуют.")
    lines.append("")

    lines.append("## 6. Следующие шаги")
    if next_actions:
        for a in next_actions:
            lines.append(f"- {a}")
    else:
        lines.append("- Следующие шаги не заданы.")
    lines.append("")

    # 6.1. Статус анализа (если есть)
    if status:
        lines.append("### 6.1. Статус анализа")
        for k, v in status.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # 7. План экспериментов
    lines.append("## 7. План экспериментов (now / next / later)")
    if experiment_plan:
        for step in experiment_plan:
            lines.append(
                f"- [{step.get('priority')}] {step.get('title')}: "
                f"{step.get('description')}"
            )
    else:
        lines.append("- План экспериментов не сформирован.")
    lines.append("")

    # 8. Идеи новых фич
    lines.append("## 8. Идеи новых признаков")
    if feature_suggestions:
        for fs in feature_suggestions:
            lines.append(f"- {fs}")
    else:
        lines.append("- Идей новых признаков нет или они не были сохранены.")
    lines.append("")

    # 9. Подсказки по коду
    lines.append("## 9. Подсказки по коду и приёмы")
    if code_hints:
        for hint in code_hints:
            title = hint.get("title") or "Сниппет"
            desc = hint.get("description") or ""
            code = (hint.get("code") or "").strip()

            lines.append(f"### {title}")
            if desc:
                lines.append(desc)
                lines.append("")
            if code:
                lines.append("```python")
                lines.append(code)
                lines.append("```")
                lines.append("")
    else:
        lines.append("- Подсказки по коду не сформированы.")
    lines.append("")

    md = "\n".join(lines)
    return PlainTextResponse(md)

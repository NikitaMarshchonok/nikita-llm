# agent/tools.py
from __future__ import annotations

import os
import re
import json
import uuid
import base64
from io import BytesIO
from typing import Optional, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

import matplotlib
matplotlib.use("Agg")  # чтобы рендерить без GUI
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------------
def basic_eda(df: pd.DataFrame) -> dict:
    eda = {
        "shape": list(df.shape),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats = {}
    for c in numeric_cols[:20]:
        ser = df[c]
        stats[c] = {
            "mean": float(ser.mean()),
            "std": float(ser.std() or 0),
            "min": float(ser.min()),
            "max": float(ser.max()),
        }
    eda["numeric_stats"] = stats
    return eda


# ---------------------------------------------------------------------
# 2. угадывание таргета и задачи
# ---------------------------------------------------------------------
ID_LIKE = {"id", "ID", "Id", "index", "Rk", "rank"}


def _looks_like_id(colname: str) -> bool:
    return colname in ID_LIKE or re.search(r"id$", colname, re.IGNORECASE) is not None


def _guess_target(df: pd.DataFrame) -> tuple[Literal["eda", "classification", "regression"], Optional[str]]:
    lower_cols = {c.lower(): c for c in df.columns}

    # популярные названия
    for cand in ("target", "label", "class", "y"):
        if cand in lower_cols:
            col = lower_cols[cand]
            if df[col].nunique() <= 50:
                return "classification", col
            else:
                return "regression", col

    # маленькие категориальные
    for c in df.columns:
        if _looks_like_id(c):
            continue
        uniq = df[c].nunique(dropna=True)
        if 2 <= uniq <= 30:
            return "classification", c

    # числовые
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if _looks_like_id(c):
            continue
        if df[c].nunique() > 1:
            return "regression", c

    return "eda", None


def detect_task(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    if target is not None and target in df.columns:
        nunique = df[target].nunique()
        if df[target].dtype == "object" or nunique <= 30:
            task = "classification"
        else:
            task = "regression"
        return {"task": task, "target": target}

    task, tgt = _guess_target(df)
    return {"task": task, "target": tgt}


# ---------------------------------------------------------------------
# 3. приведение строк к числам и препроцессинг
# ---------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Пробуем строки, похожие на числа, привести к числам.
    Без warnings.
    """
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == "object":
            # сначала чистим строку
            ser = new_df[col].astype(str).str.replace(",", "").str.replace(" ", "")
            try:
                converted = pd.to_numeric(ser)
            except Exception:
                # не получилось — оставляем как было
                continue
            else:
                # получилось — подменяем колонку
                new_df[col] = converted
    return new_df



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------
# 4. обучение базовой модели
# ---------------------------------------------------------------------
def train_baseline(
    df: pd.DataFrame,
    target: str,
    task: str,
    return_model: bool = False,
) -> Optional[dict]:
    """
    Обучаем очень базовую модель поверх авто-препроцессинга.
    Возвращаем метрики и тип модели.
    Если return_model=True — ещё и сам sklearn-пайплайн (для /predict).
    """
    if target not in df.columns:
        return None

    # попробуем привести строковые числа
    df = _coerce_numeric(df)

    y = df[target]
    X = df.drop(columns=[target])

    if X.shape[1] == 0:
        return None

    preprocessor = build_preprocessor(X)

    # ----- классификация -----
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        # проверяем, можно ли стратифицировать
        counts = y.value_counts(dropna=False)
        can_stratify = (counts >= 2).all()

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if (y.nunique() < 50 and can_stratify) else None,
        )

        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        acc = float(accuracy_score(y_val, preds))
        f1 = float(f1_score(y_val, preds, average="weighted"))

        res: dict = {
            "model_type": "RandomForestClassifier",
            "accuracy": acc,
            "f1": f1,
        }
        if return_model:
            res["pipeline"] = pipe
        return res

    # ----- регрессия -----
    elif task == "regression":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        mse = float(mean_squared_error(y_val, preds))
        rmse = mse ** 0.5

        res: dict = {
            "model_type": "RandomForestRegressor",
            "rmse": rmse,
        }
        if return_model:
            res["pipeline"] = pipe
        return res

    else:
        return None


# ---------------------------------------------------------------------
# 5. отчёт в виде текста
# ---------------------------------------------------------------------
def build_report(df: pd.DataFrame, eda: dict, task: dict, model: dict | None) -> str:
    rows, cols = eda["shape"]
    lines: list[str] = []

    lines.append(f"📊 В датасете {rows} строк и {cols} колонок.")

    # пропуски
    nulls = eda.get("nulls", {})
    top_nulls = {k: v for k, v in nulls.items() if v > 0}
    if top_nulls:
        lines.append("🕳️ Пропуски (топ):")
        for k, v in list(top_nulls.items())[:10]:
            lines.append(f"  • {k}: {v}")

    # немного про числа
    num_stats = eda.get("numeric_stats", {})
    if num_stats:
        lines.append("📐 Числовые признаки (mean / std / min / max):")
        for name, st in list(num_stats.items())[:10]:
            lines.append(
                f"  • {name}: {st['mean']:.3f}/{st['std']:.3f}/{st['min']}/{st['max']}"
            )

    # задача
    if task["task"] == "eda" or task["target"] is None:
        lines.append("🧠 Подходящей целевой колонки не нашлось — сделан только EDA.")
    else:
        lines.append(f'🧠 Задача: {task["task"]} по колонке "{task["target"]}".')

    # модель
    if model:
        if "accuracy" in model:
            lines.append(
                f'🧪 Модель: {model["model_type"]}, accuracy={model["accuracy"]:.3f}, f1={model["f1"]:.3f}'
            )
        elif "rmse" in model:
            lines.append(
                f'🧪 Модель: {model["model_type"]}, RMSE={model["rmse"]:.3f}'
            )
    else:
        lines.append("📦 Модель не обучалась.")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# 6. графики → base64
# ---------------------------------------------------------------------
def make_plots_base64(df: pd.DataFrame) -> list[dict]:
    plots: list[dict] = []

    # гистограммы по первым 3 числовым
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()[:3]
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f"Distribution of {col}")
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plots.append({"name": f"hist_{col}", "image_base64": b64})

    # корреляция
    if len(df.select_dtypes(include=["number"]).columns) >= 2:
        corr = df.select_dtypes(include=["number"]).corr()
        fig, ax = plt.subplots(figsize=(4, 3))
        cax = ax.imshow(corr, cmap="viridis")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(cax)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        plots.append({"name": "correlation", "image_base64": b64})

    return plots


# ---------------------------------------------------------------------
# 7. сохранение ранa на диск (если надо с диском работать)
# ---------------------------------------------------------------------
def save_run(run_data: dict, model_pipeline) -> str:
    """
    Сохраняет run в папку runs/<uuid>/ :
      - report.json
      - model.joblib (если есть модель)
    """
    run_id = str(uuid.uuid4())
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)

    if model_pipeline is not None:
        joblib.dump(model_pipeline, os.path.join(run_dir, "model.joblib"))

    return run_id

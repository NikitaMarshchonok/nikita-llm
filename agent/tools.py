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
import re
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
# 1. EDA (расширенный)
# ---------------------------------------------------------------------
def basic_eda(df: pd.DataFrame) -> dict:
    """
    Расширенный EDA:
    - форма
    - типы
    - пропуски (кол-во и доля)
    - базовые статистики по числовым
    - константные и почти константные признаки
    - пары с высокой корреляцией
    """
    eda: dict = {
        "shape": list(df.shape),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }

    # пропуски
    null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    null_frac = {
        c: float(df[c].isna().mean()) for c in df.columns
    }
    eda["nulls"] = null_counts
    eda["null_fractions"] = null_frac

    # базовые stats
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats = {}
    for c in numeric_cols[:30]:
        ser = df[c]
        stats[c] = {
            "mean": float(ser.mean()),
            "std": float(ser.std() or 0),
            "min": float(ser.min()),
            "max": float(ser.max()),
        }
    eda["numeric_stats"] = stats

    # константные / почти константные
    constant_features = []
    quasi_constant_features = []
    for c in df.columns:
        uniq = df[c].nunique(dropna=True)
        if uniq <= 1:
            constant_features.append(c)
        else:
            # доля самого частого значения
            top_frac = float(df[c].value_counts(normalize=True, dropna=False).iloc[0])
            if top_frac > 0.98:
                quasi_constant_features.append(c)
    eda["constant_features"] = constant_features
    eda["quasi_constant_features"] = quasi_constant_features

    # пары с высокой корреляцией (только числовые)
    high_corr_pairs = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        # только верхний треугольник
        for i, c1 in enumerate(numeric_cols):
            for j in range(i + 1, len(numeric_cols)):
                c2 = numeric_cols[j]
                val = float(corr.loc[c1, c2])
                if val >= 0.9:
                    high_corr_pairs.append(
                        {"feature_1": c1, "feature_2": c2, "corr": val}
                    )
    eda["high_corr_pairs"] = high_corr_pairs

    return eda


def build_recommendations(
    eda: dict,
    task: dict,
    model: dict | None = None,
) -> list[str]:
    """
    На основе EDA/задачи/модели выдаём список рекомендаций.
    Это то, что мы покажем в UI справа.
    """
    recs: list[str] = []

    # пропуски
    null_fracs = eda.get("null_fractions", {})
    big_nulls = [c for c, f in null_fracs.items() if f > 0.3]
    if big_nulls:
        recs.append(
            f"Высокая доля пропусков в колонках: {', '.join(big_nulls)} — стоит либо имPUTировать, либо удалить."
        )

    # константы
    consts = eda.get("constant_features", [])
    if consts:
        recs.append(
            f"Константные признаки: {', '.join(consts)} — их можно удалить без потери качества."
        )

    quasi = eda.get("quasi_constant_features", [])
    if quasi:
        recs.append(
            f"Почти константные признаки: {', '.join(quasi)} — стоит проверить, действительно ли они полезны."
        )

    # высокая корреляция
    high_corr = eda.get("high_corr_pairs", [])
    if high_corr:
        top_pairs = ", ".join([f"{p['feature_1']}/{p['feature_2']} ({p['corr']:.2f})" for p in high_corr[:5]])
        recs.append(
            f"Есть сильно коррелирующие пары признаков: {top_pairs} — можно сделать отбор признаков или regularization."
        )

    # про задачу
    if task.get("task") == "classification" and task.get("target"):
        # если мы знаем таргет — можно посчитать дисбаланс
        # (тут лучше считать в api, где есть сам df, но сделаем простое правило)
        recs.append(
            "Для классификации стоит проверить дисбаланс классов и при необходимости использовать class_weight/oversampling."
        )

    if model:
        if "accuracy" in model and model["accuracy"] < 0.7:
            recs.append("Точность ниже 0.7 — попробуй более мощную модель или лучшее препроцессирование.")
        if "rmse" in model:
            recs.append("Для регрессии можно попробовать логарифмировать таргет, если распределение с хвостом.")

    if not recs:
        recs.append("Структура данных выглядит ок, можно двигаться к фиче-инжинирингу и модели.")

    return recs
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
# 4. обучение базовой модели (без падений)
# ---------------------------------------------------------------------
from typing import Optional

def train_baseline(
    df: pd.DataFrame,
    target: str,
    task: str,
    return_model: bool = False,
) -> Optional[dict]:
    """
    Обучаем очень базовую модель.
    ВАЖНО: тут стараемся НИКОГДА не кидать исключения, чтобы /upload не падал.
    Если что-то не так с данными — просто вернём None.
    """
    try:
        if target not in df.columns:
            return None

        # приведём строки-похожие-на-числа
        df = _coerce_numeric(df)

        # выкинем строки, где нет таргета
        df = df[~df[target].isna()].copy()
        if df.shape[0] < 20:  # слишком мало данных
            return None

        y = df[target]
        X = df.drop(columns=[target])

        if X.shape[1] == 0:
            return None

        preprocessor = build_preprocessor(X)

        # КЛАССИФИКАЦИЯ
        if task == "classification":
            # если всего 1 класс — нечего учить
            if y.nunique() < 2:
                return None

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )

            # можно ли стратифицировать
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

            res = {
                "model_type": "RandomForestClassifier",
                "accuracy": acc,
                "f1": f1,
            }
            if return_model:
                res["pipeline"] = pipe
            return res

        # РЕГРЕССИЯ
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

            res = {
                "model_type": "RandomForestRegressor",
                "rmse": rmse,
            }
            if return_model:
                res["pipeline"] = pipe
            return res

        # если задача "eda" — не учим
        else:
            return None

    except Exception:
        # вообще на всё — тишина, просто без модели
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


def analyze_dataset(df: pd.DataFrame, eda: dict, task: dict) -> dict:
    """
    Вычисляем диагностическую инфу: константы, квази-константы, корреляции,
    дисбаланс (если классификация), NaN в таргете и т.п.
    Это отдаём в API, чтобы фронт мог подсветить.
    """
    problems: dict[str, object] = {}

    # 1) константы и почти константы
    constant_cols = []
    quasi_constant_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            constant_cols.append(col)
        elif nunique <= max(3, int(0.01 * len(df))):
            quasi_constant_cols.append(col)
    if constant_cols:
        problems["constant_features"] = constant_cols
    if quasi_constant_cols:
        problems["quasi_constant_features"] = quasi_constant_cols

    # 2) высокая корреляция по числовым
    num_df = df.select_dtypes(include=["number"])
    high_corr_pairs = []
    if num_df.shape[1] >= 2:
        corr = num_df.corr().abs()
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] >= 0.9:  # порог можно крутить
                    high_corr_pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    if high_corr_pairs:
        problems["high_corr_pairs"] = high_corr_pairs

    # 3) пропуски (в процентах)
    null_perc = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    high_nulls = null_perc[null_perc > 30].to_dict()  # >30% считаем много
    if high_nulls:
        problems["high_null_features"] = {k: float(v) for k, v in high_nulls.items()}

    # 4) NaN в таргете
    target = task.get("target")
    if target and target in df.columns:
        n_nan_target = int(df[target].isna().sum())
        if n_nan_target > 0:
            problems["target_has_nan"] = {"column": target, "nan_count": n_nan_target}

    # 5) дисбаланс классов
    if task.get("task") == "classification" and target and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        if len(vc) >= 2:
            max_c = int(vc.iloc[0])
            min_c = int(vc.iloc[-1])
            ratio = max_c / max(1, min_c)
            if ratio >= 5:  # дисбаланс
                problems["class_imbalance"] = {
                    "max_class": vc.index[0],
                    "max_count": max_c,
                    "min_class": vc.index[-1],
                    "min_count": min_c,
                    "ratio": float(ratio),
                }

    # 6) очень много категорий
    high_cardinality = []
    for col in df.select_dtypes(include=["object"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > 200:  # многа букаф
            high_cardinality.append({"column": col, "n_unique": int(nunique)})
    if high_cardinality:
        problems["high_cardinality"] = high_cardinality

    return problems

# agent/tools.py
from __future__ import annotations

import os
import json
import uuid
import base64
from io import BytesIO
from typing import Optional, Literal

import numpy as np
import pandas as pd
import re

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
# 1. Расширенный EDA
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
    eda["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
    eda["null_fractions"] = {c: float(df[c].isna().mean()) for c in df.columns}

    # stats по числовым
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

    # константы / почти константы
    constant_features = []
    quasi_constant_features = []
    for c in df.columns:
        uniq = df[c].nunique(dropna=True)
        if uniq <= 1:
            constant_features.append(c)
        else:
            top_frac = float(df[c].value_counts(normalize=True, dropna=False).iloc[0])
            if top_frac > 0.98:
                quasi_constant_features.append(c)
    eda["constant_features"] = constant_features
    eda["quasi_constant_features"] = quasi_constant_features

    # высокая корреляция
    high_corr_pairs = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
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


# ---------------------------------------------------------------------
# 2. угадывание таргета и типа задачи
# ---------------------------------------------------------------------
ID_LIKE = {"id", "ID", "Id", "index", "Rk", "rank"}


def _looks_like_id(colname: str) -> bool:
    return colname in ID_LIKE or re.search(r"id$", colname, re.IGNORECASE) is not None


def _guess_target(
    df: pd.DataFrame,
) -> tuple[Literal["eda", "classification", "regression"], Optional[str]]:
    lower_cols = {c.lower(): c for c in df.columns}

    # популярные
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
    """Определяем задачу и колонку-таргет."""
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
# 3. Приведение строк к числам и препроцессинг
# ---------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Пробуем строки, похожие на числа, привести к числам.
    Без FutureWarning.
    """
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == "object":
            ser = new_df[col].astype(str).str.replace(",", "").str.replace(" ", "")
            try:
                converted = pd.to_numeric(ser)
            except Exception:
                continue
            else:
                if converted.dtype != "object":
                    new_df[col] = converted
    return new_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# ---------------------------------------------------------------------
# 4. Обучение базовой модели безопасно
# ---------------------------------------------------------------------
def train_baseline(
    df: pd.DataFrame,
    target: str,
    task: str,
    return_model: bool = False,
) -> Optional[dict]:
    """
    Обучаем очень базовую модель.
    Возвращаем dict с метриками.
    Если что-то пошло не так — возвращаем None, чтобы /upload не падал.
    """
    try:
        if target not in df.columns:
            return None

        df = _coerce_numeric(df)

        # выбросим строки без таргета
        df = df[~df[target].isna()].copy()
        if df.shape[0] < 20:
            return None

        y = df[target]
        X = df.drop(columns=[target])

        if X.shape[1] == 0:
            return None

        preprocessor = build_preprocessor(X)

        # КЛАССИФИКАЦИЯ
        if task == "classification":
            if y.nunique() < 2:
                return None

            model = RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )

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
                n_estimators=200, random_state=42, n_jobs=-1
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

        # если задача только EDA
        return None

    except Exception:
        return None


# ---------------------------------------------------------------------
# 5. Отчёт текстом
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

    # числовые
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
# 6. Анализ проблем (для рекомендаций)
# ---------------------------------------------------------------------
def analyze_dataset(df: pd.DataFrame, eda: dict, task: dict) -> dict:
    """
    Собираем диагностическую инфу:
    - константы / квази-константы
    - высокая корреляция
    - много пропусков
    - NaN в таргете
    - дисбаланс классов
    - высокая кардинальность
    """
    problems: dict[str, object] = {}

    # 1) константы и почти константы
    constant_cols = []
    quasi_constant_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            constant_cols.append(col)
        else:
            top_frac = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_frac > 0.98:
                quasi_constant_cols.append(col)

    if constant_cols:
        problems["constant_features"] = constant_cols
    if quasi_constant_cols:
        problems["quasi_constant_features"] = quasi_constant_cols

    # 2) высокая корреляция
    num_df = df.select_dtypes(include=["number"])
    high_corr_pairs = []
    if num_df.shape[1] >= 2:
        corr = num_df.corr().abs()
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                cval = float(corr.iloc[i, j])
                if cval >= 0.9:
                    high_corr_pairs.append((cols[i], cols[j], cval))
    if high_corr_pairs:
        problems["high_corr_pairs"] = high_corr_pairs

    # 3) признаки с большим % пропусков
    null_perc = (df.isna().sum() / len(df) * 100)
    high_nulls = {col: float(round(p, 1)) for col, p in null_perc.items() if p >= 30.0}
    if high_nulls:
        problems["high_null_features"] = high_nulls

    # 4) NaN в таргете
    target = task.get("target")
    if target and target in df.columns:
        nan_cnt = int(df[target].isna().sum())
        if nan_cnt > 0:
            problems["target_has_nan"] = {
                "column": target,
                "nan_count": nan_cnt,
                "share": float(round(nan_cnt / len(df) * 100, 1)),
            }

    # 5) дисбаланс классов
    if task.get("task") == "classification" and target and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        if len(vc) >= 2:
            max_c = int(vc.iloc[0])
            min_c = int(vc.iloc[-1])
            ratio = max_c / max(1, min_c)
            if ratio >= 5:
                problems["class_imbalance"] = {
                    "max_class": vc.index[0],
                    "max_count": max_c,
                    "min_class": vc.index[-1],
                    "min_count": min_c,
                    "ratio": float(round(ratio, 1)),
                }

    # 6) высокая кардинальность категориальных
    high_cardinality = []
    for col in df.select_dtypes(include=["object"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > 200:
            high_cardinality.append(
                {"column": col, "n_unique": int(nunique)}
            )
    if high_cardinality:
        problems["high_cardinality"] = high_cardinality

    return problems


# ---------------------------------------------------------------------
# 7. Рекомендации на основе проблем
# ---------------------------------------------------------------------
def build_recommendations(
    df: pd.DataFrame,
    eda: dict,
    task: dict,
    problems: dict,
    model: dict | None,
    max_items: int = 6,
) -> list[str]:
    """
    На основе анализа собираем приоритезированный список рекомендаций.
    Возвращаем список строк (чтобы фронт мог их просто показать).
    """
    rec_objs: list[dict] = []

    # 1. критичные штуки
    # NaN в target
    if problems.get("target_has_nan"):
        info = problems["target_has_nan"]
        rec_objs.append({
            "priority": 100,
            "text": (
                f"В целевой колонке {info['column']} есть пропуски: {info['nan_count']} "
                f"({info['share']}%). Перед обучением их нужно удалить или заимпутить."
            ),
        })

    # много пропусков в признаках
    high_nulls = problems.get("high_null_features") or {}
    if high_nulls:
        top = list(high_nulls.items())[:5]
        pretty = ", ".join([f"{c} ({p}%)" for c, p in top])
        rec_objs.append({
            "priority": 90,
            "text": (
                f"Есть признаки с большим числом пропусков: {pretty}. "
                "Рекомендуется удалить такие столбцы или настроить имputaцию."
            ),
        })

    # дисбаланс
    if problems.get("class_imbalance"):
        ci = problems["class_imbalance"]
        rec_objs.append({
            "priority": 85,
            "text": (
                f"Обнаружен дисбаланс классов: {ci['max_class']}={ci['max_count']} vs "
                f"{ci['min_class']}={ci['min_count']} (≈{ci['ratio']}:1). "
                "Используй class_weight='balanced', stratify при train_test_split или oversampling."
            ),
        })

    # 2. важные, но не критичные
    consts = problems.get("constant_features") or []
    if consts:
        rec_objs.append({
            "priority": 70,
            "text": (
                f"Константные признаки: {', '.join(consts[:8])}. "
                "Их можно смело удалить — они не несут информации."
            ),
        })

    quasi = problems.get("quasi_constant_features") or []
    if quasi:
        rec_objs.append({
            "priority": 60,
            "text": (
                f"Почти константные признаки: {', '.join(quasi[:8])}. "
                "Проверь их полезность перед моделированием."
            ),
        })

    corr_pairs = problems.get("high_corr_pairs") or []
    if corr_pairs:
        short = [f"{a}↔{b} ({c:.2f})" for a, b, c in corr_pairs[:6]]
        rec_objs.append({
            "priority": 55,
            "text": (
                "Есть сильно коррелирующие пары признаков: "
                + ", ".join(short)
                + ". Можно сделать отбор признаков или регуляризацию, чтобы избежать переобучения."
            ),
        })

    high_card = problems.get("high_cardinality") or []
    if high_card:
        show = [f"{x['column']} ({x['n_unique']})" for x in high_card[:3]]
        rec_objs.append({
            "priority": 50,
            "text": (
                "Категориальные признаки с очень большим числом значений: "
                + ", ".join(show)
                + ". Лучше использовать CatBoost/target encoding/частотное кодирование."
            ),
        })

    # 3. по задаче
    if task.get("task") == "eda" or task.get("target") is None:
        rec_objs.append({
            "priority": 95,
            "text": "Целевой признак не найден. Укажи его явно при загрузке (поле target).",
        })
    elif task.get("task") == "regression":
        rec_objs.append({
            "priority": 40,
            "text": "Это регрессия — можно попробовать более сильные модели (CatBoostRegressor, LightGBM) и лог-трансформацию таргета, если есть перекос.",
        })
    elif task.get("task") == "classification":
        rec_objs.append({
            "priority": 40,
            "text": "Это классификация — кроме accuracy/f1 имеет смысл считать ROC-AUC и PR-AUC, особенно при дисбалансе.",
        })

    # 4. по модели
    if model is None:
        rec_objs.append({
            "priority": 30,
            "text": "Базовую модель не удалось обучить — нужно почистить данные или указать корректный target.",
        })
    else:
        mtype = model.get("model_type")
        if mtype == "RandomForestClassifier":
            rec_objs.append({
                "priority": 35,
                "text": "Текущая модель: RandomForestClassifier. Для повышения качества попробуй бустинг (CatBoost/XGBoost/LightGBM) и подбор гиперпараметров.",
            })
        elif mtype == "RandomForestRegressor":
            rec_objs.append({
                "priority": 35,
                "text": "Текущая модель: RandomForestRegressor. Можно усилить модель градиентным бустингом и фичеинжинирингом.",
            })

    # 5. сортируем и обрезаем
    rec_objs.sort(key=lambda x: x["priority"], reverse=True)
    recs = [r["text"] for r in rec_objs[:max_items]]

    return recs


# ---------------------------------------------------------------------
# 8. Графики → base64
# ---------------------------------------------------------------------
def make_plots_base64(df: pd.DataFrame) -> list[dict]:
    plots: list[dict] = []

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
# 9. Сохранение ран-а на диск (опционально)
# ---------------------------------------------------------------------
def save_run(run_data: dict, model_pipeline) -> str:
    run_id = str(uuid.uuid4())
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2)

    if model_pipeline is not None:
        joblib.dump(model_pipeline, os.path.join(run_dir, "model.joblib"))

    return run_id

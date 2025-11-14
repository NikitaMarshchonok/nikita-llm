# agent/tools.py
from __future__ import annotations

import os
import json
import uuid
import base64
from io import BytesIO
from typing import Optional, Literal

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Расширенный EDA
# ---------------------------------------------------------------------
def basic_eda(df: pd.DataFrame) -> dict:
    eda: dict = {
        "shape": list(df.shape),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }

    eda["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
    eda["null_fractions"] = {c: float(df[c].isna().mean()) for c in df.columns}

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
# 2. угадывание таргета и задачи
# ---------------------------------------------------------------------
ID_LIKE = {"id", "ID", "Id", "index", "Rk", "rank"}


def _looks_like_id(colname: str) -> bool:
    return colname in ID_LIKE or re.search(r"id$", colname, re.IGNORECASE) is not None


def _guess_target(
    df: pd.DataFrame,
) -> tuple[Literal["eda", "classification", "regression"], Optional[str]]:
    lower_cols = {c.lower(): c for c in df.columns}

    # явный target / label / class / y
    for cand in ("target", "label", "class", "y"):
        if cand in lower_cols:
            col = lower_cols[cand]
            if df[col].nunique() <= 50:
                return "classification", col
            else:
                return "regression", col

    # "псевдо-таргет" по маленькому числу уникальных значений
    for c in df.columns:
        if _looks_like_id(c):
            continue
        uniq = df[c].nunique(dropna=True)
        if 2 <= uniq <= 30:
            return "classification", c

    # fallback — числовой регрессионный таргет
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
    Пытаемся конвертировать object-колонки в числа:
    '1 234,5' → 1234.5
    Сейчас не используется, но оставляем для будущих апдейтов.
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
# 4. обучение базовой модели
# ---------------------------------------------------------------------
def train_baseline(
    df: pd.DataFrame,
    target: str,
    task: str,
    problems: dict | None = None,
    return_model: bool = False,
) -> Optional[dict]:
    """
    Обучаем RandomForest (классификация/регрессия) с учётом:
    - class_weight при дисбалансе
    - дропа id-подобных и константных колонок
    """
    try:
        if target not in df.columns:
            return {"model_type": "skipped", "reason": f"Колонка '{target}' не найдена."}

        problems = problems or {}

        # 0. убираем строки без таргета
        df = df[~df[target].isna()].copy()
        if df.shape[0] < 20:
            return {"model_type": "skipped", "reason": "Мало строк после удаления NaN в таргете."}

        # 1. что дропаем (id + константы)
        drop_cols: list[str] = []
        for c in problems.get("id_like", []) or []:
            if c in df.columns and c != target:
                drop_cols.append(c)
        for c in problems.get("constant_features", []) or []:
            if c in df.columns and c != target and c not in drop_cols:
                drop_cols.append(c)

        # 2. X, y
        X = df.drop(columns=[target] + drop_cols)
        y = df[target]

        if X.shape[1] == 0:
            return {"model_type": "skipped", "reason": "После очистки не осталось признаков."}

        preprocessor = build_preprocessor(X)

        # ===== классификация =====
        if task == "classification":
            if y.nunique() < 2:
                return {"model_type": "skipped", "reason": "В таргете один класс."}

            rf_kwargs = dict(n_estimators=200, random_state=42, n_jobs=-1)

            used_class_weight = False
            if problems.get("class_imbalance"):
                rf_kwargs["class_weight"] = "balanced"
                used_class_weight = True

            model = RandomForestClassifier(**rf_kwargs)

            # можно ли стратифицировать
            counts = y.value_counts(dropna=False)
            can_stratify = (counts >= 2).all()

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if can_stratify else None,
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
                "training_log": {
                    "dropped_columns": drop_cols,
                    "used_class_weight": used_class_weight,
                    "stratified_split": bool(can_stratify),
                },
            }

            # бинарка → ROC-AUC
            if y_val.nunique() == 2:
                try:
                    proba = pipe.predict_proba(X_val)[:, 1]
                    res["roc_auc"] = float(roc_auc_score(y_val, proba))
                except Exception:
                    pass

            if return_model:
                res["pipeline"] = pipe

            return res

        # ===== регрессия =====
        elif task == "regression":
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
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
                "training_log": {
                    "dropped_columns": drop_cols,
                    "used_class_weight": False,
                    "stratified_split": False,
                },
            }

            if return_model:
                res["pipeline"] = pipe

            return res

        # если задача непонятна
        return {"model_type": "skipped", "reason": "Задача не определена."}

    except Exception as e:
        return {"model_type": "skipped", "reason": f"Ошибка обучения: {e}"}


# ---------------------------------------------------------------------
# 5. отчёт (текст для UI и markdown)
# ---------------------------------------------------------------------
def build_report(
    df: pd.DataFrame,
    eda: dict,
    task: dict,
    model: dict | None,
    problems: dict | None = None,
) -> str:
    problems = problems or {}
    lines: list[str] = []

    rows, cols = eda.get("shape", (len(df), df.shape[1]))
    lines.append("📦 Данные")
    lines.append(f"• Размер: {rows} строк × {cols} колонок.")

    nulls = eda.get("nulls", {})
    nz = {k: v for k, v in nulls.items() if v > 0}
    if nz:
        lines.append("• Пропуски (топ):")
        for k, v in list(nz.items())[:8]:
            lines.append(f"   - {k}: {v}")

    num_stats = eda.get("numeric_stats", {})
    if num_stats:
        lines.append("• Числовые признаки (mean / std / min / max):")
        for name, st in list(num_stats.items())[:8]:
            lines.append(
                f"   - {name}: {st['mean']:.3f}/{st['std']:.3f}/{st['min']}/{st['max']}"
            )

    lines.append("")
    lines.append("🧩 Проблемы в данных")
    any_problems = (
        problems.get("constant_features")
        or problems.get("quasi_constant_features")
        or problems.get("high_corr_pairs")
        or problems.get("high_null_features")
        or problems.get("target_has_nan")
        or problems.get("class_imbalance")
        or problems.get("high_cardinality")
    )
    if not any_problems:
        lines.append("• Явных проблем не найдено ✅")
    else:
        if problems.get("target_has_nan"):
            info = problems["target_has_nan"]
            lines.append(
                f"• В таргете {info['column']} есть {info['nan_count']} пропусков — убрать перед обучением."
            )
        consts = problems.get("constant_features") or []
        if consts:
            lines.append("• Константные признаки: " + ", ".join(consts[:8]))
        qconsts = problems.get("quasi_constant_features") or []
        if qconsts:
            lines.append("• Почти константные: " + ", ".join(qconsts[:8]))
        corr_pairs = problems.get("high_corr_pairs") or []
        if corr_pairs:
            short = [f"{a}↔{b} ({c:.2f})" for a, b, c in corr_pairs[:6]]
            lines.append("• Сильная корреляция: " + ", ".join(short))
        high_nulls = problems.get("high_null_features") or {}
        if high_nulls:
            show = [f"{k} ({v:.1f}%)" for k, v in list(high_nulls.items())[:6]]
            lines.append("• Много пропусков: " + ", ".join(show))
        if problems.get("class_imbalance"):
            ci = problems["class_imbalance"]
            lines.append(
                f"• Дисбаланс классов: {ci['max_class']}:{ci['min_class']} ≈ {ci['ratio']:.1f}"
            )
        if problems.get("high_cardinality"):
            cols = [f"{x['column']} ({x['n_unique']})" for x in problems["high_cardinality"][:4]]
            lines.append("• Высокая кардинальность: " + ", ".join(cols))

    lines.append("")
    lines.append("🤖 Модель")
    if task.get("task") == "eda" or not task.get("target"):
        lines.append("• Целевой признак не найден — обучать нечего.")
    elif model is None:
        lines.append("• Модель не обучалась (невернулся результат).")
    elif model.get("model_type") == "skipped":
        lines.append("• Модель пропущена.")
        if model.get("reason"):
            lines.append("• Причина: " + model["reason"])
    else:
        lines.append(f"• Задача: {task['task']} по колонке “{task['target']}”.")
        lines.append(f"• Модель: {model.get('model_type')}.")
        if "accuracy" in model:
            lines.append(f"• accuracy = {model['accuracy']:.3f}")
        if "f1" in model:
            lines.append(f"• f1 = {model['f1']:.3f}")
        if "roc_auc" in model:
            lines.append(f"• ROC-AUC = {model['roc_auc']:.3f}")
        if "rmse" in model:
            lines.append(f"• RMSE = {model['rmse']:.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# 6. анализ проблем датасета
# ---------------------------------------------------------------------
def analyze_dataset(df: pd.DataFrame, task: dict) -> dict:
    problems: dict[str, object] = {}

    # ID / ключи
    id_like_cols: list[str] = []
    n_rows = len(df)
    for col in df.columns:
        col_l = col.lower()
        nunique = df[col].nunique(dropna=True)

        name_looks_like_id = (
            col_l == "id"
            or col_l.endswith("_id")
            or col_l in ("index", "idx", "rk", "rank")
        )
        value_looks_like_id = nunique > 0.9 * n_rows

        if name_looks_like_id or value_looks_like_id:
            id_like_cols.append(col)

    if id_like_cols:
        problems["id_like"] = id_like_cols

    # константы
    constant_cols = []
    quasi_constant_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            constant_cols.append(col)
        else:
            top_frac = float(df[col].value_counts(normalize=True, dropna=False).iloc[0])
            if top_frac > 0.98:
                quasi_constant_cols.append(col)
    if constant_cols:
        problems["constant_features"] = constant_cols
    if quasi_constant_cols:
        problems["quasi_constant_features"] = quasi_constant_cols

    # корреляции
    num_df = df.select_dtypes(include=["number"])
    high_corr_pairs = []
    if num_df.shape[1] >= 2:
        corr = num_df.corr().abs()
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr.iloc[i, j] >= 0.9:
                    high_corr_pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    if high_corr_pairs:
        problems["high_corr_pairs"] = high_corr_pairs

    # много пропусков
    null_perc = (df.isna().sum() / max(1, n_rows) * 100).sort_values(ascending=False)
    high_nulls = null_perc[null_perc > 30].to_dict()
    if high_nulls:
        problems["high_null_features"] = {k: float(v) for k, v in high_nulls.items()}

    # NaN в таргете
    target = task.get("target")
    if target and target in df.columns:
        n_nan_target = int(df[target].isna().sum())
        if n_nan_target > 0:
            problems["target_has_nan"] = {"column": target, "nan_count": n_nan_target}

    # дисбаланс классов
    if task.get("task") == "classification" and target and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        if len(vc) >= 2:
            max_c = int(vc.iloc[0])
            min_c = int(vc.iloc[-1])
            ratio = max_c / max(1, min_c)
            if ratio >= 5:
                problems["class_imbalance"] = {
                    "n_classes": int(len(vc)),
                    "max_class": str(vc.index[0]),
                    "max_count": max_c,
                    "min_class": str(vc.index[-1]),
                    "min_count": min_c,
                    "ratio": float(ratio),
                }

    # высокая кардинальность
    high_cardinality = []
    for col in df.select_dtypes(include=["object"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > 200 and col not in problems.get("id_like", []):
            high_cardinality.append({"column": col, "n_unique": int(nunique)})
    if high_cardinality:
        problems["high_cardinality"] = high_cardinality

    return problems


# ---------------------------------------------------------------------
# 6.0 роли колонок
# ---------------------------------------------------------------------
def detect_column_roles(df: pd.DataFrame, task: dict | None = None) -> dict:
    """
    Определяем "роль" каждой колонки:
      id / datetime / bool / numeric / categorical / text
    + флаг возможной утечки для числовых фич, сильно коррелирующих с таргетом.
    Возвращаем dict[col_name] = {...}.
    """
    task = task or {}
    target = task.get("target")
    n_rows = len(df)

    roles: dict[str, dict] = {}

    for col in df.columns:
        ser = df[col]
        info: dict[str, object] = {
            "role": None,
            "n_unique": int(ser.nunique(dropna=True)),
            "has_missing": bool(ser.isna().any()),
        }

        role: str | None = None
        nunique = info["n_unique"]

        # 1) id-подобные
        if _looks_like_id(col) or (n_rows > 0 and nunique > 0.9 * n_rows):
            role = "id"

        # 2) datetime
        if role is None:
            if pd.api.types.is_datetime64_any_dtype(ser):
                role = "datetime"
            elif ser.dtype == "object":
                if re.search(r"(date|time|timestamp)", col, re.IGNORECASE):
                    try:
                        parsed = pd.to_datetime(ser, errors="coerce")
                        if parsed.notna().mean() > 0.7:
                            role = "datetime"
                    except Exception:
                        pass

        # 3) bool
        if role is None:
            if pd.api.types.is_bool_dtype(ser):
                role = "bool"
            else:
                uniques = pd.Series(ser.dropna().unique())
                if 0 < len(uniques) <= 2:
                    role = "bool"

        # 4) numeric / text / categorical
        if role is None:
            if pd.api.types.is_numeric_dtype(ser):
                role = "numeric"
            else:
                try:
                    avg_len = float(
                        ser.dropna().astype(str).str.len().mean() or 0.0
                    )
                except Exception:
                    avg_len = 0.0

                if avg_len > 50 and nunique > 30:
                    role = "text"
                else:
                    role = "categorical"

        info["role"] = role

        # 5) возможная утечка (для числовых таргетов)
        if target and target in df.columns and col != target:
            tgt = df[target]
            if pd.api.types.is_numeric_dtype(ser) and pd.api.types.is_numeric_dtype(tgt):
                try:
                    corr = float(tgt.corr(ser))
                    if abs(corr) > 0.98:
                        info["possible_leakage"] = True
                        info["leakage_corr"] = corr
                    else:
                        info["possible_leakage"] = False
                except Exception:
                    pass

        roles[col] = info

    return roles


# ---------------------------------------------------------------------
# 6.1 оценка здоровья датасета
# ---------------------------------------------------------------------
def evaluate_dataset_health(eda: dict, problems: dict) -> dict:
    score = 100
    reasons: list[str] = []

    if problems.get("high_null_features"):
        score -= 15
        reasons.append("много признаков с пропусками (>30%)")

    if problems.get("constant_features"):
        score -= 10
        reasons.append("есть полностью константные признаки")

    if problems.get("high_corr_pairs"):
        score -= 10
        reasons.append("есть сильно коррелирующие признаки")

    if problems.get("class_imbalance"):
        score -= 20
        reasons.append("сильный дисбаланс классов")

    if problems.get("high_cardinality"):
        score -= 5
        reasons.append("категориальные с очень большим числом значений")

    score = max(30, min(100, score))

    if score >= 80:
        level = "green"
    elif score >= 55:
        level = "yellow"
    else:
        level = "red"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------
# 6.2 план экспериментов как у mid+ DS
# ---------------------------------------------------------------------
def build_experiment_plan(
    task: dict | None,
    problems: dict | None,
    dataset_health: dict | None,
    model: dict | None,
    column_roles: dict | None,
) -> list[dict]:
    """
    Строим план экспериментов как сделал бы mid+ DS.
    Формат шага:
    {
      "priority": "now" | "next" | "later",
      "title": "Краткий заголовок",
      "description": "Что сделать и зачем",
      "tags": ["data", "model", "metrics"]
    }
    """
    task = task or {}
    problems = problems or {}
    dataset_health = dataset_health or {}
    column_roles = column_roles or {}

    plan: list[dict] = []

    def add(priority, title, description, tags=None):
        plan.append({
            "priority": priority,
            "title": title,
            "description": description,
            "tags": tags or [],
        })

    # --- NOW: критичные проблемы с данными ---

    if problems.get("target_has_nan"):
        info = problems["target_has_nan"]
        add(
            "now",
            "Очистить таргет от NaN",
            f"В колонке {info.get('column')} {info.get('nan_count')} пропусков. "
            "Удали строки с NaN или заполни их, иначе метрики будут искажены.",
            ["data", "target"],
        )

    if problems.get("high_null_features"):
        add(
            "now",
            "Обработать признаки с большими пропусками",
            "Есть признаки с >30% пропусков. Реши: дропнуть их, заимпутить или добавить флаги 'is_null'.",
            ["data", "missing"],
        )

    # дисбаланс классов
    if problems.get("class_imbalance") and task.get("task") == "classification":
        add(
            "now",
            "Сделать устойчивый train/test при дисбалансе",
            "Используй stratify в train_test_split, class_weight='balanced' или oversampling (SMOTE/RandomOverSampler).",
            ["data", "imbalance"],
        )

    # возможная утечка фич
    leak_cols = [
        c for c, info in column_roles.items()
        if isinstance(info, dict) and info.get("possible_leakage")
    ]
    if leak_cols:
        add(
            "now",
            "Проверить возможную утечку признаков",
            "Найдены признаки, почти идеально коррелирующие с таргетом: "
            + ", ".join(leak_cols[:5])
            + ". Убедись, что это не «target в другой форме».",
            ["data", "leakage"],
        )

    # --- NEXT: фичи и модели ---

    # high cardinality → CatBoost / target encoding
    if problems.get("high_cardinality"):
        add(
            "next",
            "Обработать high-cardinality категориальные признаки",
            "Для колонок с очень большим числом значений попробуй CatBoost или target/frequency encoding, "
            "чтобы не раздувать one-hot.",
            ["features", "categorical"],
        )

    # datetime
    has_datetime = any(
        isinstance(info, dict) and info.get("role") == "datetime"
        for info in column_roles.values()
    )
    if has_datetime:
        add(
            "next",
            "Сделать time-based фичи и сплит",
            "Из дат выдели year/month/day/dayofweek, is_weekend. "
            "Для временных рядов используй TimeSeriesSplit вместо обычного сплита.",
            ["features", "datetime"],
        )

    # text
    has_text = any(
        isinstance(info, dict) and info.get("role") == "text"
        for info in column_roles.values()
    )
    if has_text:
        add(
            "next",
            "Обработать текстовые признаки",
            "Для текстовых колонок попробуй TF-IDF + линейную модель или text-эмбеддинги. "
            "Сейчас они, вероятно, игнорируются или кодируются грубо.",
            ["features", "text"],
        )

    # модель: апгрейд с RF
    if model and model.get("model_type") in ("RandomForestClassifier", "RandomForestRegressor"):
        add(
            "next",
            "Попробовать бустинг и подбор гиперпараметров",
            "Текущая модель — RandomForest. Следующий шаг — XGBoost/LightGBM/CatBoost "
            "с подбором гиперпараметров (GridSearchCV/Optuna).",
            ["model", "tuning"],
        )

    # --- LATER: метрики и продакшен ---

    if task.get("task") == "classification":
        add(
            "later",
            "Добавить продвинутые метрики",
            "Помимо accuracy и F1, посчитай ROC-AUC и PR-AUC, особенно при дисбалансе.",
            ["metrics"],
        )
    elif task.get("task") == "regression":
        add(
            "later",
            "Проверить R², MAE и распределение ошибок",
            "Посмотри, не заваливается ли модель на крайних значениях таргета, построй plot y_true vs y_pred.",
            ["metrics"],
        )

    add(
        "later",
        "Задуматься о продакшен-пайплайне",
        "Когда качество устроит — заверни модель в отдельный сервис с версионированием, мониторингом и логами.",
        ["production"],
    )

    return plan


# ---------------------------------------------------------------------
# 7. текстовые рекомендации
# ---------------------------------------------------------------------
def build_recommendations(
    df: pd.DataFrame,
    eda: dict,
    task: dict,
    problems: dict,
    model: dict | None,
) -> list[str]:
    recs: list[str] = []

    id_like = set(problems.get("id_like", []))

    consts = [c for c in (problems.get("constant_features") or []) if c not in id_like]
    if consts:
        recs.append(
            f"Есть полностью константные признаки: {', '.join(consts[:8])} — можно удалить перед моделированием."
        )

    quasi = [c for c in (problems.get("quasi_constant_features") or []) if c not in id_like]
    if quasi:
        recs.append(
            f"Есть почти константные признаки: {', '.join(quasi[:8])} — стоит проверить их пользу."
        )

    corr_pairs = problems.get("high_corr_pairs") or []
    if corr_pairs:
        short = []
        for a, b, corr in corr_pairs[:6]:
            if a in id_like and b in id_like:
                continue
            short.append(f"{a}↔{b} ({corr:.2f})")
        if short:
            recs.append(
                "Есть сильно коррелирующие пары признаков: "
                + ", ".join(short)
                + " — можно сделать отбор признаков или регуляризацию."
            )

    high_nulls = problems.get("high_null_features") or {}
    if high_nulls:
        show = [f"{k} ({v:.1f}%)" for k, v in list(high_nulls.items())[:6] if k not in id_like]
        if show:
            recs.append(
                "Есть признаки с большим числом пропусков: "
                + ", ".join(show)
                + " — заполни/удали/сделай отдельный флаг."
            )

    # дисбаланс
    if problems.get("class_imbalance"):
        ci = problems["class_imbalance"]
        recs.append(
            f"Найден дисбаланс классов ({ci['max_class']}:{ci['min_class']} ≈ {ci['ratio']:.1f}). "
            "Используй class_weight='balanced', stratify при train_test_split или oversampling."
        )

    if problems.get("target_has_nan"):
        info = problems["target_has_nan"]
        recs.append(
            f"В целевой колонке {info['column']} есть пропуски ({info['nan_count']}) — нужно убрать их перед обучением."
        )

    high_card = problems.get("high_cardinality") or []
    if high_card:
        cols = [f"{x['column']} ({x['n_unique']})" for x in high_card[:4] if x["column"] not in id_like]
        if cols:
            recs.append(
                "Есть категориальные признаки с большим числом значений: "
                + ", ".join(cols)
                + " — лучше использовать CatBoost/target encoding/частотное кодирование."
            )

    if task.get("task") == "eda":
        recs.append("Целевой признак не найден — можно явно указать target при загрузке.")
    elif task.get("task") == "regression":
        recs.append("Для регрессии можно попробовать более сильные модели (CatBoostRegressor, LightGBM).")
    elif task.get("task") == "classification":
        recs.append("Для классификации имеет смысл посчитать ROC-AUC и PR-AUC, особенно при дисбалансе.")

    if model is None:
        recs.append("Модель не обучалась — скорее всего, нет подходящего target или данных слишком мало.")
    else:
        if model.get("model_type") == "RandomForestClassifier":
            recs.append("Текущая модель — RandomForestClassifier. Можно улучшить бустингом и подбором гиперпараметров.")
        if model.get("model_type") == "RandomForestRegressor":
            recs.append("Текущая модель — RandomForestRegressor. Можно улучшить CatBoost/LightGBM.")

    return recs


# ---------------------------------------------------------------------
# 8. графики → base64
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
# 9. сохранение run на диск (опционально)
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


# ---------------------------------------------------------------------
# 10. Краткий статус и next actions
# ---------------------------------------------------------------------
def build_analysis_status(task: dict, problems: dict, model: dict | None) -> dict:
    problems = problems or {}
    task = task or {}

    status: dict = {
        "task": task.get("task", "eda"),
        "target": task.get("target"),
        "dataset": "ok",
        "model": "ok",
        "notes": [],
    }

    if problems.get("target_has_nan"):
        info = problems["target_has_nan"]
        status["dataset"] = "warning"
        status["notes"].append(
            f"В таргете {info.get('column', '?')} есть {info.get('nan_count', 0)} пропусков"
        )

    if problems.get("high_null_features"):
        status["dataset"] = "warning"
        status["notes"].append("Есть признаки с >30% пропусков")

    ci = problems.get("class_imbalance")
    if ci:
        max_cls = ci.get("max_class") or ci.get("most_common_class") or "?"
        min_cls = ci.get("min_class") or ci.get("rarest_class") or "?"
        ratio = ci.get("ratio", "?")
        status["dataset"] = "warning"
        status["notes"].append(f"Дисбаланс классов: {max_cls}:{min_cls} ≈ {ratio}")

    if model is None:
        status["model"] = "not_trained"
        status["notes"].append("Модель не обучалась")
    elif model.get("model_type") == "skipped":
        status["model"] = "skipped"
        if model.get("reason"):
            status["notes"].append(f"Модель пропущена: {model['reason']}")
    else:
        status["model"] = "ok"

    return status


def build_next_actions(task: dict, problems: dict, model: dict | None) -> list[str]:
    actions: list[str] = []
    task = task or {}
    problems = problems or {}

    if problems.get("target_has_nan"):
        actions.append("Очисти таргет от NaN (удали строки или заполни).")

    if problems.get("high_null_features"):
        actions.append("Обработай признаки с >30% пропусков: дроп/импутация/флаг.")

    if problems.get("class_imbalance") and task.get("task") == "classification":
        actions.append("При обучении укажи class_weight='balanced' или сделай oversampling.")

    if problems.get("high_cardinality"):
        actions.append("Для колонок с высокой кардинальностью используй CatBoost/target encoding.")

    if task.get("task") == "classification":
        actions.append("Посчитай ROC-AUC и PR-AUC, если важны редкие классы.")
    elif task.get("task") == "regression":
        actions.append("Попробуй бустинг (CatBoost/LightGBM) для улучшения качества.")

    if model is not None and model.get("model_type") == "RandomForestClassifier":
        actions.append("Сделай подбор гиперпараметров или попробуй бустинг (XGBoost/LightGBM).")

    if not actions:
        actions.append("Датасет выглядит ок — можно двигаться к фичам/подбору модели.")

    return actions


# ---------------------------------------------------------------------
# 11. Расширенная сводка по таргету (для LLM и UI)
# ---------------------------------------------------------------------
def summarize_target(df: pd.DataFrame, task: dict) -> dict:
    """
    Возвращает json-friendly сводку по таргету:
    - есть ли таргет
    - сколько пропусков / строк
    - для классификации: число классов + топ классы
    - для регрессии: min/max/mean/std
    """
    target = (task or {}).get("target")
    task_type = (task or {}).get("task", "eda")

    if not target or target not in df.columns:
        return {
            "has_target": False,
            "reason": "target_not_found",
        }

    col = df[target]
    n_rows = len(df)
    n_missing = int(col.isna().sum())

    base = {
        "has_target": True,
        "target": target,
        "task": task_type,
        "n_rows": int(n_rows),
        "n_missing": n_missing,
        "missing_frac": float(n_missing / n_rows) if n_rows else 0.0,
    }

    # КЛАССИФИКАЦИЯ
    if task_type == "classification":
        vc = col.value_counts(dropna=False)
        n_classes = int(len(vc))

        top_classes = []
        for cls, cnt in vc.head(20).items():
            label = "__NaN__" if pd.isna(cls) else str(cls)
            top_classes.append({
                "label": label,
                "count": int(cnt),
                "share": float(cnt / n_rows) if n_rows else 0.0,
            })

        base.update({
            "n_classes": n_classes,
            "top_classes": top_classes,
        })
        return base

    # РЕГРЕССИЯ / ЧИСЛОВОЙ ТАРГЕТ
    if pd.api.types.is_numeric_dtype(col):
        base.update({
            "min": float(col.min(skipna=True)) if col.notna().any() else 0.0,
            "max": float(col.max(skipna=True)) if col.notna().any() else 0.0,
            "mean": float(col.mean(skipna=True)) if col.notna().any() else 0.0,
            "std": float(col.std(skipna=True)) if col.notna().any() else 0.0,
        })
    else:
        base.update({
            "note": "target_is_not_numeric",
        })

    return base


# ---------------------------------------------------------------------
# 12. Ранжирование проблем по важности (для UI и LLM)
# ---------------------------------------------------------------------
def rank_problems(problems: dict) -> list[dict]:
    """
    Превращает dict из analyze_dataset в список структур:
    [{key, severity, message, data}, ...]
    чтобы фронт мог показать сначала high, потом medium, потом low.
    """
    problems = problems or {}
    ranked: list[dict] = []

    # 1. NaN в таргете — всегда high
    if problems.get("target_has_nan"):
        info = problems["target_has_nan"]
        ranked.append({
            "key": "target_has_nan",
            "severity": "high",
            "message": f"В таргете {info.get('column')} есть {info.get('nan_count')} пропусков — убери перед обучением.",
            "data": info,
        })

    # 2. дисбаланс
    if problems.get("class_imbalance"):
        ci = problems["class_imbalance"]
        ranked.append({
            "key": "class_imbalance",
            "severity": "high",
            "message": (
                "Найден дисбаланс классов — используй class_weight='balanced', "
                "stratify при train_test_split или oversampling."
            ),
            "data": ci,
        })

    # 3. много пропусков в фичах
    if problems.get("high_null_features"):
        ranked.append({
            "key": "high_null_features",
            "severity": "medium",
            "message": "Есть признаки с >30% пропусков — заполни/удали/сделай флаг.",
            "data": problems["high_null_features"],
        })

    # 4. константы
    if problems.get("constant_features"):
        ranked.append({
            "key": "constant_features",
            "severity": "medium",
            "message": "Есть полностью константные признаки — можно удалить.",
            "data": problems["constant_features"],
        })

    # 5. почти константные
    if problems.get("quasi_constant_features"):
        ranked.append({
            "key": "quasi_constant_features",
            "severity": "low",
            "message": "Есть почти константные признаки — проверь полезность.",
            "data": problems["quasi_constant_features"],
        })

    # 6. корреляции
    if problems.get("high_corr_pairs"):
        ranked.append({
            "key": "high_corr_pairs",
            "severity": "low",
            "message": "Есть сильно коррелирующие признаки — можно отфильтровать.",
            "data": problems["high_corr_pairs"][:20],
        })

    # 7. высокая кардинальность
    if problems.get("high_cardinality"):
        ranked.append({
            "key": "high_cardinality",
            "severity": "medium",
            "message": "Категориальные признаки с большим числом значений — лучше CatBoost/target encoding.",
            "data": problems["high_cardinality"],
        })

    return ranked


# ---------------------------------------------------------------------
# 13. Идеи новых фич
# ---------------------------------------------------------------------
def auto_feature_suggestions(df: pd.DataFrame) -> list[str]:
    """
    Очень простые подсказки, какие фичи можно докрутить.
    Это не меняет df, это просто идеи для пользователя/агента.
    """
    suggestions: list[str] = []

    # даты
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "timestamp" in col.lower():
            suggestions.append(
                f"Из колонки {col} можно вынести year/month/day/dayofweek и, возможно, признак 'is_weekend'."
            )

    # большие категориальные
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        nun = df[col].nunique(dropna=True)
        if nun > 200:
            suggestions.append(
                f"Категориальная колонка {col} имеет много значений ({nun}) — стоит использовать target/frequency encoding или CatBoost."
            )
        elif 2 < nun <= 50:
            suggestions.append(
                f"Колонка {col} — аккуратная категориальная, можно one-hot (у тебя это уже есть в пайплайне)."
            )

    # числовые с пропусками
    null_frac = df.isna().mean()
    for col, frac in null_frac.items():
        if frac > 0.0 and col not in obj_cols:
            suggestions.append(
                f"В числовой колонке {col} есть пропуски ({frac:.1%}) — можно добавить бинарный флаг 'is_{col}_missing'."
            )

    if not suggestions:
        suggestions.append("Явных идей по новым фичам нет — можно переходить к отбору признаков/моделям.")

    return suggestions


# ---------------------------------------------------------------------
# 14. Feature importance из pipeline
# ---------------------------------------------------------------------
def extract_feature_importance(pipeline) -> list[dict]:
    """
    Достаём топ-важные признаки из обученного pipeline, если модель умеет feature_importances_.
    Возвращаем список словарей: {"feature": ..., "importance": ...}
    Если достать нельзя — возвращаем пустой список.
    """
    try:
        model = pipeline.named_steps.get("model")
        preprocess = pipeline.named_steps.get("preprocess")

        if model is None or not hasattr(model, "feature_importances_"):
            return []

        importances = model.feature_importances_

        feature_names: list[str] = []
        if hasattr(preprocess, "get_feature_names_out"):
            feature_names = list(preprocess.get_feature_names_out())
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        items = []
        for name, imp in zip(feature_names, importances):
            items.append({"feature": str(name), "importance": float(imp)})

        items.sort(key=lambda x: x["importance"], reverse=True)
        return items[:50]
    except Exception:
        return []


# ---------------------------------------------------------------------
# 15. Код-подсказки (snippets) под найденные проблемы
# ---------------------------------------------------------------------
def build_code_hints(problems: dict, task: dict) -> list[dict]:
    """
    Возвращаем список "код-подсказок", чтобы фронт мог показать
    готовые куски кода под обнаруженные проблемы.
    Формат элемента:
    {
        "title": "Обработать дисбаланс",
        "snippet": "from imblearn.over_sampling import SMOTE\n...",
        "reason": "Найден дисбаланс классов"
    }
    """
    problems = problems or {}
    task = task or {}
    hints: list[dict] = []

    # 1) дисбаланс классов
    if problems.get("class_imbalance") and task.get("task") == "classification":
        hints.append({
            "title": "Классификация с class_weight='balanced'",
            "reason": "Найден дисбаланс классов",
            "snippet": (
                "from sklearn.ensemble import RandomForestClassifier\n"
                "clf = RandomForestClassifier(class_weight='balanced', n_estimators=300, random_state=42)\n"
                "clf.fit(X_train, y_train)\n"
                "preds = clf.predict(X_val)"
            ),
        })
        hints.append({
            "title": "Oversampling через imblearn",
            "reason": "Найден дисбаланс классов",
            "snippet": (
                "from imblearn.over_sampling import RandomOverSampler\n"
                "ros = RandomOverSampler(random_state=42)\n"
                "X_res, y_res = ros.fit_resample(X_train, y_train)\n"
                "# дальше обучай модель на X_res, y_res"
            ),
        })

    # 2) много категорий → CatBoost
    if problems.get("high_cardinality"):
        hints.append({
            "title": "CatBoost для колонок с высокой кардинальностью",
            "reason": "Есть категориальные признаки с большим числом значений",
            "snippet": (
                "from catboost import CatBoostClassifier\n"
                "# индексами укажи категориальные признаки\n"
                "cat_features = [0, 3, 5]\n"
                "model = CatBoostClassifier(depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=False)\n"
                "model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))"
            ),
        })

    # 3) пропуски в таргете
    if problems.get("target_has_nan"):
        col = problems["target_has_nan"]["column"]
        hints.append({
            "title": "Удалить строки с NaN в таргете",
            "reason": f"В целевой колонке {col} есть пропуски",
            "snippet": (
                f"df = df[~df['{col}'].isna()].copy()\n"
                "# дальше делай train/test split"
            ),
        })

    # 4) много пропусков в фичах
    if problems.get("high_null_features"):
        hints.append({
            "title": "Пайплайн с SimpleImputer",
            "reason": "Есть признаки с >30% пропусков",
            "snippet": (
                "from sklearn.impute import SimpleImputer\n"
                "from sklearn.pipeline import Pipeline\n"
                "from sklearn.ensemble import RandomForestClassifier\n"
                "pipe = Pipeline([\n"
                "    ('imputer', SimpleImputer(strategy='median')),\n"
                "    ('model', RandomForestClassifier())\n"
                "])\n"
                "pipe.fit(X_train, y_train)"
            ),
        })

    # 5) просто заготовка под задачу
    if task.get("task") == "regression":
        hints.append({
            "title": "Базовая регрессия (RF)",
            "reason": "Задача определена как регрессия",
            "snippet": (
                "from sklearn.ensemble import RandomForestRegressor\n"
                "model = RandomForestRegressor(n_estimators=300, random_state=42)\n"
                "model.fit(X_train, y_train)\n"
                "preds = model.predict(X_val)"
            ),
        })
    elif task.get("task") == "classification":
        hints.append({
            "title": "Базовая классификация (RF)",
            "reason": "Задача определена как классификация",
            "snippet": (
                "from sklearn.ensemble import RandomForestClassifier\n"
                "model = RandomForestClassifier(n_estimators=300, random_state=42)\n"
                "model.fit(X_train, y_train)\n"
                "preds = model.predict(X_val)"
            ),
        })

    return hints

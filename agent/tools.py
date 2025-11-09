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

    # пропуски
    eda["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
    eda["null_fractions"] = {c: float(df[c].isna().mean()) for c in df.columns}

    # числовая статистика
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

    # константы / квази-константы
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

    # сильно коррелирующие пары
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

    for cand in ("target", "label", "class", "y"):
        if cand in lower_cols:
            col = lower_cols[cand]
            if df[col].nunique() <= 50:
                return "classification", col
            else:
                return "regression", col

    for c in df.columns:
        if _looks_like_id(c):
            continue
        uniq = df[c].nunique(dropna=True)
        if 2 <= uniq <= 30:
            return "classification", c

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
    try:
        if target not in df.columns:
            return None

        df = _coerce_numeric(df)
        df = df[~df[target].isna()].copy()
        if df.shape[0] < 20:
            return None

        y = df[target]
        X = df.drop(columns=[target])
        if X.shape[1] == 0:
            return None

        preprocessor = build_preprocessor(X)

        # классификация
        if task == "classification":
            if y.nunique() < 2:
                return None

            rf_kwargs = dict(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )

            # если нашли дисбаланс — включаем веса
            if problems and problems.get("class_imbalance"):
                rf_kwargs["class_weight"] = "balanced"

            model = RandomForestClassifier(**rf_kwargs)

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

            # если бинарка — считаем AUC
            if y_val.nunique() == 2:
                try:
                    proba = pipe.predict_proba(X_val)[:, 1]
                    auc = float(roc_auc_score(y_val, proba))
                    res["roc_auc"] = auc
                except Exception:
                    pass

            if return_model:
                res["pipeline"] = pipe
            return res

        # регрессия
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

        return None

    except Exception:
        return None


# ---------------------------------------------------------------------
# 5. отчёт
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
        consts = problems.get("constant_features") or []
        if consts:
            lines.append(
                "• Константные признаки: " + ", ".join(consts[:8]) + " — можно удалить."
            )
        qconst = problems.get("quasi_constant_features") or []
        if qconst:
            lines.append(
                "• Почти константные признаки: "
                + ", ".join(qconst[:8])
                + " — проверь их полезность."
            )
        corr_pairs = problems.get("high_corr_pairs") or []
        if corr_pairs:
            short = [f"{a}↔{b} ({c:.2f})" for a, b, c in corr_pairs[:6]]
            lines.append(
                "• Сильно коррелирующие пары: " + ", ".join(short) + " — возможен отбор фич."
            )
        high_nulls = problems.get("high_null_features") or {}
        if high_nulls:
            show = [f"{k} ({v:.1f}%)" for k, v in list(high_nulls.items())[:6]]
            lines.append("• Много пропусков: " + ", ".join(show))
        if problems.get("target_has_nan"):
            info = problems["target_has_nan"]
            lines.append(
                f"• В таргете {info['column']} есть {info['nan_count']} пропусков — убрать перед обучением."
            )
        if problems.get("class_imbalance"):
            ci = problems["class_imbalance"]
            lines.append(
                f"• Дисбаланс классов: {ci['max_class']}:{ci['min_class']} ≈ {ci['ratio']:.1f} — используй class_weight/oversampling."
            )
        high_card = problems.get("high_cardinality") or []
        if high_card:
            cols_txt = [f"{x['column']} ({x['n_unique']})" for x in high_card[:4]]
            lines.append(
                "• Высокая кардинальность категориальных: " + ", ".join(cols_txt)
            )

    lines.append("")
    lines.append("🤖 Модель")
    if task.get("task") == "eda" or not task.get("target"):
        lines.append("• Целевой признак не найден — обучать нечего.")
    elif model is None:
        lines.append("• Модель не обучалась — мало данных или один класс.")
    else:
        lines.append(f"• Задача: {task['task']} по колонке “{task['target']}”.")
        lines.append(f"• Модель: {model['model_type']}.")
        if "accuracy" in model:
            lines.append(f"• accuracy = {model['accuracy']:.3f}")
        if "f1" in model:
            lines.append(f"• f1 = {model['f1']:.3f}")
        if "roc_auc" in model:
            lines.append(f"• ROC-AUC = {model['roc_auc']:.3f}")
        if "rmse" in model:
            lines.append(f"• RMSE = {model['rmse']:.3f}")

    lines.append("")
    lines.append("🪜 Что сделать дальше")
    lines.append("• Посмотри на константы/корреляции и сократи фичи.")
    if task.get("task") == "classification":
        lines.append("• Для дисбаланса — class_weight='balanced' или oversampling.")
        lines.append("• Посчитай ROC-AUC/PR-AUC, если важен редкий класс.")
    if task.get("task") == "regression":
        lines.append("• Попробуй бустинг (CatBoost/LightGBM) для улучшения RMSE.")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# 6. анализ проблем (один вариант!)
# ---------------------------------------------------------------------
def analyze_dataset(df: pd.DataFrame, eda: dict, task: dict) -> dict:
    """
    Сигналы по датасету: константы, квазиконстанты, корреляции,
    много пропусков, дисбаланс, высокая кардинальность, ID-колонки.
    eda мы сейчас не используем, но передаём для единообразия с app.py
    """
    problems: dict[str, object] = {}

    # 0) детект ID / ключей
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
        value_looks_like_id = nunique > 0.9 * n_rows  # почти уникально

        if name_looks_like_id or value_looks_like_id:
            id_like_cols.append(col)

    if id_like_cols:
        problems["id_like"] = id_like_cols

    # 1) константы и почти константы
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

    # 2) высокая корреляция по числовым
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

    # 3) пропуски (в процентах)
    null_perc = (df.isna().sum() / max(1, n_rows) * 100).sort_values(ascending=False)
    high_nulls = null_perc[null_perc > 30].to_dict()
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
            if ratio >= 5:
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
        if nunique > 200 and col not in problems.get("id_like", []):
            high_cardinality.append({"column": col, "n_unique": int(nunique)})
    if high_cardinality:
        problems["high_cardinality"] = high_cardinality

    return problems



# ---------------------------------------------------------------------
# 7. рекомендации (под сигнатуру из app.py)
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

    if problems.get("target_has_nan"):
        info = problems["target_has_nan"]
        recs.append(
            f"В целевой колонке {info['column']} есть пропуски ({info['nan_count']}) — нужно убрать их перед обучением."
        )

    if problems.get("class_imbalance"):
        ci = problems["class_imbalance"]
        recs.append(
            f"Найден дисбаланс классов ({ci['max_class']}:{ci['min_class']} ≈ {ci['ratio']:.1f}). "
            "Используй class_weight='balanced', stratify при train_test_split или oversampling."
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
# 9. сохранение на диск (по желанию)
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

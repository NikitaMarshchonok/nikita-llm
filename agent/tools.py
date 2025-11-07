# agent/tools.py

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------------
def basic_eda(df: pd.DataFrame) -> dict:
    """Простой EDA: размер, типы, пропуски, немного статистики."""
    eda = {
        "shape": list(df.shape),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
    }

    # добавим чуть-чуть статистики по числовым — полезно в отчёте
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats = {}
    for c in numeric_cols[:20]:  # не спамим сотней колонок
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
# 2. угадывание таргета и типа задачи
# ---------------------------------------------------------------------
ID_LIKE = {"id", "ID", "Id", "index", "Rk", "rank"}


def _looks_like_id(colname: str) -> bool:
    return colname in ID_LIKE or re.search(r"id$", colname, re.IGNORECASE) is not None


def _guess_target(df: pd.DataFrame) -> tuple[Literal["eda", "classification", "regression"], Optional[str]]:
    """
    Если пользователь target не дал — попробуем сами.
    Алгоритм простой:
      1. сначала ищем 'label', 'target', 'y'
      2. потом небольшой категориальный столбец
      3. потом числовой
      4. если ничего — только EDA
    """
    lower_cols = {c.lower(): c for c in df.columns}

    # 1) популярные имена
    for cand in ("target", "label", "class", "y"):
        if cand in lower_cols:
            col = lower_cols[cand]
            if df[col].nunique() <= 50:
                return "classification", col
            else:
                return "regression", col

    # 2) маленькие категориальные — хороши для классификации
    for c in df.columns:
        if _looks_like_id(c):
            continue
        uniq = df[c].nunique(dropna=True)
        if 2 <= uniq <= 30:
            return "classification", c

    # 3) любой числовой, который не id
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if _looks_like_id(c):
            continue
        if df[c].nunique() > 1:
            return "regression", c

    # не нашли
    return "eda", None


def detect_task(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    """Определяем задачу и колонку-таргет."""
    if target is not None and target in df.columns:
        # пользователь сказал явно
        nunique = df[target].nunique()
        if df[target].dtype == "object" or nunique <= 30:
            task = "classification"
        else:
            task = "regression"
        return {"task": task, "target": target}

    # иначе угадываем
    task, tgt = _guess_target(df)
    return {"task": task, "target": tgt}
    

# ---------------------------------------------------------------------
# 3. препроцессинг
# ---------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Попробуем строки, похожие на числа, привести к числам.
    Это как раз нужно для твоих cricket / sports датасетов.
    """
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == "object":
            # попробуем
            converted = pd.to_numeric(new_df[col].str.replace(",", "").str.replace(" ", ""), errors="ignore")
            # если стало числом — заменим
            if converted.dtype != "object":
                new_df[col] = converted
    return new_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Строим sklearn-препроцессор под наш датафрейм."""
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
def train_baseline(df: pd.DataFrame, target: str, task: str) -> Optional[dict]:
    """
    Обучаем очень базовую модель поверх авто-препроцессинга.
    Возвращаем метрики и тип модели.
    """
    if target not in df.columns:
        return None

    # сначала попытаемся привести строковые числа
    df = _coerce_numeric(df)

    y = df[target]
    X = df.drop(columns=[target])

    # если всё ещё пусто
    if X.shape[1] == 0:
        return None

    preprocessor = build_preprocessor(X)

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 50 else None
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc = float(accuracy_score(y_val, preds))
        f1 = float(f1_score(y_val, preds, average="weighted"))
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": acc,
            "f1": f1,
        }

    elif task == "regression":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        reg = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_val)

        rmse = float(mean_squared_error(y_val, preds, squared=False))
        return {
            "model_type": "RandomForestRegressor",
            "rmse": rmse,
        }

    else:
        # 'eda' и т.п.
        return None

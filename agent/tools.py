# agent/tools.py

from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer


# ---------- 1. EDA ----------
def basic_eda(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": list(df.shape),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
    }


# ---------- 2. определение задачи ----------
def detect_task(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Если target передали и он есть в колонках — определяем тип задачи.
    Если не передали — просто EDA.
    """
    if target is not None and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            task = "regression"
        else:
            task = "classification"
        return {"task": task, "target": target}

    return {"task": "eda", "target": None}


# ---------- 3. обучение базовой модели ----------
def train_baseline(df: pd.DataFrame, target: str, task: str) -> Dict[str, Any]:
    """
    Простой табличный пайплайн с заполнением пропусков.
    """
    # 1) выбрасываем строки, где нет таргета
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    # 2) train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) какие колонки числовые, какие категориальные
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in numeric_features]

    # 4) трансформеры с имьютерами
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
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # 5) модель
    if task == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    # 6) учим
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    # 7) метрики
    if task == "classification":
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": float(acc),
            "f1": float(f1),
        }
    else:
        # пытаемся посчитать rmse “правильно”, а если sklearn старый — руками
        try:
            rmse = mean_squared_error(y_test, preds, squared=False)
        except TypeError:
            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5

        return {
            "model_type": "RandomForestRegressor",
            "rmse": float(rmse),
        }

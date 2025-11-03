# agent/tools.py

from io import BytesIO
from typing import Optional, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ------------- ЗАГРУЗКА ДАННЫХ ------------- #

def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Чтение CSV из загруженного файла (для веба)."""
    return pd.read_csv(BytesIO(file_bytes))


def load_csv_from_path(path: str) -> pd.DataFrame:
    """Чтение CSV с диска (для локальных тестов)."""
    return pd.read_csv(path)


# ------------- БАЗОВЫЙ EDA ------------- #

def basic_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Простая разведочная аналитика, чтобы показать пользователю."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isna().sum().to_dict(),
        "head": df.head(5).to_dict(orient="records"),
    }


# ------------- ОПРЕДЕЛЕНИЕ ЗАДАЧИ ------------- #

def detect_task(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Пытается понять, что за задача:
    - если target указан → классификация или регрессия по нему
    - если нет → ищет колонку с небольшим числом уникальных значений и делает классификацию
    иначе отдаёт режим "просто EDA"
    """
    # если пользователь сказал таргет — работаем с ним
    if target and target in df.columns:
        y = df[target]
        if y.nunique() <= 20:
            return {"task": "classification", "target": target}
        else:
            return {"task": "regression", "target": target}

    # автоопределение
    for col in df.columns:
        nunique = df[col].nunique()
        # простое правило: немного уникальных → скорее класс
        if 2 <= nunique <= 20:
            return {"task": "classification", "target": col}

    # не нашли нормальный таргет
    return {"task": "eda", "target": None}


# ------------- ОБУЧЕНИЕ БЕЙЗЛАЙНА ------------- #

def train_baseline(df: pd.DataFrame, target: str, task: str) -> Dict[str, Any]:
    """
    Обучает простую модель (RandomForest) и возвращает метрики.
    ВАЖНО: перед обучением кодируем ВСЕ нечисловые колонки через get_dummies,
    чтобы не падать на строках и датах.
    """
    # отделяем фичи/таргет
    X = df.drop(columns=[target])
    y = df[target]

    # one-hot для всех категорий/строк
    # drop_first=True слегка уменьшает размерность
    X_encoded = pd.get_dummies(X, drop_first=True)

    # сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    if task == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": acc,
            "f1": f1,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_encoded.shape[1],
        }

    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        return {
            "model_type": "RandomForestRegressor",
            "rmse": rmse,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_encoded.shape[1],
        }

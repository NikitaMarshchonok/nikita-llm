# agent/tools.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Чтение CSV из загруженного файла (для веба)."""
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


def load_csv_from_path(path: str) -> pd.DataFrame:
    """Чтение CSV с диска (для локальных тестов)."""
    return pd.read_csv(path)


def basic_eda(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isna().sum().to_dict(),
        "head": df.head(5).to_dict(orient="records"),
    }


def detect_task(df: pd.DataFrame, target: str | None = None) -> dict:
    # если пользователь сказал, какая колонка таргет
    if target and target in df.columns:
        y = df[target]
        if y.nunique() <= 20:
            return {"task": "classification", "target": target}
        else:
            return {"task": "regression", "target": target}

    # попробовать угадать
    for col in df.columns:
        nunique = df[col].nunique()
        if 2 <= nunique <= 20:
            return {"task": "classification", "target": col}

    return {"task": "eda", "target": None}


def train_baseline(df: pd.DataFrame, target: str, task: str) -> dict:
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
        }

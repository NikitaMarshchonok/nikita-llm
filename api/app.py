# api/app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO

from agent.tools import basic_eda, detect_task, train_baseline

app = FastAPI(title="Nikita DS Agent")


def read_csv_safely(file_bytes: bytes) -> pd.DataFrame:
    """
    Пытаемся аккуратно прочитать CSV:
    - сначала как обычный csv с запятой
    - если не вышло — пробуем ; (часто в русских выгрузках)
    - если не вышло — кидаем исключение
    """
    bio = BytesIO(file_bytes)

    # 1 попытка — обычная
    try:
        bio.seek(0)
        return pd.read_csv(bio)
    except Exception:
        pass

    # 2 попытка — с ;
    try:
        bio.seek(0)
        return pd.read_csv(bio, sep=";")
    except Exception:
        pass

    # 3 попытка — пусть падает
    bio.seek(0)
    return pd.read_csv(bio)


@app.get("/")
def root():
    return {"msg": "Nikita DS Agent is running"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    try:
        contents = await file.read()

        # читаем csv "умно"
        df = read_csv_safely(contents)

        eda = basic_eda(df)
        task = detect_task(df, target=target)

        model_res = None
        if task["task"] != "eda" and task["target"]:
            model_res = train_baseline(df, task["target"], task["task"])

        return JSONResponse(
            {
                "filename": file.filename,
                "eda": eda,
                "task": task,
                "model": model_res,
            }
        )

    except Exception as e:
        # вместо 500 без объяснения вернём нормальный текст
        return JSONResponse(
            status_code=400,
            content={
                "error": "failed_to_process_file",
                "details": str(e),
                "hint": "проверь разделитель (',' или ';'), названия колонок и target",
            },
        )

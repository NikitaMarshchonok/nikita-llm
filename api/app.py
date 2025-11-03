# api/app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd

from agent.tools import basic_eda, detect_task, train_baseline


app = FastAPI(title="Nikita DS Agent")


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    """
    Принимает CSV, делает EDA и пытается обучить базовую модель.
    """
    contents = await file.read()
    # читаем csv из байтов
    from io import BytesIO

    df = pd.read_csv(BytesIO(contents))

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


@app.get("/")
def root():
    return {"msg": "Nikita DS Agent is running"}

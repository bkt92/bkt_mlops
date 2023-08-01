from fastapi import FastAPI, Request
import sys
sys.path.append('./src')
from model_predictor import ModelPredictor
from pydantic import BaseModel
from utils import AppPath
from contextlib import asynccontextmanager

class Data(BaseModel):
    id: str | None | int | float
    rows: list
    columns: list

config_path = {}

config_path[1] = (AppPath.MODEL_CONFIG_DIR / "phase-3_prob-1.yaml").as_posix()
config_path[2] = (AppPath.MODEL_CONFIG_DIR / "phase-3_prob-2.yaml").as_posix()

predictor = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    predictor[1] = await ModelPredictor.create(config_file_path=config_path[1])
    predictor[2] = await ModelPredictor.create(config_file_path=config_path[2])
    # Clear cache
    await predictor[1].clear_cache()
    await predictor[2].clear_cache()
    yield
    # Clean up the ML models and release the resources
    predictor.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Welcome to bkt api for mlops competition!"}

@app.post("/phase-3/prob-1/predict")
async def predict(data: Data, request: Request):
    response = await predictor[1].predict_proba(data)
    return response

@app.post("/phase-3/prob-2/predict")
async def predict(data: Data, request: Request):
    response = await predictor[2].predict(data)
    return response

#@app.post("/clearcache")
#async def clearcache(request: Request):
#    await predictor[1].clear_cache()
#    await predictor[2].clear_cache()
#    return {"msg": "Cache cleared"}
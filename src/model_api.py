import sys
sys.path.append('./src')
#from dataclasses import dataclass
from pydantic import BaseModel
from blacksheep import Application, Request, FromJSON
from model_predictor import ModelPredictor
from utils import AppPath

#@dataclass
#class Data:
#    id: str
#    rows: list
#    columns: list

class Data(BaseModel):
    id: str | None | int | float
    rows: list
    columns: list

config_path = {}

config_path[1] = (AppPath.MODEL_CONFIG_DIR / "phase-2_prob-1.yaml").as_posix()
config_path[2] = (AppPath.MODEL_CONFIG_DIR / "phase-2_prob-2.yaml").as_posix()

predictor = {}

app = Application(show_error_details=True)
get = app.router.get
post = app.router.post

@app.lifespan
async def lifespan():
    # Load the ML model
    predictor[1] = await ModelPredictor.create(config_file_path=config_path[1])
    predictor[2] = await ModelPredictor.create(config_file_path=config_path[2])
    # Clear cache
    await predictor[1].clear_cache()
    await predictor[2].clear_cache()
    yield
    # Clean up the ML models and release the resources
    await predictor[1].stop()
    await predictor[2].stop()
    predictor.clear()

@get("/")
async def root():
    return {"message": "Welcome to bkt api for mlops competition!"}

@post("/phase-2/prob-1/predict")
async def predict(data: Data):
    response = await predictor[1].predict_proba(data)
    return response

@post("/phase-2/prob-2/predict")
async def predict(data: Data):
    response = await predictor[2].predict(data)
    return response

#@post("/clearcache")
#async def clearcache(request: Request):
#    await predictor[1].clear_cache()
#    await predictor[2].clear_cache()
#    return {"msg": "Cache cleared"}

import sys
sys.path.append('./src')
from pydantic import BaseModel
import falcon.asgi
from model_predictor import ModelPredictor
from utils import AppPath

config_path = {}

config_path[1] = (AppPath.MODEL_CONFIG_DIR / "phase-2_prob-1.yaml").as_posix()
config_path[2] = (AppPath.MODEL_CONFIG_DIR / "phase-2_prob-2.yaml").as_posix()

class Data(BaseModel):
    id: str
    rows: list
    columns: list

predictor = {}

class init_predictor:
    async def process_startup(self, scope, event):
        predictor[1] = await ModelPredictor.create(config_file_path=config_path[1])
        predictor[2] = await ModelPredictor.create(config_file_path=config_path[2])
        await predictor[1].clear_cache()
        await predictor[2].clear_cache()
    async def process_shutdown(self, scope, event):
        predictor.clear()

class Predictor1:
    async def on_post(self, req, resp):
        data = await req.get_media()
        resp.media = await predictor[1].predict_proba(Data(**data))
        resp.status = falcon.HTTP_200

class Predictor2:
    async def on_post(self, req, resp):
        data = await req.get_media()
        resp.media = await predictor[2].predict(Data(**data))
        resp.status = falcon.HTTP_200

#class Clearcache:
#    async def on_post(self, req, resp):
#        await predictor[1].clear_cache()
#        await predictor[2].clear_cache()
#        resp.media = {"msg": "Cache cleared"}
#        resp.status = falcon.HTTP_200

app = falcon.asgi.App(middleware=[init_predictor()])
app.add_route('/phase-2/prob-1/predict', Predictor1())
app.add_route('/phase-2/prob-2/predict', Predictor2())
#app.add_route('/clearcache', Clearcache())
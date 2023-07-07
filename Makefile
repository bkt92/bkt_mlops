# teardown
teardown:
	make predictor_down
	make mlflow_down

# mlflow
build_api:
	docker build -t  bkt92/model_predictor:latest -f deploy/api.Dockerfile .

init_api:
	. .venv/bin/activate
	export MLFLOW_TRACKING_URI=http://localhost:5000
	export REDIS_ENDPOINT=localhost
	export MEMCACHED_ENDPOIN=localhost
	python src/init_startup.py

api_hyper_fal:
	hypercorn -b 0.0.0.0:8000 -w 4 src.model_falcon:app

api_gra_fal:
	granian --interface asgi --host 0.0.0.0 --port 8000 --workers 4 src.model_api:app

api_uvi_fal:
	uvicorn --host 0.0.0.0 --port 8000 --workers 6 src.model_falcon:app

api_hyper_bl:
	hypercorn -b 0.0.0.0:8000 -w 4 src.model_api:app

api_gra_bl:
	granian --interface asgi --host 0.0.0.0 --port 8000 --workers 4 src.model_api:app

api_uvi_bl:
	uvicorn --host 0.0.0.0 --port 8000 --workers 4 src.model_api:app


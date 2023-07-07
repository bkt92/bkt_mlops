build_api:
	docker build -t  bkt92/model_predictor:latest -f deploy/api.Dockerfile .

push_docker:
	docker push bkt92/model_predictor:latest
	docker push bkt92/mlflow:latest

build_mlflow:
	 docker build -t bkt92/mlflow:latest  -f deploy/mlflow.Dockerfile .

init_api:
	. .venv/bin/activate
	export MLFLOW_TRACKING_URI=http://localhost:5000
	export REDIS_ENDPOINT=localhost
	export MEMCACHED_ENDPOINT=localhost
	python src/init_startup.py

start_docker_api:
	docker-compose -f deploy/docker-compose.yml up -d

stop_docker_api:
	docker-compose -f deploy/docker-compose.yml down

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

ping_server:
	ansible server -m ping -i deploy/inventory.ini
	# -k for asking password
deploy_to_server:
	ansible-playbook deploy/deploy_cloud.yml -i deploy/inventory.ini

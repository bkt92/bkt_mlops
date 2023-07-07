# teardown
teardown:
	make predictor_down
	make mlflow_down

# mlflow
docker_build:
	docker-compose -f deployment/mlflow/docker-compose.yml up -d

api_hyper_fal:
	. venv/bin/activate
	export MLFLOW_TRACKING_URI=http://localhost:5000
	export REDIS_ENDPOINT=localhost
	export MEMCACHED_ENDPOIN=localhost
	python src/init_startup.py
	hypercorn -b 0.0.0.0:8000 -w 4 src.model_falcon:app


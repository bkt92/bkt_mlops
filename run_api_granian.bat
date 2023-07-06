call .venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
granian --interface asgi --host 0.0.0.0 --port 8000 --workers 1 --threads 12 src.model_api:app
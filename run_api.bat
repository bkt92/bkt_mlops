call .venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
set SET_CACHE_REQUEST=True
uvicorn --host 0.0.0.0 --port 8000 --workers 1 src.model_api:app
call .venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
uvicorn --host 0.0.0.0 --port 8000 --workers 1 --reload src.model_falcon:app
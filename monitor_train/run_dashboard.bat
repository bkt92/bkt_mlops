call ../.venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
python src/dashboard.py
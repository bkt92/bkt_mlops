call .venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
python src/init_startup.py
hypercorn -b 0.0.0.0:8000 -w 12 src.model_falcon:app
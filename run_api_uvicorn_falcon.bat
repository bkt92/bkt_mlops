call .venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
set MEMCACHED_ENDPOINT=localhost
echo Running Startup Script
python src/init_startup.py
echo Start API Server
uvicorn --host 0.0.0.0 --port 8000 --workers 1 --reload src.model_falcon:app
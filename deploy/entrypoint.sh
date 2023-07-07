python src/init_startup.py
hypercorn -b 0.0.0.0:8000 -w 6 src.model_api:app

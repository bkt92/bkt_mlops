#!/bin/bash -x

PWD=`pwd`

activate () {
    . $PWD/.venv/bin/activate
}

activate

export MLFLOW_TRACKING_URI=http://localhost:5000
export REDIS_ENDPOINT=localhost
python src/init_startup.py
hypercorn -b 0.0.0.0:8000 -w 4 src.model_falcon:app

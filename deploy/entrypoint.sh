#!/bin/bash
export PATH="/opt/venv/bin:$PATH"
worker=1
framework='blacksheep'
help()
{
    echo "Usage: entrypoint [ -s | --server ]
               [ -k | --kernel ]
               [ -f | --framework  ]
               [ -w | --worker  ]
               [ -h | --help  ]"
    exit 2
}

SHORT=s:,k:,f:,w:,h
LONG=server:,kernel:,framework:,worker:,help

OPTS=$(getopt -a -n weather --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
  help
fi

eval set -- "$OPTS"

while :
do
  case "$1" in
    -s | --server )
      server="$2"
      shift 2
      ;;
    -k | --kernel )
      kernel="$2"
      shift 2
      ;;
    -f | --framework )
      framework="$2"
      shift 2
      ;;
    -w | --worker )
      worker="$2"
      shift 2
      ;;
    -h | --help)
      help
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      help
      ;;
  esac
done

echo "Start load and compile models"
#python src/init_startup.py

if [ $server = 'uvicorn' ]  && [ -z "$kernel" ];then
	if [ $framework = 'fastapi' ];then
		echo "Running fastapi uvicorn:" $worker "worker"
		uvicorn --host 0.0.0.0 --port 8000 --workers $worker src.model_fastapi:app
	elif [ $framework = 'falcon' ];then
		echo "Running falcon uvicorn:" $worker "worker"
		uvicorn --host 0.0.0.0 --port 8000 --workers $worker src.model_falcon:app
	else
		echo "Running default(blksheep) uvicorn:" $worker "worker"
		uvicorn --host 0.0.0.0 --port 8000 --workers $worker src.model_api:app
	fi
elif [ $server = 'uvicorn' ]  && [ $kernel = 'uvloop' ];then
        if [ $framework = 'fastapi' ];then
                echo "Running fastapi uvicorn+uvloop:" $worker "worker"
		uvicorn --host 0.0.0.0 --port 8000 --workers $worker --loop uvloop src.model_fastapi:app
        elif [ $framework = 'falcon' ];then
                echo "Running falcon uvicorn+uvloop:" $worker "worker"
		uvicorn --host 0.0.0.0 --port 8000 --workers $worker --loop uvloop src.model_falcon:app
        else
                echo "Running default(blksheep) uvicorn+uvloop:" $worker "worker"
		uvicorn --host 0.0.0.0 --port 8000 --workers $worker --loop uvloop src.model_api:app
        fi
elif [ $server = 'hypercorn' ]  && [ -z "$kernel" ];then
        if [ $framework = 'fastapi' ];then
                echo "Running fastapi hypercorn:" $worker "worker"
		hypercorn -b 0.0.0.0:8000 -w $worker src.model_fastapi:app
        elif [ $framework = 'falcon' ];then
                echo "Running falcon hypercorn:" $worker "worker"
		hypercorn -b 0.0.0.0:8000 -w $worker src.model_falcon:app
        else
                echo "Running default(blksheep) uvicorn:" $worker "worker"
		hypercorn -b 0.0.0.0:8000 -w $worker src.model_api:app
        fi
elif [ $server = 'hypercorn' ]  && [ $kernel = 'uvloop' ];then
        if [ $framework = 'fastapi' ];then
                echo "Running fastapi hypercorn+uvloop:" $worker "worker"
		hypercorn -b 0.0.0.0:8000 -w $worker -k uvloop src.model_fastapi:app
        elif [ $framework = 'falcon' ];then
                echo "Running falcon hypercorn+uvloop:" $worker "worker"
		hypercorn -b 0.0.0.0:8000 -w $worker -k uvloop src.model_falcon:app
        else
                echo "Running default(blksheep) hypercorn+uvloop:" $worker "worker"
		hypercorn -b 0.0.0.0:8000 -w $worker -k uvloop src.model_api:app
        fi
else
	echo "Default blacksheep+hypercorn:" $worker "worker"
	hypercorn -b 0.0.0.0:8000 -w $worker src.model_api:app
fi

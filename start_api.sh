#!/bin/bash

# === Configuration ===
HOST="127.0.0.1"      # 127.0.0.1 = localhost
PORT="5000"
WORKERS=4
APP_MODULE="run:app"
LOG_FILE="gunicorn.log"
HEALTH_ENDPOINT="http://${HOST}:${PORT}/health"

# === 1. Start Gunicorn in Background ===
echo "Starting API server with Gunicorn..."
echo "Host: $HOST, Port: $PORT, Workers: $WORKERS"
echo "Logs will be written to $LOG_FILE"

# the final '&' runs in background.
gunicorn --workers $WORKERS --bind ${HOST}:${PORT} $APP_MODULE > $LOG_FILE 2>&1 &

# gunicorn --workers 4 --bind 0.0.0.0:5000 run:app

# === 2. Health Check Loop ===
echo "Waiting for server to be ready at $HEALTH_ENDPOINT..."

# Loop until curl successfully connects (exit code 0)
# -f: Fail silently (non-zero exit code on 4xx/5xx)
# -s: Silent mode (don't show progress)
# > /dev/null: Discard the output, we only care about the exit code
while ! curl -f -s $HEALTH_ENDPOINT > /dev/null; do
    echo "Server not ready yet. Retrying in 2 seconds..."
    sleep 2
done

# === 3. Success ===
echo ""
echo "---------------------------------"
echo "API is UP AND RUNNING!"
echo "Final health check response:"
curl $HEALTH_ENDPOINT
echo ""
echo "---------------------------------"
echo "Server is running in the background."
echo "To view logs, run: tail -f $LOG_FILE"
echo "To stop the server, run: pkill gunicorn"

#!/bin/sh
set -eu

APP_PORT="${APP_PORT:-9050}"
if [ "${APP_RELOAD:-1}" = "1" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}" --reload --reload-dir /app/app
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}"

#!/bin/sh
set -eu

python -m app.bootstrap_db
alembic upgrade head

APP_PORT="${APP_PORT:-9150}"
if [ "${APP_RELOAD:-1}" = "1" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}" --reload --reload-dir /app/app
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}"

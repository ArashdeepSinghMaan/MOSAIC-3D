#!/usr/bin/env bash
set -euo pipefail
case "${1:-api}" in
  api)   exec uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 ;;
  mvp)   shift; exec python scripts/run_mvp.py "$@" ;;
  test)  exec pytest ;;
  *)     exec "$@" ;;
esac

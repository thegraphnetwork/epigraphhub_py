#!/usr/bin/env bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd ../.. && pwd )"

# force to get env variables from .env
# shellcheck disable=SC2046,SC2002
export $(cat "${PROJECT_DIR}/docker/.env" | xargs)

epigraphhub-config \
  --db-host "${POSTGRES_HOST}" \
  --db-port "${POSTGRES_PORT}" \
  --db-credential "dev-epigraphhub:${POSTGRES_EPIGRAPH_DB}/${POSTGRES_EPIGRAPH_USER}/${POSTGRES_EPIGRAPH_PASSWORD}"

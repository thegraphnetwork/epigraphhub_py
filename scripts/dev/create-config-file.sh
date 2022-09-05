#!/usr/bin/env bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd ../.. && pwd )"

# force to get env variables from .env
# shellcheck disable=SC2046,SC2002
export $(cat "${PROJECT_DIR}/docker/.env" | xargs)

DB_URI="${POSTGRES_EPIGRAPH_USER}:${POSTGRES_EPIGRAPH_PASSWORD}"
DB_URI="${DB_URI}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_EPIGRAPH_DB}"

epigraphhub-config --name "dev-epigraphhub" --db-uri "${DB_URI}"

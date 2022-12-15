#!/usr/bin/env bash

set -e

echo "${1}" | grep -E "^(chore|fix|docs|feat|test)(\(.+\))?\:"

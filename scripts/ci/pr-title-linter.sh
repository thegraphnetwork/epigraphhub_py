#!/usr/bin/env bash

set -e

echo ${1} | grep -Ei "^(chore|fix|docs|feat|breaking change|test)\:?(\(.+\)\:)?"

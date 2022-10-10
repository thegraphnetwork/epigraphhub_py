#!/usr/bin/env bash

set -e

echo ${1} | egrep -i "^(chore|fix|docs|feat|breaking change|test)\:?(\(.+\)\:)?"

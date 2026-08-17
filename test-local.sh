#!/usr/bin/env sh
set -eu

exec python -m unittest discover -s tests -t . -v

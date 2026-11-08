#!/bin/sh
# Starts as root only to make sure the mounted ./data folder is writable,
# then runs the app as the unprivileged "appuser".
set -e
mkdir -p /code/data
chown -R appuser:appuser /code/data
exec setpriv --reuid=appuser --regid=appuser --init-groups "$@"

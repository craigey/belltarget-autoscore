#!/usr/bin/env bash
# Start the scoring app from its own folder, using the virtual environment if one exists.
cd "$(dirname "$0")" || exit 1
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . venv/bin/activate
fi
exec python3 app.py

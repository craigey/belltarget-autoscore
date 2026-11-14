#!/usr/bin/env bash
# Installs belltarget-autoscore as a service that starts at boot.
# Run from the project folder:  bash install.sh
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_USER="${SUDO_USER:-$USER}"
SERVICE=belltarget-autoscore

echo "Installing system packages..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip v4l-utils

echo "Creating virtual environment..."
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# An earlier install used the service name "air-target". Stop and remove it if present.
if systemctl list-unit-files air-target.service >/dev/null 2>&1; then
  echo "Removing the old air-target service..."
  sudo systemctl disable --now air-target.service || true
  sudo rm -f /etc/systemd/system/air-target.service
fi

echo "Installing systemd service (user: $APP_USER)..."
sudo tee /etc/systemd/system/$SERVICE.service > /dev/null << UNIT
[Unit]
Description=belltarget-autoscore: automatic scoring for air rifle bell targets
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
SupplementaryGroups=video
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/start.sh
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now $SERVICE
echo "Done. Open http://<pi-address>:5000 in a browser."
echo "Status: systemctl status $SERVICE   Logs: journalctl -u $SERVICE -f"

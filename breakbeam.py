"""
Break-beam bell sensors, one per camera. A falling edge on a sensor's pin is one
pellet through that camera's bell hole.

Round start and stop come from the dashboard, not from the beams.
"""

import time

import config_store
from detection import register_bell_event

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

DEBOUNCE_MS = 500
_configured = False
_watching = False


def setup_gpio():
    """Set the BCM numbering once. Safe to call more than once."""
    global _configured
    if not GPIO_AVAILABLE:
        print("⚠️ GPIO not available (not running on a Raspberry Pi)")
        return False
    if not _configured:
        GPIO.setmode(GPIO.BCM)
        _configured = True
    return True


def _on_break(cam_id):
    register_bell_event(time.time(), cam_id)


def watch_breakbeam():
    """
    Register one falling-edge detector per configured sensor. Returns immediately.
    Config: "bell_sensors": {"<camera id>": <GPIO pin>, ...}
    """
    global _watching
    if not config_store.get("enable_breakbeam", False):
        return
    sensors = config_store.get("bell_sensors", {})
    if _watching or not sensors or not setup_gpio():
        return
    for cam, pin in sensors.items():
        cam_id, pin = int(cam), int(pin)
        try:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(pin, GPIO.FALLING,
                                  callback=lambda ch, c=cam_id: _on_break(c),
                                  bouncetime=DEBOUNCE_MS)
            print(f"👁️ Bell sensor for camera {cam_id} on GPIO {pin}")
        except RuntimeError as e:
            print(f"⚠️ Bell sensor for camera {cam_id} (GPIO {pin}) not started: {e}")
    _watching = True

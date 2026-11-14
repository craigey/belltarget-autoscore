"""
Camera controls through V4L2 (v4l2-ctl), the same path the web sliders use.

Every value is clamped to the range the camera reports, written with
v4l2-ctl, and read back so the caller knows what the camera actually holds.
OpenCV's property setters are not used for controls, because their scaling
differs by camera and they can silently overwrite values set by v4l2-ctl.
"""

import re
import subprocess

_ranges = {}   # device path -> {control name: {min, max, step, default, value}}

# Control names that differ between kernel versions
ALIASES = {
    "white_balance_automatic": ["white_balance_automatic", "white_balance_temperature_auto"],
    "exposure_time_absolute": ["exposure_time_absolute", "exposure_absolute"],
    "focus_automatic_continuous": ["focus_automatic_continuous", "focus_auto"],
    "auto_exposure": ["auto_exposure", "exposure_auto"],
}  # any name in a group works, whichever the kernel uses

_LINE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")


def device_for(cam_id):
    return f"/dev/video{cam_id}"


def _run(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def load_ranges(cam_id, refresh=False):
    dev = device_for(cam_id)
    if dev in _ranges and not refresh:
        return _ranges[dev]
    code, out = _run(["v4l2-ctl", "-d", dev, "-L"])
    ranges = {}
    if code == 0:
        for line in out.splitlines():
            m = _LINE.match(line)
            if not m:
                continue
            name, _kind, rest = m.groups()
            attrs = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", rest)}
            if "min" in attrs and "max" in attrs:
                ranges[name] = {
                    "min": attrs["min"], "max": attrs["max"],
                    "step": max(1, attrs.get("step", 1)),
                    "default": attrs.get("default"), "value": attrs.get("value"),
                }
    _ranges[dev] = ranges
    return ranges


def _candidates(name):
    """Every name that refers to the same control, the requested one first."""
    for group in ALIASES.values():
        if name in group:
            return [name] + [g for g in group if g != name]
    return [name]


def _resolve(cam_id, name):
    ranges = load_ranges(cam_id)
    for candidate in _candidates(name):
        if candidate in ranges:
            return candidate
    return None


def clamp(value, info):
    """Clamp to min/max and snap to the control's step."""
    v = max(info["min"], min(info["max"], int(round(value))))
    step = info["step"]
    v = info["min"] + round((v - info["min"]) / step) * step
    return max(info["min"], min(info["max"], v))


def supported(cam_id, name):
    return _resolve(cam_id, name) is not None


def get_range(cam_id, name):
    real = _resolve(cam_id, name)
    return load_ranges(cam_id)[real] if real else None


def get_control(cam_id, name):
    real = _resolve(cam_id, name)
    if real is None:
        return None
    code, out = _run(["v4l2-ctl", "-d", device_for(cam_id), "--get-ctrl", real])
    m = re.search(r":\s*(-?\d+)", out)
    return int(m.group(1)) if code == 0 and m else None


def set_control(cam_id, name, value):
    """
    Returns (ok, actual, message).
    ok is True only if the camera reports back the value that was written.
    """
    real = _resolve(cam_id, name)
    if real is None:
        return False, None, f"{name} is not supported by this camera"
    info = load_ranges(cam_id)[real]
    target = clamp(value, info)
    code, out = _run(["v4l2-ctl", "-d", device_for(cam_id), "-c", f"{real}={target}"])
    if code != 0:
        return False, None, out.strip() or "v4l2-ctl failed"
    actual = get_control(cam_id, name)
    notes = []
    if target != value:
        notes.append(f"requested {value}, clamped to {target}")
    if actual != target:
        notes.append(f"camera reports {actual}")
    return actual == target, actual, "; ".join(notes)


def apply_profile(cam_id, profile):
    """Apply a dict of control -> value in order. Returns a result per control."""
    results = {}
    for name, value in profile.items():
        ok, actual, msg = set_control(cam_id, name, value)
        results[name] = {"ok": ok, "actual": actual, "msg": msg}
        if not ok:
            print(f"⚠️ {name}={value} on camera {cam_id}: {msg or 'not applied'}")
    return results

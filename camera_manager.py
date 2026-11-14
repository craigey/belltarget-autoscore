import cv2
import numpy as np
import time
import json
import subprocess
import threading
import calibration
import camera_controls

DEFAULT_PROFILE = {
    "auto_exposure": 3,
    "exposure_dynamic_framerate": 1,
    "focus_automatic_continuous": 1,
    "power_line_frequency": 2,
    "brightness": 128,
    "contrast": 128,
    "saturation": 128,
    "sharpness": 128,
    "gain": 0,
}

# Load config
import config_store

caps = {}

# One capture thread per camera keeps only the newest frame, so every frame used is
# current. Streams and the detection thread read these; nothing else reads the camera.
_raw = {}          # camera id -> newest frame, after rotation and zoom
_overlay = {}      # camera id -> newest frame with rings and detection drawn on it
_frame_lock = threading.Lock()
_capture_threads = {}
exposure_locked = {}
zoom_levels = {}
rotation_angles = {}  # Rotation per camera
white_balance_locked = {}

def init_cameras(camera_ids=[0]):
    """
    Initialize USB cameras with optimized settings.
    
    Args:
        camera_ids: list of camera IDs (default [0])
    """
    global caps, exposure_locked, zoom_levels, rotation_angles, white_balance_locked

    for cam_id in camera_ids:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # avoid stale frames
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera {cam_id}")

        caps[cam_id] = cap
        exposure_locked[cam_id] = False
        zoom_levels[cam_id] = 1.0
        rotation_angles[cam_id] = 0  # Default: no rotation
        white_balance_locked[cam_id] = False
        
        print(f"✅ Initialized camera {cam_id}")

    print(f"✅ All cameras ready: {list(caps.keys())}")


def lock_exposure(cam_id=0, stabilization_frames=30):
    """
    Let auto-exposure settle, then switch to manual at the value it chose.
    Uses v4l2-ctl so the value is the one the camera really holds.
    """
    cap = caps.get(cam_id)
    if cap is None:
        print(f"⚠️ Camera {cam_id} not initialized")
        return False

    camera_controls.set_control(cam_id, "auto_exposure", 3)   # aperture priority
    print(f"📸 Stabilizing exposure for camera {cam_id}...")
    for _ in range(stabilization_frames):
        ret, _frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame during stabilization")
            return False
        time.sleep(0.033)

    value = camera_controls.get_control(cam_id, "exposure_time_absolute")
    if value is None:
        print("⚠️ Cannot read exposure_time_absolute; exposure not locked")
        return False

    ok_mode, _, _ = camera_controls.set_control(cam_id, "auto_exposure", 1)   # manual
    ok_val, actual, msg = camera_controls.set_control(cam_id, "exposure_time_absolute", value)
    exposure_locked[cam_id] = ok_mode and ok_val
    print(f"✅ Exposure locked at {actual}" if exposure_locked[cam_id]
          else f"⚠️ Exposure lock incomplete: {msg}")
    return exposure_locked[cam_id]


def unlock_exposure(cam_id=0):
    """Return exposure to automatic (aperture priority)."""
    camera_controls.set_control(cam_id, "auto_exposure", 3)
    exposure_locked[cam_id] = False
    print(f"🔓 Exposure unlocked for camera {cam_id}")


def lock_white_balance(cam_id=0):
    """Turn automatic white balance off, keeping the current temperature."""
    ok, _, msg = camera_controls.set_control(cam_id, "white_balance_automatic", 0)
    white_balance_locked[cam_id] = ok
    print(f"🎨 White balance locked for camera {cam_id}" if ok
          else f"⚠️ White balance lock failed: {msg}")
    return ok


def unlock_white_balance(cam_id=0):
    """Turn automatic white balance back on."""
    camera_controls.set_control(cam_id, "white_balance_automatic", 1)
    white_balance_locked[cam_id] = False
    print(f"🎨 White balance unlocked for camera {cam_id}")


def set_white_balance(cam_id=0, temperature=4000):
    """Set a fixed colour temperature (K). Returns the temperature the camera reports."""
    camera_controls.set_control(cam_id, "white_balance_automatic", 0)
    ok, actual, msg = camera_controls.set_control(cam_id, "white_balance_temperature", temperature)
    print(f"🎨 Camera {cam_id} white balance: {actual}K" + (f" ({msg})" if msg else ""))
    return actual


def is_exposure_locked(cam_id=0):
    """Check if exposure is locked for a camera."""
    return exposure_locked.get(cam_id, False)


def set_zoom(cam_id=0, zoom_level=1.0):
    """
    Set digital zoom (software crop). Changing zoom changes the image geometry,
    so any calibration taken at the old zoom is invalidated.
    """
    min_zoom = config_store.get("min_zoom", 1.0)
    max_zoom = config_store.get("max_zoom", 4.0)
    zoom_level = max(min_zoom, min(max_zoom, zoom_level))

    old = zoom_levels.get(cam_id, 1.0)
    zoom_levels[cam_id] = zoom_level
    if abs(old - zoom_level) > 1e-6:
        calibration.invalidate_calibration(cam_id)
        print(f"🔍 Camera {cam_id} zoom set to {zoom_level:.1f}x; recalibrate the target")
    return zoom_level


def adjust_zoom(cam_id=0, delta=0.1):
    """
    Adjust zoom level by a delta amount.
    
    Args:
        cam_id: Camera ID
        delta: Amount to change zoom (positive = zoom in, negative = zoom out)
    
    Returns:
        New zoom level
    """
    current_zoom = zoom_levels.get(cam_id, 1.0)
    new_zoom = current_zoom + delta
    
    return set_zoom(cam_id, new_zoom)


def get_zoom(cam_id=0):
    """Get current zoom level for camera."""
    return zoom_levels.get(cam_id, 1.0)


def set_rotation(cam_id=0, angle=0):
    """Set image rotation. Changing it invalidates any calibration."""
    angle = angle % 360
    old = rotation_angles.get(cam_id, 0)
    rotation_angles[cam_id] = angle
    if old != angle:
        calibration.invalidate_calibration(cam_id)
        print(f"🔄 Camera {cam_id} rotation set to {angle}°; recalibrate the target")
    return angle


def get_rotation(cam_id=0):
    """Get current rotation angle for camera."""
    return rotation_angles.get(cam_id, 0)


def apply_rotation(frame, angle):
    """
    Apply rotation to frame.
    
    Args:
        frame: Input frame
        angle: Rotation angle in degrees
    
    Returns:
        Rotated frame
    """
    if angle == 0:
        return frame
    
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions if needed for 90/270 rotations
    if angle == 90 or angle == 270:
        # Swap width and height
        new_w, new_h = h, w
    else:
        new_w, new_h = w, h
    
    # Apply rotation
    rotated = cv2.warpAffine(frame, M, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    
    return rotated


def apply_digital_zoom(frame, zoom_level=1.0):
    """
    Apply digital zoom to a frame.
    
    Args:
        frame: Input frame
        zoom_level: Zoom factor
    
    Returns:
        Zoomed frame
    """
    if zoom_level <= 1.0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Calculate crop dimensions
    crop_w = int(w / zoom_level)
    crop_h = int(h / zoom_level)
    
    # Center crop
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    # Crop and scale back to original size
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed


def optimize_camera_settings(cam_id=0):
    """
    Apply the camera profile from config.json (camera_profile) through v4l2-ctl.
    Values are clamped to each control's real range. Frame rate is set through OpenCV.
    """
    cap = caps.get(cam_id)
    if cap is None:
        return
    print(f"⚙️ Applying camera profile to camera {cam_id}...")
    results = camera_controls.apply_profile(cam_id, config_store.get("camera_profile", DEFAULT_PROFILE))
    failed = [k for k, v in results.items() if not v["ok"]]
    if failed:
        print(f"⚠️ Not applied: {', '.join(failed)}")
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"✅ Camera {cam_id} profile applied")


def process_frame(frame, cam_id):
    """
    Apply all transformations to a frame (zoom, rotation).
    
    Args:
        frame: Input frame
        cam_id: Camera ID
    
    Returns:
        Processed frame
    """
    # Apply rotation first
    angle = rotation_angles.get(cam_id, 0)
    if angle != 0:
        frame = apply_rotation(frame, angle)
    
    # Then apply zoom
    zoom = zoom_levels.get(cam_id, 1.0)
    if zoom > 1.0:
        frame = apply_digital_zoom(frame, zoom)
    
    return frame


def get_frame(cam_id=0):
    """Newest frame with the rings and detection overlay (falls back to the raw frame)."""
    with _frame_lock:
        frame = _overlay.get(cam_id, _raw.get(cam_id))
        return None if frame is None else frame.copy()


def get_raw_frame(cam_id=0):
    """Newest frame after rotation and zoom, without any overlay (used for calibration)."""
    with _frame_lock:
        frame = _raw.get(cam_id)
        return None if frame is None else frame.copy()


def get_camera_info(cam_id=0):
    """Current camera settings. Controls are read from the camera through v4l2-ctl."""
    cap = caps.get(cam_id)
    if cap is None:
        return {}
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "exposure_locked": exposure_locked.get(cam_id, False),
        "white_balance_locked": white_balance_locked.get(cam_id, False),
        "zoom": zoom_levels.get(cam_id, 1.0),
        "rotation": rotation_angles.get(cam_id, 0),
    }
    for name in ["brightness", "contrast", "saturation", "sharpness", "gain",
                 "auto_exposure", "exposure_time_absolute",
                 "white_balance_temperature", "zoom_absolute"]:
        info[name] = camera_controls.get_control(cam_id, name)
    return info


def release_cameras():
    """Release all camera resources."""
    for cap in caps.values():
        cap.release()
    
    caps.clear()
    exposure_locked.clear()
    zoom_levels.clear()
    rotation_angles.clear()
    white_balance_locked.clear()
    print("📷 All cameras released")


def set_overlay(cam_id, frame):
    with _frame_lock:
        _overlay[cam_id] = frame


def _capture_loop(cam_id):
    while True:
        cap = caps.get(cam_id)
        if cap is None:
            time.sleep(0.5)
            continue
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = process_frame(frame, cam_id)        # rotation and zoom
        with _frame_lock:
            _raw[cam_id] = frame


def start_capture(camera_ids):
    """Start one capture thread per camera that has been opened."""
    for cam_id in camera_ids:
        if cam_id in caps and cam_id not in _capture_threads:
            t = threading.Thread(target=_capture_loop, args=(cam_id,), daemon=True,
                                 name=f"capture-cam{cam_id}")
            t.start()
            _capture_threads[cam_id] = t
            print(f"📷 Capture thread started for camera {cam_id}")

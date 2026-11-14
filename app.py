from flask import Flask, render_template, Response, request, jsonify, send_file
from camera_manager import (
    init_cameras, get_frame, get_raw_frame, lock_exposure, unlock_exposure, 
    is_exposure_locked, optimize_camera_settings, get_camera_info,
    set_zoom, adjust_zoom, get_zoom,
    set_rotation, get_rotation,
    lock_white_balance, unlock_white_balance, set_white_balance
)
from calibration import set_calibration, get_calibration_quality
from scoring import set_scoring_config, get_scoring_config
from shot_recorder import (
    set_shots_directory,
    override_on_disk,
    start_new_round, get_shot_image, stack_shots, unstack_shot,
    create_round_summary, get_round_statistics, list_rounds, load_round,
    delete_round, enable_recording
)
import json
import time
import cv2
import camera_controls
import camera_manager
import config_store
import workers
import threading
from heatmap import toggle_heatmap, reset_heatmap
import os
import subprocess
import re

# Load configuration
cfg = config_store.load()   # startup values only; use config_store for anything that can change

# Try to import detection module
from detection import toggle_detection, get_hits, get_all_hits, reset_hits, get_round_summary, set_override

# Try to import breakbeam module
try:
    from breakbeam import watch_breakbeam, setup_gpio
    BREAKBEAM_AVAILABLE = True
except ImportError:
    print("⚠️ Breakbeam module not available")
    BREAKBEAM_AVAILABLE = False

# Start breakbeam monitoring thread if available
if BREAKBEAM_AVAILABLE:
    threading.Thread(
        target=watch_breakbeam,
        daemon=True
    ).start()

# Capture and detection run on their own threads, so the web streams stay light
camera_manager.start_capture(cfg["cameras"])
workers.start_detection(cfg["cameras"])

app = Flask(__name__)

# Initialize cameras
init_cameras(cfg["cameras"])

# Optimize camera settings if configured
if cfg.get("optimize_camera_on_start", True):
    for cam_id in cfg["cameras"]:
        optimize_camera_settings(cam_id)

# Auto-lock exposure if configured
if cfg.get("auto_lock_exposure", True):
    for cam_id in cfg["cameras"]:
        lock_exposure(cam_id, cfg.get("exposure_stabilization_frames", 30))


def check_v4l_utils():
    """Check if v4l-utils is installed."""
    try:
        subprocess.run(["v4l2-ctl", "--version"], capture_output=True)
        print("✅ v4l-utils is available")
        return True
    except FileNotFoundError:
        print("❌ v4l-utils not found. Install: sudo apt install v4l-utils")
        return False


def get_v4l2_device(cam_id):
    """Get V4L2 device path for camera ID."""
    return f"/dev/video{cam_id}"


def get_v4l2_control_value(cam_id, control):
    """Get current value of a V4L2 control."""
    device = get_v4l2_device(cam_id)
    try:
        cmd = ["v4l2-ctl", "-d", device, "-C", control]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse: "brightness: 128"
            match = re.search(r':\s*(-?\d+)', result.stdout)
            if match:
                return int(match.group(1))
        return None
    except Exception as e:
        print(f"❌ Failed to get {control}: {e}")
        return None


def get_v4l2_control_info(cam_id, control):
    """Get min, max, default, and current value for a control."""
    device = get_v4l2_device(cam_id)
    try:
        cmd = ["v4l2-ctl", "-d", device, "-L"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if control in line:
                    # Parse: "brightness 0x00980900 (int) : min=-64 max=64 step=1 default=0 value=10"
                    match = re.search(r'min=(-?\d+)\s+max=(-?\d+).*default=(-?\d+)\s+value=(-?\d+)', line)
                    if match:
                        return {
                            "min": int(match.group(1)),
                            "max": int(match.group(2)),
                            "default": int(match.group(3)),
                            "value": int(match.group(4)),
                            "supported": True
                        }
        return {"supported": False}
    except Exception as e:
        print(f"❌ Failed to get info for {control}: {e}")
        return {"supported": False}


def set_v4l2_control(cam_id, control, value):
    """Set a V4L2 control. Clamped to the camera's range and read back (camera_controls)."""
    ok, actual, msg = camera_controls.set_control(cam_id, control, value)
    if not ok:
        print(f"❌ V4L2 control {control}={value} on camera {cam_id}: {msg}")
    return ok


def get_all_v4l2_controls(cam_id):
    """Get all available controls with their full info."""
    device = get_v4l2_device(cam_id)
    controls = {}
    try:
        cmd = ["v4l2-ctl", "-d", device, "-L"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                # Parse control lines
                match = re.match(r'\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*min=(-?\d+)\s+max=(-?\d+).*default=(-?\d+)\s+value=(-?\d+)', line)
                if match:
                    name = match.group(1)
                    controls[name] = {
                        "min": int(match.group(3)),
                        "max": int(match.group(4)),
                        "default": int(match.group(5)),
                        "value": int(match.group(6)),
                        "type": match.group(2)
                    }
        return controls
    except Exception as e:
        print(f"❌ Failed to list controls: {e}")
        return {}


def get_focus_control_name(cam_id):
    """
    Detect which focus control name this camera uses.
    Different cameras use different names:
    - Logitech 925e uses 'focus_automatic_continuous'
    - Some cameras use 'focus_auto'
    
    Returns:
        str: Control name ('focus_automatic_continuous' or 'focus_auto') or None if not supported
    """
    controls = get_all_v4l2_controls(cam_id)
    
    # Check for Logitech-style control first
    if 'focus_automatic_continuous' in controls:
        return 'focus_automatic_continuous'
    elif 'focus_auto' in controls:
        return 'focus_auto'
    else:
        return None


def check_focus_controls(cam_id):
    """
    Check which focus controls are available on this camera.
    
    Returns:
        dict: Available focus controls and their info
    """
    controls = get_all_v4l2_controls(cam_id)
    focus_controls = {}
    
    for key in ['focus_auto', 'focus_automatic_continuous', 'focus_absolute']:
        if key in controls:
            focus_controls[key] = controls[key]
    
    return focus_controls


@app.route("/")
def index():
    return render_template("index.html")


def _ctrl_response(cam_id, name, value):
    """Set a control and report what the camera actually holds after clamping."""
    ok, actual, msg = camera_controls.set_control(cam_id, name, value)
    if not ok:
        print(f"⚠️ {name}={value} on camera {cam_id}: {msg}")
    return jsonify({"success": ok, "control": name, "requested": value,
                    "value": actual, "message": msg})


@app.route("/get_control/<int:cam_id>/<control_name>")
def get_control_route(cam_id, control_name):
    """Current value and the range the camera reports for one control."""
    info = camera_controls.get_range(cam_id, control_name)
    if info is None:
        return jsonify({"success": False, "error": f"{control_name} is not a control on this camera"}), 404
    info = {k: v for k, v in info.items() if k != "value"}     # cached value; read live below
    return jsonify({"success": True, "control": control_name,
                    "value": camera_controls.get_control(cam_id, control_name), **info})


@app.route("/min_mark_size")
def min_mark_size():
    """Smallest mark that will be counted, in pixels, for camera 0 (0 if not calibrated)."""
    import math
    from calibration import get_target
    from detection import PELLET_MM
    percent = min(100, max(5, float(config_store.get("min_mark_percent", 25))))
    t = get_target(request.args.get("camera", default=0, type=int))
    if not t.get("scale_mm"):
        return jsonify({"percent": percent, "calibrated": False})
    pellet_px = PELLET_MM / t["scale_mm"]
    min_px = math.sqrt(percent / 100) * pellet_px      # diameter of a disc with that fraction of the area
    return jsonify({"percent": percent, "calibrated": True,
                    "pellet_diameter_px": round(pellet_px, 1),
                    "min_diameter_px": round(min_px, 1)})


@app.route("/override_shot", methods=["POST"])
def override_shot():
    """
    Manually set a shot's score. score = null restores the calculated score.
    Without "round", the live round is changed; with "round", a saved round is.
    """
    data = request.json or {}
    try:
        cam = int(data.get("camera", 0))
        shot_number = int(data["shot_number"])
        score = data.get("score")
        if score is not None:
            score = float(score)
            if not 0 <= score <= 5.1:
                return jsonify({"success": False, "error": "score must be 0 to 5.1"}), 400
        note = str(data.get("note", ""))[:200]
        if data.get("round"):
            final = override_on_disk(cam, str(data["round"]), shot_number, score, note)
        else:
            final = set_override(cam, shot_number, score, note)
        return jsonify({"success": True, "score": final})
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/video/<int:cam>")
def video(cam):
    """Stream video from specified camera with detection overlay."""
    def gen():
        while True:
            frame = get_frame(cam)
            if frame is None:
                time.sleep(0.05)
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            time.sleep(0.1)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/raw_video/<int:cam>")
def raw_video(cam):
    """Stream raw video (no detection overlay) from specified camera."""
    def gen():
        while True:
            frame = get_raw_frame(cam)
            if frame is None:
                time.sleep(0.05)
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            time.sleep(0.1)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/toggle", methods=["POST"])
def toggle():
    """Toggle shot detection on/off."""
    return {"active": toggle_detection()}


@app.route("/hits")
def hits():
    """All hits, keyed by camera."""
    return jsonify(get_all_hits())


@app.route("/hits/<int:cam>")
def hits_for_camera(cam):
    """Hits for one camera."""
    return jsonify(get_hits(cam))


@app.route("/round_summary")
def round_summary():
    """Current round summary for a camera (?camera=0)."""
    return jsonify(get_round_summary(request.args.get("camera", default=0, type=int)))


@app.route("/spectator")
def spectator():
    """Multi-camera spectator view page."""
    cameras = cfg.get("cameras", [0])
    return render_template("spectator.html", cameras=cameras)


@app.route("/spectator_fullscreen")
def spectator_fullscreen():
    """Full-screen multi-camera spectator mode."""
    cameras = cfg.get("cameras", [0])
    return render_template("spectator_fullscreen.html", cameras=cameras)


@app.route("/reset", methods=["POST"])
def reset():
    """Clear every camera's hits and start new rounds."""
    reset_hits()
    return "OK"


@app.route("/toggle_heatmap", methods=["POST"])
def toggle_heat():
    """Toggle heatmap visualization."""
    state = toggle_heatmap()
    return {"heatmap": state}


@app.route("/heatmap_status")
def heatmap_status():
    """Get heatmap status."""
    try:
        from heatmap import is_heatmap_enabled, get_heatmap
        enabled = is_heatmap_enabled()
        
        heatmap_data = get_heatmap()
        has_data = False
        if heatmap_data is not None:
            has_data = bool(heatmap_data.max() > 0)  # Convert numpy.bool_ to Python bool
        
        return jsonify({
            "enabled": bool(enabled),  # Ensure Python bool
            "has_data": bool(has_data)  # Ensure Python bool
        })
    except Exception as e:
        print(f"❌ Heatmap status error: {e}")
        return jsonify({"enabled": False, "has_data": False})


@app.route("/reset_heatmap", methods=["POST"])
def reset_heat():
    """Clear heatmap data."""
    reset_heatmap()
    return "OK"


@app.route("/set_calibration", methods=["POST"])
def set_cal():
    """Set calibration data from frontend."""
    data = request.json
    set_calibration(data)
    quality = get_calibration_quality(data.get("camera_id", 0))
    return {"status": "OK", "quality": quality}


@app.route("/calibrate")
def calibrate():
    """Calibration page."""
    cameras = cfg.get("cameras", [0])
    return render_template("calibrate.html", cameras=cameras)

@app.route("/video_cropped/<int:cam_id>")
def video_cropped(cam_id):
    """
    Stream cropped video centered on calibrated target rings.
    This provides a zoomed-in view of just the target area.
    """
    def gen():
        from calibration import get_target
        
        while True:
            frame = get_frame(cam_id)
            if frame is None:
                time.sleep(0.05)
                continue
            
            # Get calibration data
            target = get_target(cam_id)
            
            if target.get("center") and target.get("rings_px"):
                cx, cy = target["center"]
                
                # Calculate crop area: 120% of outer ring diameter
                outer_ring_radius = int(target["rings_px"][-1])  # Last ring
                crop_radius = int(outer_ring_radius * 1.2)
                
                # Ensure crop stays within frame bounds
                h, w = frame.shape[:2]
                x1 = max(0, cx - crop_radius)
                y1 = max(0, cy - crop_radius)
                x2 = min(w, cx + crop_radius)
                y2 = min(h, cy + crop_radius)
                
                # Crop frame
                frame = frame[y1:y2, x1:x2]
            
            # Apply overlay (target rings will be drawn relative to new crop)
            frame = draw_overlay(frame, cam_id)
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ========== Camera Control Endpoints ==========

@app.route("/camera_urls")
def camera_urls():
    """Get list of camera URLs and lane names."""
    cameras = []
    lane_names = config_store.get("lane_names", {})
    
    for cam_id in cfg.get("cameras", [0]):
        cameras.append({
            "camera_id": cam_id,
            "video_url": f"/video/{cam_id}",
            "raw_url": f"/raw_video/{cam_id}",
            "lane": lane_names.get(str(cam_id), f"Lane {cam_id + 1}"),
            "display_num": cam_id + 1
        })
    
    return jsonify({"cameras": cameras})


@app.route("/camera_info/<int:cam_id>")
def camera_info(cam_id):
    """Get detailed camera information."""
    info = get_camera_info(cam_id)
    return jsonify(info)


@app.route("/set_zoom/<int:cam_id>/<float:zoom>")
def set_zoom_route(cam_id, zoom):
    """Set zoom level for camera."""
    actual_zoom = set_zoom(cam_id, zoom)
    return jsonify({"zoom": actual_zoom})


@app.route("/adjust_zoom/<int:cam_id>/<delta>")
def adjust_zoom_route(cam_id, delta):
    """Adjust zoom by delta (handles negative values like -0.5, -0.1)."""
    try:
        delta_float = float(delta)
        new_zoom = adjust_zoom(cam_id, delta_float)
        print(f"🔍 Zoom adjusted by {delta_float}: new zoom = {new_zoom}")
        return jsonify({"zoom": new_zoom, "success": True})
    except ValueError as e:
        print(f"❌ Invalid delta value: {delta}")
        return jsonify({"error": f"Invalid delta: {delta}", "success": False}), 400
    except Exception as e:
        print(f"❌ Adjust zoom error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/reset_zoom/<int:cam_id>")
def reset_zoom_route(cam_id):
    """Reset zoom to default (1.0x)."""
    try:
        actual_zoom = set_zoom(cam_id, 1.0)
        print(f"🔍 Zoom reset to 1.0x (actual: {actual_zoom})")
        return jsonify({"zoom": actual_zoom, "success": True})
    except Exception as e:
        print(f"❌ Reset zoom error: {e}")
        return jsonify({"error": str(e), "success": False}), 500
        return jsonify({"error": f"Invalid delta: {delta}", "success": False}), 400
    except Exception as e:
        print(f"❌ Adjust zoom error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/get_zoom/<int:cam_id>")
def get_zoom_route(cam_id):
    """Get current zoom level."""
    zoom = get_zoom(cam_id)
    return jsonify({"zoom": zoom})


@app.route("/set_rotation/<int:cam_id>/<int:angle>")
def set_rotation_route(cam_id, angle):
    """Set rotation angle."""
    actual_angle = set_rotation(cam_id, angle)
    return jsonify({"rotation": actual_angle})


@app.route("/get_rotation/<int:cam_id>")
def get_rotation_route(cam_id):
    """Get current rotation."""
    rotation = get_rotation(cam_id)
    return jsonify({"rotation": rotation})


@app.route("/lock_exposure/<int:cam_id>")
def lock_exposure_route(cam_id):
    """Lock camera exposure."""
    success = lock_exposure(cam_id)
    return jsonify({"success": success, "locked": True})


@app.route("/unlock_exposure/<int:cam_id>")
def unlock_exposure_route(cam_id):
    """Unlock camera exposure."""
    unlock_exposure(cam_id)
    return jsonify({"success": True, "locked": False})


@app.route("/lock_white_balance/<int:cam_id>")
def lock_white_balance_route(cam_id):
    """Lock white balance."""
    success = lock_white_balance(cam_id)
    return jsonify({"success": success})


@app.route("/unlock_white_balance/<int:cam_id>")
def unlock_white_balance_route(cam_id):
    """Unlock white balance."""
    unlock_white_balance(cam_id)
    return jsonify({"success": True})


@app.route("/set_white_balance/<int:cam_id>/<int:temp>")
def set_white_balance_route(cam_id, temp):
    """Set white balance temperature."""
    actual = set_white_balance(cam_id, temp)
    return jsonify({"temperature": actual})


@app.route("/get_white_balance_status/<int:cam_id>")
def get_white_balance_status(cam_id):
    """Get white balance lock status."""
    from camera_manager import white_balance_locked
    locked = white_balance_locked.get(cam_id, False)
    
    # Try to get current temperature
    temp = None
    try:
        info = get_v4l2_control_info(cam_id, "white_balance_temperature")
        if info.get("supported"):
            temp = info.get("value")
    except:
        pass
    
    return jsonify({
        "locked": locked,
        "temperature": temp,
        "auto": not locked
    })


@app.route("/optimize_camera/<int:cam_id>")
def optimize_camera_route(cam_id):
    """Optimize camera to actual working defaults (same as reset)."""
    try:
        print(f"\n🔧 OPTIMIZE CAMERA {cam_id}")
        
        # Set to actual working values (same as reset)
        set_v4l2_control(cam_id, "brightness", 128)
        set_v4l2_control(cam_id, "contrast", 128)
        set_v4l2_control(cam_id, "saturation", 128)
        set_v4l2_control(cam_id, "sharpness", 128)
        set_v4l2_control(cam_id, "gain", 0)
        set_v4l2_control(cam_id, "white_balance_temperature", 4000)
        set_v4l2_control(cam_id, "power_line_frequency", 2)
        set_v4l2_control(cam_id, "focus_automatic_continuous", 1)  # Auto-focus ON
        set_v4l2_control(cam_id, "auto_exposure", 3)  # Aperture Priority
        set_v4l2_control(cam_id, "exposure_dynamic_framerate", 1)
        set_v4l2_control(cam_id, "zoom_absolute", 100)
        set_zoom(cam_id, 1.0)   # digital zoom back to 1x (same as hardware zoom at 100)
        
        print("✅ Camera optimized:")
        print("   All sliders: 128 (except gain=0)")
        print("   Auto-focus: ON")
        print("   Auto-exposure: Aperture Priority\n")
        
        # Also run OpenCV optimization
        optimize_camera_settings(cam_id)
        
        return jsonify({
            "success": True,
            "message": "Camera optimized (brightness/contrast/saturation/sharpness=128, gain=0, autofocus=ON)"
        })
    except Exception as e:
        print(f"❌ Optimize error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ========== V4L2 Camera Control Endpoints ==========

@app.route("/get_camera_controls/<int:cam_id>")
def get_camera_controls(cam_id):
    """Get all camera controls with corrected defaults matching actual camera state."""
    try:
        print(f"\n{'='*60}")
        print(f"GET_CAMERA_CONTROLS for camera {cam_id}")
        print(f"{'='*60}")
        
        controls = get_all_v4l2_controls(cam_id)
        
        # ACTUAL CURRENT VALUES from user's camera (from v4l2-ctl --get-ctrl)
        # These are the values the camera is currently using and working well
        actual_current_values = {
            'brightness': 128,
            'contrast': 128,
            'saturation': 128,
            'sharpness': 128,
            'gain': 0,
            'white_balance_automatic': 1,
            'power_line_frequency': 2,
            'white_balance_temperature': 4000,
            'auto_exposure': 3,  # Aperture Priority Mode
            'exposure_time_absolute': 250,
            'exposure_dynamic_framerate': 1,
            'pan_absolute': 0,
            'tilt_absolute': 0,
            'focus_absolute': 0,
            'focus_automatic_continuous': 1,
            'zoom_absolute': 100
        }
        
        # Fix garbage defaults (firmware bugs: -8193, 57343, 8189)
        for control_name, control_info in controls.items():
            default = control_info.get('default', 0)
            min_val = control_info.get('min', 0)
            max_val = control_info.get('max', 255)
            
            # Check if default is garbage (outside valid range)
            is_garbage = (default < min_val or default > max_val)
            
            if is_garbage:
                # Use actual current value if we have one, otherwise middle of range
                if control_name in actual_current_values:
                    fixed_default = actual_current_values[control_name]
                    print(f"  🔧 {control_name}: garbage {default} → {fixed_default} (current value)")
                else:
                    fixed_default = (max_val - min_val) // 2 + min_val
                    print(f"  🔧 {control_name}: garbage {default} → {fixed_default} (middle)")
                
                control_info['default'] = fixed_default
                control_info['_was_garbage'] = True
                control_info['_original_default'] = default
            
            # Special range fixes
            if control_name == 'zoom_absolute':
                # User reported actual range is 100-215, not 100-500
                if control_info.get('max') == 500:
                    print(f"  📏 zoom_absolute: fixing max from 500 to 215 (actual camera range)")
                    control_info['max'] = 215
        
        # Add focus control detection
        focus_control_name = get_focus_control_name(cam_id)
        controls["_focus_control_name"] = focus_control_name
        
        focus_info = check_focus_controls(cam_id)
        controls["_focus_controls"] = focus_info
        
        # Log what we're returning
        if 'brightness' in controls:
            print(f"  → brightness.default = {controls['brightness'].get('default')}")
        if 'contrast' in controls:
            print(f"  → contrast.default = {controls['contrast'].get('default')}")
        if 'saturation' in controls:
            print(f"  → saturation.default = {controls['saturation'].get('default')}")
        if 'focus_automatic_continuous' in controls:
            print(f"  → focus_automatic_continuous.value = {controls['focus_automatic_continuous'].get('value')}")
        
        print(f"{'='*60}\n")
        
        return jsonify(controls)
    except Exception as e:
        print(f"❌ Get controls error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/set_brightness/<int:cam_id>/<int:value>")
def set_brightness(cam_id, value):
    """Set camera brightness."""
    return _ctrl_response(cam_id, "brightness", value)


@app.route("/set_contrast/<int:cam_id>/<int:value>")
def set_contrast(cam_id, value):
    """Set camera contrast."""
    return _ctrl_response(cam_id, "contrast", value)


@app.route("/set_saturation/<int:cam_id>/<int:value>")
def set_saturation(cam_id, value):
    """Set camera saturation."""
    return _ctrl_response(cam_id, "saturation", value)


@app.route("/set_hue/<int:cam_id>/<int:value>")
def set_hue(cam_id, value):
    """Set camera hue."""
    return _ctrl_response(cam_id, "hue", value)


@app.route("/set_gamma/<int:cam_id>/<int:value>")
def set_gamma(cam_id, value):
    """Set camera gamma."""
    return _ctrl_response(cam_id, "gamma", value)


@app.route("/set_gain/<int:cam_id>/<int:value>")
def set_gain(cam_id, value):
    """Set camera gain (ISO)."""
    return _ctrl_response(cam_id, "gain", value)


@app.route("/set_sharpness/<int:cam_id>/<int:value>")
def set_sharpness(cam_id, value):
    """Set camera sharpness."""
    return _ctrl_response(cam_id, "sharpness", value)


@app.route("/set_backlight_compensation/<int:cam_id>/<int:value>")
def set_backlight_compensation(cam_id, value):
    """Set backlight compensation (0 or 1)."""
    return _ctrl_response(cam_id, "backlight_compensation", value)


@app.route("/set_power_line_frequency/<int:cam_id>/<int:value>")
def set_power_line_frequency(cam_id, value):
    """Set power line frequency (0=Disabled, 1=50Hz, 2=60Hz)."""
    return _ctrl_response(cam_id, "power_line_frequency", value)


@app.route("/set_control/<int:cam_id>/<control_name>/<int:value>")
def set_control(cam_id, control_name, value):
    """Set any control the camera reports. Unknown names are refused."""
    if not camera_controls.supported(cam_id, control_name):
        return jsonify({"success": False, "error": f"{control_name} is not a control on this camera"}), 404
    return _ctrl_response(cam_id, control_name, value)


@app.route("/set_exposure_auto/<int:cam_id>/<int:value>")
def set_exposure_auto(cam_id, value):
    """Set exposure auto mode (1=Manual, 3=Auto)."""
    return _ctrl_response(cam_id, "auto_exposure", value)


@app.route("/set_exposure_absolute/<int:cam_id>/<int:value>")
def set_exposure_absolute(cam_id, value):
    """Set manual exposure value."""
    return _ctrl_response(cam_id, "exposure_time_absolute", value)


@app.route("/set_focus_auto/<int:cam_id>/<int:value>")
def set_focus_auto(cam_id, value):
    """Set auto focus using correct control name for this camera."""
    try:
        # Detect which control name to use
        control_name = get_focus_control_name(cam_id)
        
        if control_name is None:
            return jsonify({
                "success": False,
                "error": "Auto-focus not supported on this camera"
            }), 404
        
        # Set the control
        ok, actual, msg = camera_controls.set_control(cam_id, control_name, value)
        success = ok
        
        if success:
            print(f"✅ Auto-focus {'enabled' if value else 'disabled'} using {control_name}")
            return jsonify({
                "success": True,
                "focus_auto": bool(actual),
                "control_name": control_name
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to set {control_name}"
            }), 500
            
    except Exception as e:
        print(f"❌ Focus auto error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/set_focus/<int:cam_id>/<int:value>")
def set_focus(cam_id, value):
    """Set manual focus value."""
    return _ctrl_response(cam_id, "focus_absolute", value)


@app.route("/reset_camera_controls/<int:cam_id>")
def reset_camera_controls(cam_id):
    """Reset camera controls to actual working defaults (user's current camera values)."""
    try:
        print(f"\n{'='*60}")
        print(f"RESET CAMERA CONTROLS for camera {cam_id}")
        print(f"{'='*60}")
        
        set_zoom(cam_id, 1.0)   # keep digital zoom in step with the reset
        controls = get_all_v4l2_controls(cam_id)
        
        # ACTUAL WORKING DEFAULTS from user's camera (from v4l2-ctl --get-ctrl)
        # These are the values currently set and working well
        reset_to_values = {
            'brightness': 128,
            'contrast': 128,
            'saturation': 128,
            'sharpness': 128,
            'gain': 0,
            'white_balance_automatic': 1,
            'power_line_frequency': 2,
            'white_balance_temperature': 4000,
            'auto_exposure': 3,  # Aperture Priority Mode
            'exposure_time_absolute': 250,
            'exposure_dynamic_framerate': 1,
            'pan_absolute': 0,
            'tilt_absolute': 0,
            'focus_absolute': 0,
            'focus_automatic_continuous': 1,  # Auto-focus ON
            'zoom_absolute': 100,
            'backlight_compensation': 0
        }
        
        print("Resetting to actual working values:")
        
        reset_count = 0
        for control_name, control_info in controls.items():
            min_val = control_info.get("min", 0)
            max_val = control_info.get("max", 255)
            
            # Use reset value if we have one, otherwise skip
            if control_name in reset_to_values:
                target = reset_to_values[control_name]
                
                # Clamp to valid range
                target = max(min_val, min(max_val, target))
                
                print(f"  {control_name}: → {target}")
                
                if set_v4l2_control(cam_id, control_name, target):
                    reset_count += 1
        
        print(f"\n✅ RESET COMPLETE: {reset_count} controls")
        print(f"   brightness: 128")
        print(f"   contrast: 128")
        print(f"   saturation: 128")
        print(f"   sharpness: 128")
        print(f"   gain: 0")
        print(f"   focus_automatic_continuous: 1 (ON)")
        print(f"   auto_exposure: 3 (Aperture Priority)")
        print(f"{'='*60}\n")
        
        return jsonify({
            "success": True,
            "message": f"Reset {reset_count} controls to working defaults (all 128 except gain=0)",
            "count": reset_count
        })
    except Exception as e:
        print(f"❌ Reset error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ========== Lane Name Management ==========

@app.route("/get_lane_names")
def get_lane_names():
    """Get custom lane names."""
    lane_names = config_store.get("lane_names", {})
    return jsonify(lane_names)


@app.route("/set_lane_name", methods=["POST"])
def set_lane_name():
    """Set custom lane name for a camera."""
    try:
        data = request.json or {}
        cam_id = str(int(data.get("camera_id", 0)))
        name = str(data.get("name") or f"Lane {int(cam_id) + 1}")[:40]
        lanes = dict(config_store.get("lane_names", {}))
        lanes[cam_id] = name
        config_store.update(lane_names=lanes)
        return jsonify({"success": True, "camera_id": cam_id, "name": name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ========== Detection FPS Control ==========

@app.route("/set_detection_fps/<int:fps>")
def set_detection_fps(fps):
    """Set detection framerate (1 to 30 per second)."""
    if not 1 <= fps <= 30:
        return jsonify({"success": False, "error": "fps must be 1 to 30"}), 400
    config_store.update(detection_fps=fps)
    return jsonify({"success": True, "fps": fps})


# ========== Scoring Configuration Endpoints ==========

@app.route("/set_scoring_config", methods=["POST"])
def set_scoring():
    """Set scoring configuration."""
    data = request.json or {}
    config = set_scoring_config(
        style=data.get("scoring_style"),
        use_4_ring=data.get("use_4_ring"),
        enable_bull_hole_bonus=data.get("enable_bull_hole_bonus"),
    )
    if "shots_per_round" in data:
        n = int(data["shots_per_round"])
        if not 1 <= n <= 20:
            return jsonify({"success": False, "error": "shots_per_round must be 1 to 20"}), 400
        config_store.update(shots_per_round=n)
    return jsonify(config)


@app.route("/get_scoring_config")
def get_scoring():
    """Get current scoring configuration."""
    config = get_scoring_config()
    config["shots_per_round"] = config_store.get("shots_per_round", 6)
    return jsonify(config)


# ========== Shot Recording Endpoints ==========

@app.route("/recording_stats")
def recording_stats():
    """Get shot recording statistics."""
    stats = get_round_statistics()
    return jsonify(stats)


@app.route("/get_shot_image/<int:shot_num>")
def get_shot(shot_num):
    """Image for a shot in the current round of one camera (?camera=0)."""
    cam = request.args.get("camera", default=0, type=int)
    image, metadata = get_shot_image(shot_num, cam)
    if image is None:
        return {"error": "Shot not found"}, 404
    _, buffer = cv2.imencode(".jpg", image)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


@app.route("/list_rounds")
def list_rounds_endpoint():
    """List all recorded rounds."""
    rounds = list_rounds(request.args.get("camera", type=int))
    return jsonify({"rounds": rounds})


# ========== Settings & Pages ==========

@app.route("/settings")
def settings():
    """Settings configuration page."""
    return render_template("settings.html")


@app.route("/rounds")
def rounds():
    """Rounds viewer page."""
    return render_template("rounds.html")


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/get_config")
def get_config():
    """Get complete system configuration."""
    return jsonify(config_store.load())


@app.route("/set_recording_config", methods=["POST"])
def set_recording_config():
    """Update recording settings."""
    data = request.json or {}
    config_store.update(
        record_shots=bool(data.get("record_shots", True)),
        record_full_frame=bool(data.get("record_full_frame", False)),
        record_crop_size=int(data.get("crop_size", 200)),
    )
    return jsonify({"success": True})


@app.route("/set_shot_directory", methods=["POST"])
def set_shot_directory():
    """Set the shot recording directory. Each camera gets a folder inside it."""
    try:
        directory = str((request.json or {}).get("directory", "")).strip()
        if not directory:
            return jsonify({"success": False, "error": "directory is required"}), 400
        set_shots_directory(directory)
        return jsonify({"success": True, "directory": directory})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/set_advanced_config", methods=["POST"])
def set_advanced_config():
    """Update detection settings. Only the fields that were sent are changed."""
    try:
        data = request.json or {}
        changes = {}
        for key in ("adaptive_threshold", "color_detection"):
            if key in data:
                changes[key] = bool(data[key])
        if "min_confidence" in data:
            changes["min_confidence"] = min(1.0, max(0.0, float(data["min_confidence"])))
        if "shot_cooldown_seconds" in data:
            changes["shot_cooldown_seconds"] = min(10.0, max(0.2, float(data["shot_cooldown_seconds"])))
        if "min_mark_percent" in data:
            changes["min_mark_percent"] = min(100, max(5, int(data["min_mark_percent"])))
        if "diff_threshold" in data:
            changes["diff_threshold"] = min(255, max(1, int(data["diff_threshold"])))
        config_store.update_dict(changes)
        return jsonify({"success": True, "changed": sorted(changes)})
    except (TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/set_zoom_config", methods=["POST"])
def set_zoom_config():
    """Update zoom limits."""
    try:
        data = request.json or {}
        mn = float(data.get("min_zoom", 1.0))
        mx = float(data.get("max_zoom", 4.0))
        if not 1.0 <= mn <= mx <= 10.0:
            return jsonify({"success": False, "error": "need 1 <= min_zoom <= max_zoom <= 10"}), 400
        config_store.update(min_zoom=mn, max_zoom=mx,
                            zoom_step=float(data.get("zoom_step", 0.1)))
        return jsonify({"success": True})
    except (TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    check_v4l_utils()
    # debug mode would expose the Werkzeug debugger to the whole network
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

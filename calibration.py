import math
import cv2
import numpy as np

# REAL dimensions (mm). RINGS_MM are DIAMETERS: 1", 2", 3", 4".
BULL_DIAM = 9.525
RINGS_MM = [25.4, 50.8, 76.2, 101.6]

_state = {}   # camera id -> calibration for that camera


def _new():
    return {
        "center": None,       # camera pixels (overlay, calibration page)
        "rings_px": [],       # ring radii in camera pixels (overlay)
        "bull_hole_px": None, # bell-hole radius in camera pixels (overlay)
        "scale_mm": None,     # mm per camera pixel
        "center_t": None,     # centre in corrected space (scoring)
        "scale_t": None,      # mm per corrected pixel (scoring)
        "homography": None,
        "quality": 0.0,
    }


def _cam(cam_id):
    return _state.setdefault(int(cam_id), _new())


def set_calibration(data):
    """
    data must contain:
    - camera_id: which camera this calibration belongs to (default 0)
    - center: {x,y}   centre of the bell hole (camera pixels)
    - bull:   {x,y}   edge of the 1" ring
    - outer:  {x,y}   edge of the 4" ring
    - src_pts / dst_pts (optional): 4 points for a perspective correction
    """
    cam_id = int(data.get("camera_id", 0))
    s = _cam(cam_id)

    cx, cy = int(data["center"]["x"]), int(data["center"]["y"])
    outer_r = math.dist((cx, cy), (data["outer"]["x"], data["outer"]["y"]))
    bull_r = math.dist((cx, cy), (data["bull"]["x"], data["bull"]["y"]))
    if outer_r <= 0:
        raise ValueError("Outer ring point must be away from the centre")

    mm_per_px = (RINGS_MM[-1] / 2) / outer_r
    s["center"] = (cx, cy)
    s["scale_mm"] = mm_per_px
    s["rings_px"] = [(mm / 2) / mm_per_px for mm in RINGS_MM]
    s["bull_hole_px"] = (BULL_DIAM / 2) / mm_per_px

    # Quality from the clicks: the 1" ring should sit at 1/4 of the 4" radius
    expected_bull_r = outer_r * (RINGS_MM[0] / RINGS_MM[-1])
    quality = 1.0 - min(1.0, abs(bull_r / expected_bull_r - 1.0) * 4.0)

    s["homography"] = None
    if "src_pts" in data and "dst_pts" in data:
        src = np.array(data["src_pts"], dtype=np.float32)
        dst = np.array(data["dst_pts"], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        if H is not None:
            s["homography"] = H
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
            quality = min(quality, max(0.0, 1.0 - rms / 10.0))
    s["quality"] = quality

    if s["homography"] is None:
        s["center_t"] = (cx, cy)
        s["scale_t"] = mm_per_px
    else:
        c_t = warp_point(cx, cy, cam_id)
        o_t = warp_point(data["outer"]["x"], data["outer"]["y"], cam_id)
        d_t = math.dist(c_t, o_t)
        if d_t <= 0:
            raise ValueError("Corrected outer point coincides with centre")
        s["center_t"] = c_t
        s["scale_t"] = (RINGS_MM[-1] / 2) / d_t


def warp_point(x, y, cam_id=0):
    H = _cam(cam_id)["homography"]
    if H is None:
        return int(x), int(y)
    pt = np.array([[[x, y]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt, H)
    return int(warped[0][0][0]), int(warped[0][0][1])


def get_target(cam_id=0):
    return _cam(cam_id)


def get_calibration_quality(cam_id=0):
    """0.0 to 1.0. The calibration page shows 'good' at 0.75 or above."""
    return round(_cam(cam_id)["quality"], 3)


def invalidate_calibration(cam_id=0):
    """
    Forget one camera's calibration. Used when zoom or rotation changes, because
    the measured centre and scale no longer match the image. Detection stays off
    for that camera until it is calibrated again.
    """
    _state[int(cam_id)] = _new()

"""
Pellet detection and scoring, one independent state per camera.

Each camera has its own baseline, hit list, cooldown, and round. A single
on/off switch starts and stops all cameras together.
"""

import functools
import math
import threading
import time
from statistics import median

import cv2
import numpy as np

import config_store
from scoring import (locate_mark, computed_points, clean_score, shot_points, score_distribution,
                     format_round_score, get_scoring_config)
from heatmap import register_hit
from ai_classifier import classify_hit
from overlay import draw_overlay
from calibration import warp_point, get_target
from shot_recorder import record_shot, update_shot, save_baseline, start_new_round

PELLET_MM = 4.5

# Every public function takes this lock. Detection runs on its own thread, while web
# requests, the break-beam callback, and shot saving also touch the same state.
_lock = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        with _lock:
            return fn(*args, **kwargs)
    return inner

detect_active = False
_hits = {}              # camera id -> list of shots, in order
_last_shot = {}         # camera id -> time of the last shot
_absorb = {}            # camera id -> True once a shot needs absorbing into the baseline
_baseline = {}          # camera id -> grey reference frame
_baseline_color = {}    # camera id -> colour reference frame
_saved = {}             # camera id -> baseline saved to disk for this round
_base_dark = {}        # camera id -> mask of dark pixels in the baseline (binary mode)
_last_det = {}          # camera id -> time of the last detection pass
_latest = {}            # camera id -> last raw frame (saved with bell-only shots)
rejected = []           # classifier rejects, kept for training review


def _cameras():
    return [int(c) for c in config_store.get("cameras", [0])]


def _shots_per_round():
    return int(config_store.get("shots_per_round", 6))


def _hits_for(cam_id):
    return _hits.setdefault(int(cam_id), [])


def _clear_cam(cam_id):
    cam_id = int(cam_id)
    _hits[cam_id] = []
    _last_shot.pop(cam_id, None)
    _absorb.pop(cam_id, None)


@_locked
def toggle_detection():
    global detect_active
    detect_active = not detect_active
    for cam in _cameras():
        _clear_cam(cam)
    if detect_active:
        for cam in _cameras():
            start_new_round(cam)
    else:
        _baseline.clear()
        _baseline_color.clear()
        _saved.clear()
    return detect_active


@_locked
def reset_hits(cam_id=None):
    """Clear the current round for one camera, or for all of them, and start a new round."""
    cams = _cameras() if cam_id is None else [int(cam_id)]
    for cam in cams:
        _clear_cam(cam)
        start_new_round(cam)


@_locked
def get_hits(cam_id=0):
    return _hits_for(cam_id)


@_locked
def set_override(cam_id, shot_number, score, note=""):
    """
    Set (or clear, with score=None) a manual score for one shot on the live round.
    The calculated score is kept alongside it, and the image and log are updated.
    """
    hits = _hits_for(cam_id)
    if not 1 <= shot_number <= len(hits):
        raise ValueError("no such shot")
    shot = hits[shot_number - 1]
    if score is None:
        shot.pop("override_score", None)
        shot.pop("override_note", None)
    else:
        shot["override_score"] = clean_score(score)
        shot["override_note"] = str(note)[:200]
    _finalise(shot)
    _update_image(cam_id, shot)
    return shot["score"]


@_locked
def get_all_hits():
    return {str(c): _hits_for(c) for c in _cameras()}


def detect_grey_marks(frame, baseline_frame):
    """Detect grey pellet marks on white TiO2 paint using colour information."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_baseline = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2HSV)
    value_diff = cv2.absdiff(hsv_baseline[:, :, 2], hsv[:, :, 2])
    _, grey_mask = cv2.threshold(value_diff, config_store.get("grey_threshold", 15),
                                 255, cv2.THRESH_BINARY)
    sat_diff = cv2.absdiff(hsv[:, :, 1], hsv_baseline[:, :, 1])
    low_sat = (sat_diff < 30).astype(np.uint8) * 255
    return cv2.bitwise_and(grey_mask, grey_mask, mask=low_sat)


def _dark_level(gray):
    """Grey level that separates the dark pellet holes from the light target face."""
    fixed = config_store.get("dark_threshold")
    if fixed is not None:
        return int(fixed)
    if int(gray.max()) - int(gray.min()) < 40:       # almost no contrast: Otsu is meaningless
        return 128
    level, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return level


def _reset_reference(cam_id, frame, gray):
    _baseline[cam_id] = gray.copy()
    _baseline_color[cam_id] = frame.copy()
    _base_dark[cam_id] = gray < _dark_level(gray)


def _binary_change_mask(cam_id, gray):
    """
    Black/white detection: a new pellet is a pixel that is dark now and was not dark
    in the baseline. Brightness changes are removed first by matching the median
    brightness of the frame to the baseline, so brighter or dimmer lighting does not
    create false marks.
    """
    frame = gray.astype(np.float32)
    if config_store.get("normalise_light", True):
        med_base = float(np.median(_baseline[cam_id]))
        med_frame = float(np.median(frame))
        if med_frame > 0:
            frame = np.clip(frame * (med_base / med_frame), 0, 255)
    dark_now = frame < _dark_level(_baseline[cam_id])
    return np.where(dark_now & ~_base_dark[cam_id], 255, 0).astype(np.uint8)


def _round_full(cam_id):
    return len(_hits_for(cam_id)) >= _shots_per_round()


def _awaiting_graze(cam_id, now):
    """A bell hit just happened on this camera and a face mark from the same pellet may follow."""
    hits = _hits_for(cam_id)
    return (bool(hits) and hits[-1]["bell"] and not hits[-1]["marks"]
            and now - hits[-1]["timestamp"] <= config_store.get("bell_match_window_seconds", 1.5))


def _overlaps_previous(cam_id, x, y, scale_t):
    """True if (x, y) is closer than one pellet diameter to an earlier mark on this camera."""
    limit = PELLET_MM / scale_t
    for s in _hits_for(cam_id):
        if s.get("x") is not None and math.dist((x, y), (s["x"], s["y"])) < limit:
            return True
    return False


def _large_area(cam_id, area):
    """True if a blob is clearly bigger than the typical single hole on this camera."""
    singles = [s["marks"][0]["area"] for s in _hits_for(cam_id) if len(s["marks"]) == 1]
    return len(singles) >= 3 and area > 1.5 * median(singles)


def _finalise(shot):
    shot["auto_score"] = computed_points(shot)
    shot["score"] = shot_points(shot)
    shot["status"] = "review" if shot["reasons"] else "scored"
    first = shot["marks"][0] if shot["marks"] else None
    shot["x"] = first["x"] if first else None
    shot["y"] = first["y"] if first else None
    shot["area"] = first["area"] if first else None
    shot["ai_prob"] = first["ai_prob"] if first else None
    shot["confidence"] = first["ai_prob"] if first else 1.0


def _save_image(cam_id, shot, frame, cam_xy):
    """Save the shot image. Coordinates are camera pixels so the crop lands on the hole."""
    if frame is None:
        return
    rec = _shot_record(shot, cam_xy)
    record_shot(frame, rec, shot["shot_number"], cam_id)


def _shot_record(shot, cam_xy=(0, 0)):
    return {"score": shot["score"], "auto_score": shot["auto_score"],
            "override_score": shot.get("override_score"), "override_note": shot.get("override_note", ""),
            "x": cam_xy[0], "y": cam_xy[1],
            "bell": shot["bell"], "status": shot["status"], "confidence": shot["confidence"],
            "reasons": shot["reasons"], "marks": shot["marks"]}


def _update_image(cam_id, shot):
    """The shot changed (score or status); refresh its saved metadata and image name."""
    update_shot(cam_id, shot["shot_number"], _shot_record(shot))


def _center_cam(cam_id):
    c = get_target(cam_id)["center"]
    return c if c is not None else (0, 0)


def _record_face(cam_id, cand, frame, now, center_t, scale_t, multiple):
    """Turn a face candidate into a mark. Returns the shot it belongs to."""
    hits = _hits_for(cam_id)
    dist_mm = math.dist((cand["x"], cand["y"]), center_t) * scale_t
    mark = locate_mark(dist_mm)
    mark.update(x=round(cand["x"], 2), y=round(cand["y"], 2),
                cam_x=round(cand["cx"], 1), cam_y=round(cand["cy"], 1),
                area=round(cand["area"], 1), ai_prob=round(cand["prob"], 3))

    reasons = []
    if multiple:
        reasons.append("multiple_blobs")
    if mark["zone"] == "bell_hole":
        reasons.append("inside_bell_hole")
    if _overlaps_previous(cam_id, cand["x"], cand["y"], scale_t):
        reasons.append("overlaps_previous_hole")
    if _large_area(cam_id, cand["area"]):
        reasons.append("large_area")
    if cand.get("split"):
        reasons.append("split_blob")

    if _awaiting_graze(cam_id, now):
        shot = hits[-1]                       # grazing mark from the same pellet
        shot["marks"].append(mark)
        shot["reasons"].extend(reasons)
        _finalise(shot)
        _update_image(cam_id, shot)
    else:
        shot = {"shot_number": len(hits) + 1, "timestamp": now, "bell": False,
                "marks": [mark], "reasons": reasons}
        hits.append(shot)
        _finalise(shot)
        _save_image(cam_id, shot, frame, (mark["cam_x"], mark["cam_y"]))

    register_hit(cand["x"], cand["y"], cam_id)
    _last_shot[cam_id] = now
    _absorb[cam_id] = True
    return shot


@_locked
def register_bell_event(ts=None, cam_id=0):
    """
    Called by the break-beam watcher each time a pellet passes through the bell hole.
    A bell event within the window of a face shot is treated as the same pellet.
    """
    cam_id = int(cam_id)
    if not detect_active or _round_full(cam_id):
        return None
    ts = ts or time.time()
    hits = _hits_for(cam_id)
    window = config_store.get("bell_match_window_seconds", 1.5)

    if hits and not hits[-1]["bell"] and ts - hits[-1]["timestamp"] <= window:
        shot = hits[-1]
        shot["bell"] = True
        _finalise(shot)
        _update_image(cam_id, shot)
    else:
        shot = {"shot_number": len(hits) + 1, "timestamp": ts, "bell": True,
                "marks": [], "reasons": []}
        hits.append(shot)
        _finalise(shot)
        frame = _latest.get(cam_id)
        _save_image(cam_id, shot, frame.copy() if frame is not None else None,
                    _center_cam(cam_id))

    _last_shot[cam_id] = ts
    _absorb[cam_id] = True
    print(f"🔔 Camera {cam_id} bell hit on shot {shot['shot_number']}: {shot['score']} pts")
    return shot


SPLIT_FACTOR = 1.5     # a blob bigger than this many pellet areas may be overlapping pellets


def _make_candidate(gray, cx, cy, area, cam_id, now):
    """Classify one candidate centre. Returns None if the classifier rejects it."""
    # Clamp the crop inside the frame so marks near the edge are kept
    x0 = int(min(max(cx - 8, 0), gray.shape[1] - 16))
    y0 = int(min(max(cy - 8, 0), gray.shape[0] - 16))
    crop = gray[y0:y0 + 16, x0:x0 + 16]
    prob = classify_hit(crop)
    if prob < config_store.get("min_confidence", 0.6):
        rejected.append({"timestamp": now, "camera": cam_id, "cam_x": round(cx, 1),
                         "cam_y": round(cy, 1), "area": area, "ai_prob": round(prob, 3)})
        del rejected[:-500]
        return None
    wx, wy = warp_point(cx, cy, cam_id)
    return {"cx": cx, "cy": cy, "x": wx, "y": wy, "area": area, "prob": prob}


def _split_blob(contour, pellet_px, base_area):
    """
    Try to split one blob into overlapping pellets. Each pellet shows as a peak in the
    distance transform (the centre of a circle is its furthest point from the edge).
    Returns a list of {cx, cy, area} in image coordinates, or None if no clear split.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 4
    mask = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(mask, [contour - [x - pad, y - pad]], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    radius = pellet_px / 2
    k_max = min(4, max(2, int(round(cv2.contourArea(contour) / base_area))))
    size = max(3, int(pellet_px * 0.6) | 1)
    local_max = cv2.dilate(dist, np.ones((size, size), np.uint8))
    peaks = np.argwhere((dist == local_max) & (dist > 0.6 * radius))
    if len(peaks) < 2:
        return None
    peaks = sorted(peaks.tolist(), key=lambda p: -dist[p[0], p[1]])

    chosen = []
    for py, px in peaks:
        if all(math.hypot(px - cx, py - cy) >= 0.6 * pellet_px for cx, cy in chosen):
            chosen.append((px, py))
        if len(chosen) == k_max:
            break
    if len(chosen) < 2:
        return None

    ys, xs = np.nonzero(mask)
    centres = np.array([[cx, cy] for cx, cy in chosen], dtype=np.float32)
    d2 = (xs[:, None] - centres[None, :, 0]) ** 2 + (ys[:, None] - centres[None, :, 1]) ** 2
    owner = np.argmin(d2, axis=1)
    parts = []
    for i, (cx, cy) in enumerate(chosen):
        count = int(np.sum(owner == i))
        if count < 0.25 * base_area:            # a sliver is noise, not a pellet
            return None
        parts.append({"cx": x - pad + cx, "cy": y - pad + cy, "area": float(count)})
    return parts


@_locked
def process_frame(frame, cam_id):
    cam_id = int(cam_id)
    now = time.time()
    fps = min(30, max(1, int(config_store.get("detection_fps", 5))))
    if now - _last_det.get(cam_id, 0) < 0.8 / fps:
        return draw_overlay(frame, cam_id)
    _last_det[cam_id] = now

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _latest[cam_id] = frame.copy()

    if not detect_active or cam_id not in _baseline:
        _reset_reference(cam_id, frame, gray)
        _saved[cam_id] = False
        return draw_overlay(frame, cam_id)

    if not _saved.get(cam_id):
        save_baseline(frame.copy(), cam_id)
        _saved[cam_id] = True

    if _round_full(cam_id):
        return draw_overlay(frame, cam_id)

    # After a bell hit, keep watching for its grazing mark. Otherwise respect the
    # cooldown, then absorb the settled pellet into the baseline.
    if not _awaiting_graze(cam_id, now):
        if now - _last_shot.get(cam_id, 0) < config_store.get("shot_cooldown_seconds", 1.0):
            return draw_overlay(frame, cam_id)
        if _absorb.get(cam_id):
            _reset_reference(cam_id, frame, gray)
            _absorb[cam_id] = False
            return draw_overlay(frame, cam_id)

    # Difference threshold. adaptiveThreshold with a positive C would turn unchanged
    # regions (diff = 0) white, so the offset is negative.
    diff_gray = cv2.absdiff(gray, _baseline[cam_id])
    diff_thr = config_store.get("diff_threshold", 25)
    if config_store.get("detection_mode", "binary") == "binary":
        thresh = _binary_change_mask(cam_id, gray)
    elif config_store.get("adaptive_threshold", True):
        thresh_gray = cv2.adaptiveThreshold(diff_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, -diff_thr)
    else:
        _, thresh_gray = cv2.threshold(diff_gray, diff_thr, 255, cv2.THRESH_BINARY)

    if config_store.get("detection_mode", "binary") == "difference":
        if config_store.get("color_detection", True):
            thresh = cv2.bitwise_or(thresh_gray, detect_grey_marks(frame, _baseline_color[cam_id]))
        else:
            thresh = thresh_gray

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    t = get_target(cam_id)
    if t["center_t"] is None or not t["scale_t"]:
        return draw_overlay(frame, cam_id)
    center_t, scale_t = t["center_t"], t["scale_t"]

    # Accept blobs from min_mark_percent of a pellet's area up to 3 times it.
    pellet_px = PELLET_MM / t["scale_mm"]
    base_area = math.pi * (pellet_px / 2) ** 2
    min_percent = min(100, max(5, float(config_store.get("min_mark_percent", 25))))
    min_area, max_area = (min_percent / 100) * base_area, 3.0 * base_area
    min_conf = config_store.get("min_confidence", 0.6)

    candidates = []
    blobs = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        blobs += 1
        if area > SPLIT_FACTOR * base_area:
            parts = _split_blob(c, pellet_px, base_area)
            if parts:
                for p in parts:
                    cand = _make_candidate(gray, p["cx"], p["cy"], p["area"], cam_id, now)
                    if cand:
                        cand["split"] = True
                        candidates.append(cand)
                continue
            if area > max_area:                 # too big to be a pellet and no clear split
                rejected.append({"timestamp": now, "camera": cam_id, "reason": "too_large",
                                 "area": area})
                continue
        if area > max_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cand = _make_candidate(gray, M["m10"] / M["m00"], M["m01"] / M["m00"], area, cam_id, now)
        if cand:
            candidates.append(cand)

    if not candidates:
        return draw_overlay(frame, cam_id)

    raw = frame.copy()          # saved images are taken before any drawing
    multiple = blobs > 1
    new_shots = []
    for cand in candidates:
        if _round_full(cam_id):
            break
        new_shots.append(_record_face(cam_id, cand, raw, now, center_t, scale_t, multiple))

    for shot in new_shots:
        mark = shot["marks"][-1]
        color = (0, 255, 255) if shot["status"] == "review" else (0, 255, 0)
        cx, cy = int(mark["cam_x"]), int(mark["cam_y"])
        cv2.circle(frame, (cx, cy), 6, color, 2)
        cv2.putText(frame, str(shot["score"]), (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"🎯 Camera {cam_id} shot {shot['shot_number']}: {shot['score']} pts ({shot['status']})")

    if _round_full(cam_id) and new_shots:
        print(f"✅ Camera {cam_id} round complete:\n{format_round_score(_hits_for(cam_id))}")

    return draw_overlay(frame, cam_id)


@_locked
def get_round_summary(cam_id=0):
    hits = _hits_for(cam_id)
    stats = score_distribution(hits)
    return {
        "camera": int(cam_id),
        "hits": hits,
        "total_shots": len(hits),
        "total_score": stats["sum"],
        "max_possible": _shots_per_round() * 5,
        "average": stats["average"],
        "complete": _round_full(cam_id),
        "formatted": format_round_score(hits),
        "percentage": stats["percentage"],
        "bells": stats["bells"],
        "needs_review": [s["shot_number"] for s in hits if s["reasons"]],
        "style": get_scoring_config()["style"],
    }

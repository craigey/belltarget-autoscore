"""
Shot images, metadata, and a log, kept in one folder per camera:

    <shot_directory>/cam0/round_YYYYMMDD_HHMMSS/
        baseline.jpg                  clean target at the start of the round
        shot_01_score_4.jpg           crop around each pellet (one per shot)
        shot_01_score_4.json          full detail for that shot
        round_log.txt                 one line per image: time, round, camera,
                                      image, score, confidence, status, notes
        round_summary.jpg

shot_directory is relative to the app folder unless it is an absolute path.
"""

import glob
import json
import os
import re
import shutil
from datetime import datetime

import cv2
import numpy as np

import config_store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_NAME = "round_log.txt"
LOG_HEADER = "date_time\tround\tcamera\timage\tscore\tauto_score\tconfidence\tstatus\tnotes"
_ROUND_RE = re.compile(r"round_\d{8}_\d{6}(_\d+)?")
_CAM_RE = re.compile(r"cam\d+")

_state = {}   # camera id -> {"round_dir", "images": [(path, metadata)], "baseline_time"}


def _new_state(round_dir):
    return {"round_dir": round_dir, "images": [], "baseline_time": None}


def _base_dir():
    d = config_store.get("shot_directory", "shots")
    return d if os.path.isabs(d) else os.path.join(BASE_DIR, d)


def _cam_dir(cam_id):
    return os.path.join(_base_dir(), f"cam{int(cam_id)}")


def _st(cam_id):
    return _state.setdefault(int(cam_id), _new_state(None))


def init_shot_recorder():
    os.makedirs(_base_dir(), exist_ok=True)
    print(f"✅ Shot recorder ready: {_base_dir()}")


def start_new_round(cam_id=0):
    """Create a new round folder for one camera and make it the current round."""
    base = datetime.now().strftime("round_%Y%m%d_%H%M%S")
    path, n = os.path.join(_cam_dir(cam_id), base), 2
    while os.path.exists(path):              # two rounds in the same second
        path = os.path.join(_cam_dir(cam_id), f"{base}_{n}")
        n += 1
    os.makedirs(path)
    _state[int(cam_id)] = _new_state(path)
    print(f"📁 Camera {cam_id}: new round {path}")
    return path


def _current(cam_id):
    if _st(cam_id)["round_dir"] is None:
        start_new_round(cam_id)
    return _state[int(cam_id)]


def _log_line(cam_id, round_dir, when, image, score, auto_score, confidence, status, notes):
    stamp = datetime.fromisoformat(when).strftime("%Y-%m-%d %H:%M:%S") if when else ""
    conf = "" if confidence is None else f"{float(confidence):.3f}"
    return "\t".join([stamp, os.path.basename(round_dir), f"cam{int(cam_id)}", image,
                      "" if score is None else str(score),
                      "" if auto_score is None else str(auto_score),
                      conf, status or "", notes or ""])


def _write_log(cam_id):
    st = _state[int(cam_id)]
    if not st["round_dir"]:
        return
    lines = [LOG_HEADER]
    if st["baseline_time"]:
        lines.append(_log_line(cam_id, st["round_dir"], st["baseline_time"], "baseline.jpg",
                               None, None, None, "baseline", ""))
    for path, meta in st["images"]:
        notes = ", ".join(meta.get("reasons", []))
        if meta.get("override_score") is not None:
            notes = "; ".join(filter(None, [notes, "manual" + (f": {meta['override_note']}"
                                                              if meta.get("override_note") else "")]))
        lines.append(_log_line(cam_id, st["round_dir"], meta["timestamp"],
                               os.path.basename(path), meta["score"], meta.get("auto_score"),
                               meta.get("confidence"), meta.get("status"), notes))
    with open(os.path.join(st["round_dir"], LOG_NAME), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _shot_files(round_dir, shot_number):
    return glob.glob(os.path.join(round_dir, f"shot_{shot_number:02d}_score_*"))


def _crop(frame, shot_data):
    if config_store.get("record_full_frame", False):
        return frame.copy()
    size = int(config_store.get("record_crop_size", 200))
    x, y = int(shot_data["x"]), int(shot_data["y"])
    h, w = frame.shape[:2]
    x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
    x2, y2 = min(w, x + size // 2), min(h, y + size // 2)
    image = frame[y1:y2, x1:x2].copy()
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.copyMakeBorder(image, 0, size - image.shape[0], 0, size - image.shape[1],
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image


def _write_shot(cam_id, shot_number, image, shot_data):
    st = _current(cam_id)
    for old in _shot_files(st["round_dir"], shot_number):
        os.remove(old)
    base = f"shot_{shot_number:02d}_score_{shot_data['score']}"
    filepath = os.path.join(st["round_dir"], base + ".jpg")
    cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    metadata = json.loads(json.dumps(shot_data, default=float))
    metadata.setdefault("timestamp", datetime.now().isoformat())
    metadata.update(shot_number=shot_number, filepath=filepath, camera=int(cam_id))
    with open(os.path.join(st["round_dir"], base + ".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    st["images"] = [(p, m) for p, m in st["images"] if m["shot_number"] != shot_number]
    st["images"].append((filepath, metadata))
    st["images"].sort(key=lambda item: item[1]["shot_number"])
    _write_log(cam_id)
    return filepath


def record_shot(frame, shot_data, shot_number, cam_id=0):
    """Save the image and metadata for a new shot into the camera's current round."""
    if not config_store.get("record_shots", True) or frame is None:
        return None
    filepath = _write_shot(cam_id, shot_number, _crop(frame, shot_data), shot_data)
    print(f"📸 Camera {cam_id} shot {shot_number}: {os.path.basename(filepath)}")
    return filepath


def update_shot(cam_id, shot_number, shot_data):
    """
    A shot's score or status changed (for example a bell hit joined a face mark).
    Re-save its metadata and rename the image to the new score, keeping the picture.
    """
    if not config_store.get("record_shots", True):
        return None
    st = _current(cam_id)
    existing = [p for p, m in st["images"] if m["shot_number"] == shot_number]
    if not existing or not os.path.exists(existing[0]):
        return None
    image = cv2.imread(existing[0])
    if image is None:
        return None
    original = next(m for p, m in st["images"] if m["shot_number"] == shot_number)
    return _write_shot(cam_id, shot_number, image,
                       dict(shot_data, timestamp=original["timestamp"]))


def save_baseline(frame, cam_id=0):
    """Save the clean target at the start of a round."""
    st = _current(cam_id)
    cv2.imwrite(os.path.join(st["round_dir"], "baseline.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    st["baseline_time"] = datetime.now().isoformat()
    _write_log(cam_id)
    return os.path.join(st["round_dir"], "baseline.jpg")


def get_clean_target_image(cam_id=0):
    path = os.path.join(_st(cam_id)["round_dir"] or "", "baseline.jpg")
    return cv2.imread(path) if os.path.isfile(path) else None


def get_shot_image(shot_number, cam_id=0):
    images = _st(cam_id)["images"]
    if shot_number < 1 or shot_number > len(images):
        return None, None
    filepath, metadata = images[shot_number - 1]
    if not os.path.exists(filepath):
        return None, None
    return cv2.imread(filepath), metadata


def stack_shots(cam_id=0, up_to_shot=None):
    images = _st(cam_id)["images"]
    chosen = images if up_to_shot is None else images[:up_to_shot]
    if not chosen:
        return None
    composite = cv2.imread(chosen[0][0]).copy()
    for filepath, _ in chosen[1:]:
        shot = cv2.imread(filepath)
        if shot.shape != composite.shape:
            shot = cv2.resize(shot, (composite.shape[1], composite.shape[0]))
        composite = cv2.addWeighted(composite, 0.7, shot, 0.3, 0)
    return composite


def unstack_shot(shot_number, cam_id=0):
    if shot_number <= 1:
        return get_clean_target_image(cam_id)
    return stack_shots(cam_id, up_to_shot=shot_number - 1)


def create_round_summary(cam_id=0):
    images = _st(cam_id)["images"]
    if not images or _st(cam_id)["round_dir"] is None:
        return None
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    first = cv2.imread(images[0][0])
    h, w = first.shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, (filepath, metadata) in enumerate(images):
        img = cv2.imread(filepath)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        score = metadata["score"]
        cv2.putText(img, f"#{metadata['shot_number']}: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if score >= 5 else (0, 0, 255), 2)
        r, c = idx // cols, idx % cols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    path = os.path.join(_st(cam_id)["round_dir"], "round_summary.jpg")
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def _round_dirs(cam_id=None):
    """(camera folder, round folder, path) for every stored round."""
    base = _base_dir()
    out = []
    if not os.path.isdir(base):
        return out
    for cam in sorted(os.listdir(base)):
        if not _CAM_RE.fullmatch(cam):
            continue
        if cam_id is not None and cam != f"cam{int(cam_id)}":
            continue
        cam_path = os.path.join(base, cam)
        for rnd in sorted(os.listdir(cam_path)):
            path = os.path.join(cam_path, rnd)
            if _ROUND_RE.fullmatch(rnd) and os.path.isdir(path):
                out.append((cam, rnd, path))
    return out


def list_rounds(cam_id=None):
    rounds = []
    for cam, rnd, path in _round_dirs(cam_id):
        shots = sum(1 for f in os.listdir(path) if f.startswith("shot_") and f.endswith(".json"))
        rounds.append({"camera": cam, "round": rnd, "shots": shots})
    return rounds


def _safe_round_path(cam_id, round_name):
    """Return the round path only if it is a real round folder inside the shot directory."""
    if not _CAM_RE.fullmatch(f"cam{int(cam_id)}") or not _ROUND_RE.fullmatch(round_name):
        raise ValueError("invalid round")
    path = os.path.realpath(os.path.join(_cam_dir(cam_id), round_name))
    if not path.startswith(os.path.realpath(_base_dir()) + os.sep) or not os.path.isdir(path):
        raise ValueError("round not found")
    return path


def load_round(cam_id, round_name):
    path = _safe_round_path(cam_id, round_name)
    images = []
    for name in sorted(os.listdir(path)):
        if name.startswith("shot_") and name.endswith(".json"):
            with open(os.path.join(path, name)) as f:
                meta = json.load(f)
            images.append((meta["filepath"], meta))
    st = _new_state(path)
    st["images"] = images
    baseline = os.path.join(path, "baseline.jpg")
    if os.path.exists(baseline):
        st["baseline_time"] = datetime.fromtimestamp(os.path.getmtime(baseline)).isoformat()
    _state[int(cam_id)] = st
    return images


def override_on_disk(cam_id, round_name, shot_number, score, note=""):
    """
    Set (or clear, with score=None) a manual score for a saved shot. The calculated
    score is kept as auto_score, and the image name and round log are updated.
    """
    load_round(cam_id, round_name)
    st = _state[int(cam_id)]
    match = [(p, m) for p, m in st["images"] if m["shot_number"] == shot_number]
    if not match:
        raise ValueError("no such shot")
    _, meta = match[0]
    meta = dict(meta)
    auto = meta.get("auto_score", meta["score"])
    meta["auto_score"] = auto
    if score is None:
        meta.pop("override_score", None)
        meta.pop("override_note", None)
        meta["score"] = auto
    else:
        from scoring import clean_score
        meta["override_score"] = clean_score(score)
        meta["override_note"] = str(note)[:200]
        meta["score"] = clean_score(score)
    _write_shot(cam_id, shot_number, cv2.imread(match[0][0]), meta)
    return meta["score"]


def delete_round(cam_id, round_name):
    path = _safe_round_path(cam_id, round_name)
    shutil.rmtree(path)
    if _state.get(int(cam_id), {}).get("round_dir") == path:
        _state[int(cam_id)] = _new_state(None)
    print(f"🗑️ Deleted camera {cam_id} round {round_name}")


def get_round_statistics():
    return {
        "recording_enabled": config_store.get("record_shots", True),
        "shot_directory": _base_dir(),
        "cameras": {f"cam{c}": {"round_dir": s["round_dir"], "shots_recorded": len(s["images"])}
                    for c, s in _state.items()},
        "total_rounds": len(_round_dirs()),
    }


def enable_recording(enabled=True):
    config_store.update(record_shots=bool(enabled))
    print(f"📸 Recording {'enabled' if enabled else 'disabled'}")


def set_shots_directory(directory):
    config_store.update(shot_directory=directory)
    _state.clear()
    os.makedirs(_base_dir(), exist_ok=True)
    print(f"✅ Shot directory updated: {_base_dir()}")


init_shot_recorder()

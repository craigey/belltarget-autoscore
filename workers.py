"""
Detection thread: one per camera. It reads the newest frame, runs detection, and
publishes the overlay for the web streams. The camera itself is read only by the
capture thread in camera_manager.
"""

import threading
import time

import camera_manager
import config_store
import detection

_threads = {}


def _run(cam_id):
    while True:
        fps = min(30, max(1, int(config_store.get("detection_fps", 5))))
        started = time.time()
        frame = camera_manager.get_raw_frame(cam_id)
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            camera_manager.set_overlay(cam_id, detection.process_frame(frame, cam_id))
        except Exception as e:                      # keep the thread alive; log the fault
            print(f"⚠️ Detection error on camera {cam_id}: {e}")
            time.sleep(0.5)
        time.sleep(max(0.0, 1.0 / fps - (time.time() - started)))


def start_detection(camera_ids):
    for cam_id in camera_ids:
        if cam_id not in _threads:
            t = threading.Thread(target=_run, args=(cam_id,), daemon=True,
                                 name=f"detect-cam{cam_id}")
            t.start()
            _threads[cam_id] = t
            print(f"🧠 Detection thread started for camera {cam_id}")

import cv2
from calibration import get_target
from heatmap import get_heatmap_normalized, is_heatmap_enabled


def draw_overlay(frame, cam_id=0):
    """Draw the target rings, bell hole, and (optionally) the hit heatmap."""
    t = get_target(cam_id)
    if t.get("center") is None:
        return frame

    cx, cy = t["center"]

    if t.get("bull_hole_px"):
        cv2.circle(frame, (cx, cy), int(round(t["bull_hole_px"])), (255, 255, 255), 2)

    for r in t.get("rings_px", []):
        cv2.circle(frame, (cx, cy), int(r), (0, 215, 255), 2)

    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

    if is_heatmap_enabled():
        heat = get_heatmap_normalized(cam_id)
        if heat.max() > 0:
            if heat.shape[:2] != frame.shape[:2]:
                heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
            colour = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            mask = heat > 0
            frame[mask] = cv2.addWeighted(frame, 0.5, colour, 0.5, 0)[mask]
    return frame

import numpy as np

HEIGHT, WIDTH = 720, 1280
_heat = {}              # camera id -> hit-count array
_heatmap_enabled = False


def toggle_heatmap():
    global _heatmap_enabled
    _heatmap_enabled = not _heatmap_enabled
    return _heatmap_enabled


def is_heatmap_enabled():
    return _heatmap_enabled


def _array(cam_id):
    return _heat.setdefault(int(cam_id), np.zeros((HEIGHT, WIDTH), np.float32))


def register_hit(x, y, cam_id=0):
    heat = _array(cam_id)
    ix, iy = int(x), int(y)
    if 0 <= iy < heat.shape[0] and 0 <= ix < heat.shape[1]:
        heat[iy, ix] += 1


def get_heatmap(cam_id=0):
    return _array(cam_id)


def get_heatmap_normalized(cam_id=0):
    h = _array(cam_id).copy()
    if h.max() > 0:
        h = (h / h.max()) * 255
    return h.astype(np.uint8)


def reset_heatmap(cam_id=None):
    for key in ([int(cam_id)] if cam_id is not None else list(_heat)):
        if key in _heat:
            _heat[key][:] = 0

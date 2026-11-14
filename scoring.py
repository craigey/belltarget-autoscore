"""
Scoring for bell-target shooting.

Config key "scoring_style":

"bell"  Manual sheet method. Each shot starts at 4 points.
        First ring keeps 4, next ring 3, last ring 2, optional 1-point ring 1,
        miss 0. A bell adds 1, so a clean bell = 5.
        Round = 24 + bells - face deductions.

"narpa" Ring values 4, 3, 2, 1 (1 optional). A bell scores 5.
        With enable_bull_hole_bonus on, a bell with no mark on the edge
        of the hole scores 5.1.

Ring diameters come from calibration.RINGS_MM (1", 2", 3", 4").
use_4_ring_scoring controls whether the optional 1-point ring counts.
"""

import math

import config_store
from calibration import get_target, RINGS_MM, BULL_DIAM

BASE_POINTS = 4                 # first-ring value; start of each shot in bell style
RING_POINTS = [4, 3, 2, 1]      # first ring .. optional last ring
NARPA_BELL = 5
NARPA_BELL_BONUS = 5.1
PELLET_MM = config_store.get("pellet_mm", 4.5)
BELL_MM = BULL_DIAM

_scoring_config = {
    "style": config_store.get("scoring_style", "bell"),
    "use_4_ring": config_store.get("use_4_ring_scoring", True),
    "enable_bull_hole_bonus": config_store.get("enable_bull_hole_bonus", False),
}


def get_scoring_config():
    return {
        "style": _scoring_config["style"],
        "use_4_ring": _scoring_config["use_4_ring"],
        "enable_bull_hole_bonus": _scoring_config["enable_bull_hole_bonus"],
        "bull_hole_score": NARPA_BELL_BONUS,
    }


def set_scoring_config(style=None, use_4_ring=None, enable_bull_hole_bonus=None):
    if style in ("bell", "narpa"):
        _scoring_config["style"] = style
    if use_4_ring is not None:
        _scoring_config["use_4_ring"] = bool(use_4_ring)
    if enable_bull_hole_bonus is not None:
        _scoring_config["enable_bull_hole_bonus"] = bool(enable_bull_hole_bonus)
    _save_to_config()
    print(f"✅ Scoring: style={_scoring_config['style']}, "
          f"4-ring={_scoring_config['use_4_ring']}, "
          f"bull bonus={_scoring_config['enable_bull_hole_bonus']}")
    return get_scoring_config()


def _save_to_config():
    config_store.update(
        scoring_style=_scoring_config["style"],
        use_4_ring_scoring=_scoring_config["use_4_ring"],
        enable_bull_hole_bonus=_scoring_config["enable_bull_hole_bonus"],
    )


def ring_radii_mm():
    """Outer radius of each active ring, smallest first."""
    count = 4 if _scoring_config["use_4_ring"] else 3
    return [d / 2 for d in RINGS_MM[:count]]


def ring_points(ring):
    return 0 if ring is None else RING_POINTS[ring]


def disc_overlap_fraction(d, r, R):
    """
    Fraction of a pellet (centre distance d from the ring centre, radius r)
    that lies inside a disc of radius R. Exact circle-circle intersection area.
    """
    area = math.pi * r * r
    if d >= r + R:
        return 0.0
    if d + r <= R:
        return 1.0
    if d + R <= r:
        return (R * R) / (r * r)
    if d == 0:
        return min(1.0, (R * R) / (r * r))
    a = r * r * math.acos((d * d + r * r - R * R) / (2 * d * r))
    b = R * R * math.acos((d * d + R * R - r * r) / (2 * d * R))
    c = 0.5 * math.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
    return (a + b - c) / area


def locate_mark(dist_mm):
    """
    Classify a face mark by its distance from the bell-hole centre (mm).

    Ring rule: a pellet scores the ring that holds the larger part of it. If it
    is split 50/50 across a ring line, it takes the higher score (the inner ring).

    zone:  "bell_hole" (inside the hole), "bell_edge" (touching the hole edge), or "face".
    ring:  index of the scoring ring, or None if the pellet is outside all rings.
    split: fraction of the pellet inside each ring (for review and training).
    """
    radii = ring_radii_mm()
    pellet_r = PELLET_MM / 2
    bell_r = BELL_MM / 2

    ring = None
    split = []
    for i, R in enumerate(radii):
        frac = disc_overlap_fraction(dist_mm, pellet_r, R)
        split.append(round(frac, 3))
        if ring is None and frac >= 0.5:
            ring = i

    if dist_mm < bell_r - pellet_r:
        zone = "bell_hole"
    elif abs(dist_mm - bell_r) <= pellet_r:
        zone = "bell_edge"
    else:
        zone = "face"

    return {"zone": zone, "ring": ring, "split": split, "dist_mm": round(dist_mm, 2)}


def computed_points(shot):
    """The score the code works out from the marks and bells, ignoring any manual change."""
    """
    Score for one shot. A shot is a dict with:
      "bell":  bool
      "marks": list of mark dicts (at most one per pellet), each with "ring"
    """
    marks = shot.get("marks") or []
    bell = bool(shot.get("bell"))
    mark = marks[0] if marks else None

    # A clean bell (no mark on the hole edge) scores 5.1 if the option is on, else 5.
    # A bell with a mark on the hole edge scores 5 in either style.
    clean_bell_points = NARPA_BELL_BONUS if _scoring_config["enable_bull_hole_bonus"] else NARPA_BELL

    if _scoring_config["style"] == "narpa":
        if bell:
            return clean_bell_points if not marks else NARPA_BELL
        return ring_points(mark["ring"]) if mark else 0

    # bell style: 4 per shot, plus 1 for a bell
    if mark is None and not bell:
        return 0                                  # miss
    if bell and not marks:
        return BASE_POINTS + (clean_bell_points - BASE_POINTS)
    points = ring_points(mark["ring"]) if mark else BASE_POINTS
    return points + (1 if bell else 0)


def clean_score(value):
    """Whole numbers are stored as integers (4, not 4.0); 5.1 stays a decimal."""
    value = float(value)
    return int(value) if value.is_integer() else value


def shot_points(shot):
    """Final score for a shot: the manual override if one is set, otherwise the computed score."""
    override = shot.get("override_score")
    return clean_score(override) if override is not None else computed_points(shot)


def score_distribution(hits):
    points = [shot_points(h) for h in hits]
    n = len(hits)
    total = sum(points)
    return {
        "total": n,
        "sum": round(total, 1),
        "average": round(total / n, 2) if n else 0.0,
        "max_possible": n * 5,
        "percentage": round(total / (n * 5) * 100, 1) if n else 0.0,
        "bells": sum(1 for h in hits if h.get("bell")),
    }


def format_round_score(hits):
    points = [shot_points(h) for h in hits]
    stats = score_distribution(hits)
    shots_per_round = config_store.get("shots_per_round", 6)
    lines = [
        f"Round ({_scoring_config['style']}): {stats['sum']}",
        f"Shots: {' + '.join(str(p) for p in points) if points else '-'}",
        f"Bells: {stats['bells']}",
    ]
    review = sum(1 for h in hits if h.get("reasons"))
    if review:
        lines.append(f"Shots needing review: {review}")
    if len(hits) < shots_per_round:
        lines.append(f"Incomplete round ({len(hits)}/{shots_per_round} shots)")
    return "\n".join(lines)


def score_hit(x, y, area=None, cam_id=0):
    """
    Compatibility wrapper: score one face mark at corrected-space (x, y).
    Returns (points, confidence). New code should use locate_mark/shot_points.
    """
    t = get_target(cam_id)
    if t["center_t"] is None or not t["scale_t"]:
        return None, 0.0
    dist_mm = math.dist((x, y), t["center_t"]) * t["scale_t"]
    mark = locate_mark(dist_mm)
    return ring_points(mark["ring"]), 0.9

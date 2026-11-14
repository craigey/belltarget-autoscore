# belltarget-autoscore

Automatic scoring for air rifle bell targets

Scores bell targets (4 / 3 / 2 / 1 rings, bell hole) from one or more USB cameras
on a Raspberry Pi, with a web dashboard for live scores, calibration, and recorded rounds.

## Install (Raspberry Pi OS)

If you installed the earlier version as `air-target`, the installer removes that service.

    cd ~/belltarget-autoscore
    bash install.sh

This installs the packages in `requirements.txt` into `venv/`, and sets the app to start
at boot as the `belltarget-autoscore` service. Open `http://<pi-address>:5000` in a browser.

Run by hand instead:

    ./start.sh

Useful service commands:

    sudo systemctl status belltarget-autoscore
    journalctl -u belltarget-autoscore -f

## Cameras

- Plug the cameras in and list them with `v4l2-ctl --list-devices`. The app uses
  `/dev/video0`, `/dev/video2`, and so on, as listed in `config.json` under `cameras`.
- Each camera is calibrated separately on the Calibrate page: click the centre of the bell
  hole, the edge of the 1" ring, and the edge of the 4" ring.
- Zoom and rotation are part of the calibration. Changing either clears that camera's
  calibration, so recalibrate afterwards.
- Lock exposure and white balance once the lighting is set up, so the reference frame
  and the shot frames match.

## Files

Scores and images are stored per camera:

    shots/
      cam0/
        round_20261114_101500/
          baseline.jpg          clean target at the start of the round
          shot_01_score_4.jpg   crop around each pellet (one per shot, up to 6)
          shot_01_score_4.json  score, position, and detection details
          round_log.txt         one line per image: date/time, round, camera, image,
                                score, confidence, status, and notes
          round_summary.jpg
      cam1/
        ...

`shot_directory` in `config.json` changes where this folder is (relative to the app folder,
or an absolute path).

## Settings (`config.json`)

| Key | Meaning |
| --- | --- |
| `cameras` | Camera numbers to use |
| `bell_sensors` | Break-beam sensor per camera, as `"camera": GPIO pin` (BCM), e.g. `{"0": 17, "1": 27}` |
| `lane_names` | Display names for each camera |
| `scoring_style` | `bell` (manual method: 24 + bells - deductions) or `narpa` |
| `use_4_ring_scoring` | `true` to count the optional 1-point ring |
| `enable_bull_hole_bonus` | A clean bell (no mark on the hole edge) scores 5.1 instead of 5, in either scoring style. Leave `false` if your club scores it as 5 |
| `shots_per_round` | Shots per round (6 by default) |
| `detection_fps` | Detection passes per second (1 to 30) |
| `shot_cooldown_seconds` | Wait after a shot before the next is looked for |
| `bell_match_window_seconds` | Time within which a bell hit and a face mark count as one pellet |
| `min_confidence` | Classifier threshold for accepting a mark |
| `min_mark_percent` | Smallest mark counted, as a percentage of a pellet's area (5 to 100, default 25). Lower is more sensitive; higher cuts false positives. The settings page shows the size in pixels |
| `diff_threshold` | Brightness change needed to count as a mark |
| `camera_profile` | Control values applied at start-up, clamped to each camera's range |
| `enable_breakbeam` | `true` to use the break-beam bell sensors (Raspberry Pi only) |
| `detection_mode` | `binary` (black/white: dark pixels that are new since the baseline) or `difference` |
| `normalise_light` | Match each frame's brightness to the baseline before detection |
| `dark_threshold` | Grey level for a pellet hole. Leave out to pick it from each round's baseline |

## Changing a score by hand

Click a shot on the main page to change its score. Leave the box blank to go back to the
calculated score, and add an optional reason. A changed shot shows a `*`. Shots flagged for
review have a dashed border, and hovering shows the reason.

The calculated score is kept alongside the manual one, in the shot's `.json` file and in
`round_log.txt` (`score` and `auto_score` columns, with the reason in `notes`). Totals use
the manual score. The same change can be made on a saved round through `POST /override_shot`
with a `round` name.

## Overlapping pellets

A blob much bigger than one pellet is checked for overlapping pellets. Each pellet shows
as a peak in the blob's shape, so the blob is split into one shot per peak. Each half is
flagged `split_blob` for review, so check the saved image if it matters. A blob that can't
be split clearly is kept as one mark and flagged `large_area`.

## Scoring rule for marks on a ring line

A pellet scores the ring that holds the larger part of it. If it is split 50/50 across a
ring line, it takes the higher score.

Ring sizes are set in `calibration.py` (`RINGS_MM`) and the bell hole in `BULL_DIAM`.

## Break-beam bell sensors (optional)

Each camera can have its own sensor. Connect each receiver to a GPIO pin (BCM numbering), with
the pin pulled up and the beam breaking the line to ground. List them in `bell_sensors`, set
`enable_breakbeam` to `true`, and restart the service. Each pellet through a bell hole is one
falling edge, counted for that camera.

## How it runs

Each camera has a capture thread (reads the newest frame) and a detection thread (scores
at `detection_fps`). The web streams show the latest result and never touch the camera
directly. Camera controls report the value the camera actually holds after clamping.

## Troubleshooting

- **No camera image:** check `v4l2-ctl --list-devices`, and that the service user is in
  the `video` group (`sudo usermod -aG video <user>`).
- **Camera controls don't change:** list the camera's controls with
  `v4l2-ctl -d /dev/video0 -L`. The app writes the values it is given, clamped to the range
  the camera reports, and logs any control the camera rejects.
- **Scores are wrong after a zoom or rotation change:** recalibrate that camera.
- **A pellet is counted twice:** increase `shot_cooldown_seconds`.
- **A pellet is missed:** check the review flags on the round summary, and the saved image
  for that shot.

"""
Single access point for config.json.

Every read re-reads the file and every write is a read-modify-write under a lock,
so one part of the app cannot overwrite settings another part has just saved.
"""

import json
import os
import threading

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
_lock = threading.RLock()


def load():
    with _lock:
        with open(_PATH, encoding="utf-8") as f:
            return json.load(f)


def get(key, default=None):
    return load().get(key, default)


def update(**changes):
    return update_dict(changes)


def update_dict(changes):
    with _lock:
        data = load()
        data.update(changes)
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, _PATH)
        return data

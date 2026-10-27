"""Structured playback events; importing this module does not load Gym or Torch."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path


def plain_value(value):
    """Copy tensor-like values to JSON, keeping nonfinite diagnostics readable."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _environment_value(value, index):
    value = plain_value(value)
    if isinstance(value, list):
        return value[index] if index < len(value) else None
    return value


def reset_diagnostics(infos, environment_index, fallback_command=None):
    """Use pre-reset measurements, never a teleported post-reset pose."""
    command = _environment_value(infos.get("pre_reset_commands"), environment_index)
    errors = _environment_value(infos.get("loop_error"), environment_index)
    candidates = errors if isinstance(errors, list) else [errors]
    finite_errors = [value for value in candidates if isinstance(value, (int, float))]
    flags = {
        name: bool(_environment_value(value, environment_index))
        for name, value in infos.get("termination_reasons", {}).items()
    }
    return {
        "environment_index": environment_index,
        "active_command": command if command is not None else plain_value(fallback_command),
        "pitch_rad": _environment_value(infos.get("pre_reset_pitch"), environment_index),
        "roll_rad": _environment_value(infos.get("pre_reset_roll"), environment_index),
        "anchor_error_m": errors,
        "anchor_error_max_m": max(finite_errors) if finite_errors else None,
        "termination_reasons": flags,
    }


class PlaybackEventLog:
    """Print events and optionally append a replayable JSONL event stream."""

    def __init__(self, path=None):
        self.stream = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = path.open("a", buffering=1)

    def emit(self, event, **fields):
        record = plain_value(dict(
            event=event,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            **fields,
        ))
        encoded = json.dumps(record, allow_nan=False, sort_keys=True)
        print("playback_event " + encoded, flush=True)
        if self.stream is not None:
            self.stream.write(encoded + "\n")

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

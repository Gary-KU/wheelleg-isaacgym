"""Viewer keyboard commands, kept separate from camera navigation and physics."""

import math

from .config import MIN_LEG_LENGTH, MAX_LEG_LENGTH


class KeyboardDrive:
    """Hold arrows to command velocity; release to stop that component."""

    KEYS = {
        "drive_forward": "KEY_UP",
        "drive_backward": "KEY_DOWN",
        "drive_left": "KEY_LEFT",
        "drive_right": "KEY_RIGHT",
        "drive_stop": "KEY_SPACE",
        "drive_reset": "KEY_R",
        "leg_shorten": "KEY_Q",
        "leg_extend": "KEY_E",
    }
    DIRECTIONS = frozenset(("drive_forward", "drive_backward", "drive_left", "drive_right"))
    LEG_KEYS = frozenset(("leg_shorten", "leg_extend"))

    def __init__(self, speed=0.3, turn_speed=0.8, leg_length=None,
                 leg_speed=0.02, leg_range=(MIN_LEG_LENGTH, MAX_LEG_LENGTH)):
        if not all(math.isfinite(x) and x > 0 for x in (speed, turn_speed)):
            raise ValueError("Keyboard speed and turn speed must be positive and finite")
        self.speed = float(speed)
        self.turn_speed = float(turn_speed)
        if (len(leg_range) != 2 or not all(math.isfinite(x) for x in leg_range)
                or not MIN_LEG_LENGTH <= leg_range[0] < leg_range[1] <= MAX_LEG_LENGTH):
            raise ValueError("Keyboard leg range must be ordered and within [{},{}] m".format(MIN_LEG_LENGTH, MAX_LEG_LENGTH))
        if not math.isfinite(leg_speed) or leg_speed <= 0:
            raise ValueError("Keyboard leg speed must be positive and finite")
        self.leg_range = tuple(float(x) for x in leg_range)
        self.leg_speed = float(leg_speed)
        self.leg_length = None
        self.pressed = set()
        self.blocked = set()
        if leg_length is not None:
            self.reset_leg_length(leg_length)

    def bind(self, gym, viewer, gymapi):
        for action, key in self.KEYS.items():
            gym.subscribe_viewer_keyboard_event(viewer, getattr(gymapi, key), action)

    def process_events(self, events):
        """Consume Gym ActionEvents and return whether reset was requested."""
        reset = False
        for event in events:
            name, down = event.action, event.value > 0
            if name == "drive_stop" and down:
                self.blocked.update(self.pressed)
                self.pressed.clear()
            elif name == "drive_reset" and down:
                reset = True
            elif name in self.DIRECTIONS | self.LEG_KEYS:
                if not down:
                    self.pressed.discard(name)
                    self.blocked.discard(name)
                elif name not in self.blocked:
                    self.pressed.add(name)
        return reset

    def reset_leg_length(self, length):
        """Restore the configured target; held height keys need a fresh press."""
        if not math.isfinite(length) or not self.leg_range[0] <= length <= self.leg_range[1]:
            raise ValueError("Initial keyboard leg length must be within the keyboard leg range")
        self.leg_length = float(length)
        held = self.pressed & self.LEG_KEYS
        self.blocked.update(held)
        self.pressed.difference_update(held)

    def advance_leg_length(self, dt):
        """Change the target once per simulation control period, never per event."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Keyboard control dt must be positive and finite")
        if self.leg_length is None:
            raise ValueError("Configure leg_length before advancing keyboard height")
        direction = int("leg_extend" in self.pressed)-int("leg_shorten" in self.pressed)
        self.leg_length = min(self.leg_range[1], max(self.leg_range[0],
                             self.leg_length+direction*self.leg_speed*dt))
        return self.leg_length

    def command(self, leg_length=0.08):
        forward = int("drive_forward" in self.pressed) - int("drive_backward" in self.pressed)
        turn = int("drive_left" in self.pressed) - int("drive_right" in self.pressed)
        length = leg_length if self.leg_length is None else self.leg_length
        return [self.speed * forward, 0.0, self.turn_speed * turn, length]

"""Hold-to-drive keyboard and GLFW viewer for MuJoCo.

Polling GLFW retains release/focus information, so releasing keys stops the
request. Sensor/controller code stays in environment.py.
"""

import math
import time
import numpy as np
import glfw
import mujoco


class KeyboardCommands:
    def __init__(
        self,
        speed=1.0,
        turn_speed=1.0,
        leg_range=(0.08, 0.10),
        leg_rate=0.01,
        initial_height=None,
    ):
        self.speed = speed
        self.turn_speed = turn_speed
        self.leg_range = leg_range
        self.leg_rate = leg_rate
        self.initial_height = (
            leg_range[0] if initial_height is None else float(initial_height)
        )
        if not leg_range[0] <= self.initial_height <= leg_range[1]:
            raise ValueError("initial_height must be within leg_range")
        self.height = self.initial_height
        self.reset_down = False

    def update(self, pressed, dt, focused=True):
        """Pure command mapping; R is edge-triggered, velocity keys require hold."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        keys = set(pressed) if focused else set()
        reset = "R" in keys and not self.reset_down
        self.reset_down = "R" in keys
        if reset:
            self.height = self.initial_height
        if reset or "SPACE" in keys:
            return [0.0, 0.0, 0.0, self.height], reset
        vx = self.speed * (int("UP" in keys) - int("DOWN" in keys))
        yaw = self.turn_speed * (int("LEFT" in keys) - int("RIGHT" in keys))
        self.height = float(
            np.clip(
                self.height
                + self.leg_rate * dt * (int("E" in keys) - int("Q" in keys)),
                *self.leg_range
            )
        )
        return [vx, 0.0, yaw, self.height], False


class KeyboardViewer:
    """Small main-thread viewer: release events are available via GLFW polling."""

    KEYS = {
        "UP": glfw.KEY_UP,
        "DOWN": glfw.KEY_DOWN,
        "LEFT": glfw.KEY_LEFT,
        "RIGHT": glfw.KEY_RIGHT,
        "Q": glfw.KEY_Q,
        "E": glfw.KEY_E,
        "SPACE": glfw.KEY_SPACE,
        "R": glfw.KEY_R,
    }

    def __init__(self, model, data):
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed; check DISPLAY")
        self.window = glfw.create_window(
            1280,
            720,
            "WheelLeg MuJoCo | arrows: drive | Q/E: leg | SPACE: stop | R: reset",
            None,
            None,
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Cannot open MuJoCo window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.model = model
        self.data = data
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 0.65
        self.camera.azimuth = 135
        self.camera.elevation = -20
        self.last_render = 0.0
        self.last_cursor = (0.0, 0.0)
        glfw.set_scroll_callback(self.window, self._scroll)
        glfw.set_cursor_pos_callback(self.window, self._mouse)

    def _scroll(self, window, x, y):
        self.camera.distance = max(
            0.15, min(5.0, self.camera.distance * math.exp(-0.1 * y))
        )

    def _mouse(self, window, x, y):
        oldx, oldy = self.last_cursor
        self.last_cursor = (x, y)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.camera.azimuth -= 0.3 * (x - oldx)
            self.camera.elevation = max(
                -89, min(0, self.camera.elevation - 0.3 * (y - oldy))
            )

    def keys(self):
        glfw.poll_events()
        focus = bool(glfw.get_window_attrib(self.window, glfw.FOCUSED))
        return {
            name
            for name, key in self.KEYS.items()
            if glfw.get_key(self.window, key) == glfw.PRESS
        }, focus

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def sync(self, command=None):
        if time.perf_counter() - self.last_render < 1 / 60:
            return
        self.last_render = time.perf_counter()
        glfw.make_context_current(self.window)
        self.camera.lookat[:] = self.data.xpos[self.model.body("base_link").id]
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(viewport, self.scene, self.context)
        text = "Hold arrows: drive/turn | Q/E: leg | Space: stop | R: reset"
        if command is not None:
            text += "\nvx %.2f m/s  yaw %.2f rad/s  leg %.1f cm" % (
                command[0],
                command[2],
                command[3] * 100,
            )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            text,
            "",
            self.context,
        )
        glfw.swap_buffers(self.window)

    def close(self):
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()

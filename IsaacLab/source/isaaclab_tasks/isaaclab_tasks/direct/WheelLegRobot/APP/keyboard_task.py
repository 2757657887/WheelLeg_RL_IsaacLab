from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from pynput import keyboard


class ArrowKeyPublisher:
    """Publish keyboard commands for the play process through a shared JSON file."""

    def __init__(self, command_path: Path | None = None):
        app_dir = Path(__file__).resolve().parent
        self.command_path = command_path or (app_dir / "keyboard_commands.json")
        self.command_path.parent.mkdir(parents=True, exist_ok=True)

        self.key_states: dict[object, bool] = {
            keyboard.Key.up: False,
            keyboard.Key.down: False,
            keyboard.Key.left: False,
            keyboard.Key.right: False,
            "8": False,
            "2": False,
            keyboard.Key.home: False,
            keyboard.Key.end: False,
        }
        self.command = [0.0, 0.0, 0.0, 0.08]
        self._lock = threading.Lock()
        self._running = False
        self._publish_thread: threading.Thread | None = None
        self._listener: keyboard.Listener | None = None

    def _normalize_key(self, key: object) -> object | None:
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        if key in self.key_states:
            return key
        return None

    def on_press(self, key: object):
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            self.key_states[normalized] = True
            self._recompute_command_locked()

    def on_release(self, key: object):
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            self.key_states[normalized] = False
            self._recompute_command_locked()

    def _recompute_command_locked(self):
        forward = 0.0
        yaw = 0.0
        leg_length = self.command[3]

        if self.key_states[keyboard.Key.up]:
            forward = -1.5
        elif self.key_states[keyboard.Key.down]:
            forward = 1.5

        if self.key_states[keyboard.Key.left]:
            yaw = 1.5
        elif self.key_states[keyboard.Key.right]:
            yaw = -1.5

        if self.key_states["8"] or self.key_states[keyboard.Key.home]:
            leg_length += 0.01
        if self.key_states["2"] or self.key_states[keyboard.Key.end]:
            leg_length -= 0.01

        self.command[0] = forward
        self.command[1] = 0.0
        self.command[2] = yaw
        self.command[3] = min(0.15, max(0.06, leg_length))

    def _write_command_file(self):
        payload = {
            "timestamp": time.time(),
            "commands": self.command,
        }
        tmp_path = self.command_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as command_file:
            json.dump(payload, command_file)
        tmp_path.replace(self.command_path)

    def _publish_loop(self):
        while self._running:
            with self._lock:
                self._write_command_file()
            time.sleep(0.03)

    def reset_command(self):
        with self._lock:
            self.command = [0.0, 0.0, 0.0, 0.08]
            for key in self.key_states:
                self.key_states[key] = False
            self._write_command_file()

    def start(self):
        self._running = True
        self.reset_command()
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.start()

    def stop(self):
        self._running = False
        if self._listener is not None:
            self._listener.stop()
        self.reset_command()


def main():
    publisher = ArrowKeyPublisher()
    publisher.start()

    print("Keyboard command publisher is running.")
    print(f"Command file: {publisher.command_path}")
    print("Up/Down: forward/backward")
    print("Left/Right: yaw")
    print("8 or Home: increase leg length")
    print("2 or End: decrease leg length")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        publisher.stop()
        print("Keyboard command publisher stopped.")


if __name__ == "__main__":
    main()

"""
Brailce GPIO Keypad Input Device – Raspberry Pi Hardware
========================================================

Hardware input driver for Raspberry Pi GPIO keypad:
- Reads dedicated GPIO button inputs or key matrix.
- Binds configured dedicated language toggle pin ("language_pin").
- Supports simultaneous SPACE + BACKSPACE combo press to trigger LANGUAGE_TOGGLE.
- Converts physical presses to logical InputEvent objects.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from brailce.core.events import EventType, InputEvent, character_event, control_event
from brailce.input.base import InputDevice

logger = logging.getLogger(__name__)


class GPIOKeypadInputDevice(InputDevice):
    """
    GPIO-based dedicated character keypad for Raspberry Pi.
    """

    SOURCE_NAME = "gpio_keypad"

    def __init__(self, gpio_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            gpio_config: GPIO/keypad configuration from config.json.
        """
        super().__init__()
        self._gpio_config = gpio_config or {}
        self._language_pin = self._gpio_config.get("language_pin")
        self._last_space_time: float = 0.0
        self._last_backspace_time: float = 0.0
        self._running = False
        self._buttons: list[Any] = []
        logger.info(
            "GPIOKeypadInputDevice initialized (language_pin=%s).",
            self._language_pin,
        )

    @property
    def device_name(self) -> str:
        return "GPIO Dedicated Character Keypad (Raspberry Pi)"

    def start(self) -> None:
        """Start GPIO keypad listening."""
        self._running = True
        try:
            from gpiozero import Button
            if self._language_pin is not None:
                btn = Button(self._language_pin, bounce_time=0.05)
                btn.when_pressed = self._on_language_button_pressed
                self._buttons.append(btn)
                logger.info("Bound GPIO pin %s to LANGUAGE_TOGGLE", self._language_pin)
        except Exception as exc:
            logger.debug("gpiozero hardware init skipped or unavailable: %s", exc)

    def stop(self) -> None:
        """Stop GPIO keypad listening."""
        self._running = False
        for btn in self._buttons:
            try:
                btn.close()
            except Exception:
                pass
        self._buttons.clear()
        logger.info("GPIOKeypadInputDevice stopped.")

    def _on_language_button_pressed(self) -> None:
        """Triggered directly by physical dedicated language toggle button."""
        logger.info("GPIO: Dedicated language button pressed")
        self._emit_event(control_event(EventType.LANGUAGE_TOGGLE, self.SOURCE_NAME))

    def handle_key_action(self, action: str) -> None:
        """
        Helper for keypad scanners to handle space/backspace with combo detection.
        If space and backspace are pressed within 350ms of each other,
        LANGUAGE_TOGGLE is emitted.
        """
        now = time.time()
        if action == "space":
            if now - self._last_backspace_time < 0.35:
                self._last_backspace_time = 0.0
                self._last_space_time = 0.0
                logger.info("GPIO: SPACE + BACKSPACE combo detected -> LANGUAGE_TOGGLE")
                self._emit_event(control_event(EventType.LANGUAGE_TOGGLE, self.SOURCE_NAME))
                return
            self._last_space_time = now
            self._emit_event(control_event(EventType.SPACE_PRESSED, self.SOURCE_NAME))
        elif action == "backspace":
            if now - self._last_space_time < 0.35:
                self._last_backspace_time = 0.0
                self._last_space_time = 0.0
                logger.info("GPIO: BACKSPACE + SPACE combo detected -> LANGUAGE_TOGGLE")
                self._emit_event(control_event(EventType.LANGUAGE_TOGGLE, self.SOURCE_NAME))
                return
            self._last_backspace_time = now
            self._emit_event(control_event(EventType.BACKSPACE_PRESSED, self.SOURCE_NAME))

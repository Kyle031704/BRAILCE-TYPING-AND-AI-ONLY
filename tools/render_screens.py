#!/usr/bin/env python3
"""
Brailce LCD Screen Layout Verification Tool
===========================================
Renders all operating modes of the Brailce Grade 1 learning device
(Free Input, Math, AI Assistant, and Scanner) to PNG screenshots
at the exact 480x320 resolution of the ILI9488 TFT display.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brailce.display.ili9488_display import AppMode, ILI9488Display


def render_all_modes():
    display = ILI9488Display()
    display.initialize()

    # 1. Free Input Mode
    display.show_mode(AppMode.FREE_INPUT)
    display.show_character("A")
    display.show_braille("⠁", "1", "o.....")
    display.show_text_buffer("HELLO WORLD")
    display.save_screenshot("lcd_test_free_input.png")
    print("Saved lcd_test_free_input.png")

    # 2. Math Mode
    display.show_mode(AppMode.MATH)
    display.show_math_result(
        equation="2 + 3",
        result="5",
        explanation="Start at 2, count forward 3 steps: 3, 4, 5. So 2 plus 3 equals 5.",
        short_step="2 + 3 = 5",
        language="en",
    )
    display.save_screenshot("lcd_test_math_mode.png")
    print("Saved lcd_test_math_mode.png")

    # 3. AI Assistant Mode
    display.show_mode(AppMode.AI_ASSISTANT)
    display.show_ai_response(
        query="What is rain?",
        response="Rain is water falling in drops from clouds in the sky. Clouds collect water until they become heavy and release it as rain.",
    )
    display.save_screenshot("lcd_test_ai_mode.png")
    print("Saved lcd_test_ai_mode.png")

    # 4. Scanner Mode
    display.show_mode(AppMode.SCANNER)
    display.show_scanner_viewfinder(status="Position Braille page inside box | Press Enter to Scan")
    display.save_screenshot("lcd_test_scanner_mode.png")
    print("Saved lcd_test_scanner_mode.png")


if __name__ == "__main__":
    render_all_modes()

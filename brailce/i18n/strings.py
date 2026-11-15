"""
Brailce Unified Bilingual String Table (English & Filipino)
==========================================================

Single source of truth for all user-facing strings across LCD display and TTS speech.
Every entry defines both "en" and "fil" variants. Callers pass the returned string object
directly to both the LCD and SpeechService so what is displayed is what is spoken.
"""

from __future__ import annotations

from typing import Any, Dict

STRINGS: Dict[str, Dict[str, str]] = {
    # ---- General / App Lifecycle ----
    "APP_READY": {
        "en": "Ready — press any character key.",
        "fil": "Handa na — pumindot ng anumang titik.",
    },
    "APP_SHUTDOWN": {
        "en": "Shutting down...",
        "fil": "Pinapatay ang sistema...",
    },
    "LANG_SWITCHED_EN": {
        "en": "English language.",
        "fil": "English language.",
    },
    "LANG_SWITCHED_FIL": {
        "en": "Wikang Filipino.",
        "fil": "Wikang Filipino.",
    },
    "LANG_STATUS_EN": {
        "en": "[EN] English",
        "fil": "[EN] English",
    },
    "LANG_STATUS_FIL": {
        "en": "[FIL] Wikang Filipino",
        "fil": "[FIL] Wikang Filipino",
    },
    "VOLUME_ANNOUNCE": {
        "en": "Volume {vol} percent",
        "fil": "Lakas ng tunog {vol} porsyento",
    },
    "BUFFER_EMPTY": {
        "en": "Buffer empty",
        "fil": "Walang laman",
    },
    "CLEARED": {
        "en": "Cleared",
        "fil": "Nabura",
    },
    "NUMBER_SIGN": {
        "en": "number sign",
        "fil": "tanda ng numero",
    },
    "NUMBER_MODE_ON": {
        "en": "Number Sign: Keys A-J represent digits 1-0",
        "fil": "Tanda ng Numero: Ang A hanggang J ay bilang 1 hanggang 0",
    },
    "NUMBER_MODE_OFF": {
        "en": "Number Mode: OFF",
        "fil": "Modong Numero: Sarado",
    },

    # ---- Mode Names & Introductions (LCD + Spoken) ----
    "MODE_NAME_FREE_INPUT": {
        "en": "Free Input Mode",
        "fil": "Malayang Paglalagay",
    },
    "MODE_NAME_MATH": {
        "en": "Math Mode",
        "fil": "Matematika",
    },
    "MODE_NAME_AI": {
        "en": "AI Assistant Mode",
        "fil": "Katulong na AI",
    },
    "MODE_NAME_SCANNER": {
        "en": "Scanner Mode",
        "fil": "Pagbasa ng Braille",
    },
    "MODE_INTRO_FREE_INPUT": {
        "en": "Free Input Mode. Type letters and words.",
        "fil": "Malayang Paglalagay. Maglagay ng mga titik at salita.",
    },
    "MODE_INTRO_MATH": {
        "en": "Math Mode. Type numbers and operators.",
        "fil": "Matematika. Maglagay ng mga numero at operasyon.",
    },
    "MODE_INTRO_AI": {
        "en": "AI Assistant Mode. Type your question and press Enter.",
        "fil": "Katulong na AI. I-type ang iyong tanong, tapos pindutin ang Enter.",
    },
    "MODE_INTRO_SCANNER": {
        "en": "Scanner Mode. Hold a Braille page and press Enter to scan.",
        "fil": "Pagbasa ng Braille. Itapat ang pahina sa kamera, tapos pindutin ang Enter.",
    },

    # ---- Math Mode & Arithmetic Card Labels ----
    "MATH_READY": {
        "en": "Ready for math",
        "fil": "Handa sa matematika",
    },
    "MATH_STATUS_READY": {
        "en": "Ready for math input (0-9, +, -)",
        "fil": "Handa sa matematika (0-9, +, -)",
    },
    "MATH_CLEARED": {
        "en": "Math buffer cleared",
        "fil": "Nabura ang mga numero",
    },
    "MATH_ONLY_NUMBERS": {
        "en": "Numbers and operators only",
        "fil": "Mga numero at operasyon lamang",
    },
    "MATH_ERR_NUMBERS_ONLY": {
        "en": "Numbers only",
        "fil": "Mga numero lamang",
    },
    "FREE_INPUT_STATUS": {
        "en": "Letters: {count} | Press Space for word, Enter for sentence",
        "fil": "Mga titik: {count} | Space para sa salita, Enter para sa pangungusap",
    },
    "AI_STATUS_PROMPT": {
        "en": "Length: {count} characters | Press ENTER to ask",
        "fil": "Haba: {count} titik | Pindutin ang ENTER para magtanong",
    },
    "AI_STATUS_COMPLETE": {
        "en": "Answer complete. Press any key to type or TAB.",
        "fil": "Tapos na ang sagot. Pindutin ang anumang pindutan.",
    },
    "AI_THINKING_INDICATOR": {
        "en": "AI IS THINKING...",
        "fil": "NAG-IISIP ANG AI...",
    },
    "AI_THINKING_SUBTEXT": {
        "en": "Running offline neural model on Raspberry Pi 5...",
        "fil": "Tumatakbo ang offline neural model sa Raspberry Pi 5...",
    },
    "SCANNER_INSTRUCTION_BOX": {
        "en": "Place embossed Braille page inside camera box",
        "fil": "Ilagay ang pahina ng Braille sa loob ng kahon ng kamera",
    },
    "SCANNER_STATUS_POSITION": {
        "en": "Position Braille inside box | Press Enter to Scan",
        "fil": "Itapat ang Braille sa kahon | Pindutin ang Enter para basahin",
    },
    "SCANNER_PRESS_ENTER_TO_CAPTURE": {
        "en": "• Press ENTER to capture and translate page",
        "fil": "• Pindutin ang ENTER para kunan at basahin ang pahina",
    },
    "SCANNER_TARGET_CENTERED": {
        "en": "Target  : [ + ] Centered on Braille Page",
        "fil": "Puntirya: [ + ] Nakatapat sa Pahina ng Braille",
    },
    "SCANNER_FRAME_SAVED": {
        "en": "Camera Live Feed [640x480] | Frame saved: camera_preview.jpg",
        "fil": "Live Feed ng Kamera [640x480] | Naitabi: camera_preview.jpg",
    },
    "FREE_INPUT_CONSOLE_INSTRUCTION": {
        "en": "Type on dedicated Braille keys for letter and word voice feedback.",
        "fil": "Pumindot sa mga Braille key para marinig ang titik at salita.",
    },
    "MATH_CONSOLE_INSTRUCTION": {
        "en": "Type math problem (e.g. 5+3) and press Enter for step-by-step counting solution.",
        "fil": "Maglagay ng problema sa matematika (hal. 5+3) at pindutin ang Enter.",
    },
    "AI_CONSOLE_INSTRUCTION": {
        "en": "Type your question using Braille keys and press Enter to ask AI.",
        "fil": "I-type ang iyong tanong gamit ang Braille keys at pindutin ang Enter.",
    },
    "SCANNER_CONSOLE_INSTRUCTION": {
        "en": "Aim camera at Braille paper and press Enter to scan.",
        "fil": "Itapat ang kamera sa papel na may Braille at pindutin ang Enter.",
    },
    "MATH_CARD_TITLE": {
        "en": "MATH SOLUTION (GRADE 1)",
        "fil": "SOLUSYON SA MATEMATIKA (GRADE 1)",
    },
    "MATH_LABEL_EQUATION": {
        "en": "Equation",
        "fil": "Ekspresyon",
    },
    "MATH_LABEL_STEPS": {
        "en": "Steps",
        "fil": "Mga Hakbang",
    },
    "MATH_LABEL_EXPLANATION": {
        "en": "Counting Explanation",
        "fil": "Paliwanag sa Pagbibilang",
    },
    "MATH_EMPTY_BUFFER": {
        "en": "Math buffer is empty. Please enter a math problem, such as 5 plus 3.",
        "fil": "Walang laman ang math buffer. Maglagay ng problema sa matematika tulad ng 5 dagdag 3.",
    },
    "MATH_ERR_INVALID_CHARS": {
        "en": "I found letters or unsupported symbols. Please enter numbers with plus or minus.",
        "fil": "May mga titik o maling simbolo. Maglagay ng mga numero at operasyon lamang.",
    },
    "MATH_ERR_NO_TOKENS": {
        "en": "Please enter a math problem, such as 5 plus 3.",
        "fil": "Maglagay ng problema sa matematika, tulad ng 5 dagdag 3.",
    },
    "MATH_ERR_LEADING_OP": {
        "en": "Please enter the first number before the operator.",
        "fil": "Ilagay muna ang unang numero bago ang operasyon.",
    },
    "MATH_ERR_INCOMPLETE": {
        "en": "Incomplete math problem. Please enter a second number after the operator.",
        "fil": "Kulang ang problema sa matematika. Ilagay ang pangalawang numero.",
    },
    "MATH_ERR_MALFORMED": {
        "en": "Please enter a math problem like 5 plus 3.",
        "fil": "Hindi wasto ang ekspresyon sa matematika. Subukan ang 5 dagdag 3.",
    },
    "MATH_ERR_DIVIDE_ZERO": {
        "en": "Cannot divide by zero. We cannot split items into zero groups.",
        "fil": "Hindi maaaring hatiin sa sero. Hindi tayo makakabahagi sa sero na pangkat.",
    },
    "MATH_ERR_NEGATIVE": {
        "en": "In Grade 1 math, we cannot take away {b} from {a} because {b} is bigger than {a}.",
        "fil": "Sa Grade 1 matematika, hindi natin maaaring bawasan ang {a} ng {b} dahil mas malaki ang {b} kaysa sa {a}.",
    },
    "MATH_ERR_UNSUPPORTED_OP": {
        "en": "Unsupported math operator.",
        "fil": "Hindi suportadong operasyon sa matematika.",
    },
    "MATH_ERR_CALC": {
        "en": "Could not calculate this problem.",
        "fil": "Hindi makalkula ang problemang ito.",
    },

    # ---- AI Assistant Mode ----
    "AI_THINKING": {
        "en": "Thinking, please wait.",
        "fil": "Nag-iisip, mangyaring maghintay.",
    },
    "AI_UNAVAILABLE": {
        "en": "AI assistant is currently unavailable.",
        "fil": "Hindi magamit ang AI assistant ngayon.",
    },
    "AI_EMPTY_FALLBACK": {
        "en": "Sorry, I could not answer that question right now.",
        "fil": "Paumanhin, hindi ko masagot ang tanong na iyan sa ngayon.",
    },
    "AI_MODEL_MISSING": {
        "en": "Required AI model is not installed. Please install an Ollama model.",
        "fil": "Hindi naka-install ang AI model. Mangyaring mag-install ng modelo.",
    },
    "AI_READY": {
        "en": "AI ready — type your question.",
        "fil": "Handa na ang AI — pindutin ang iyong tanong.",
    },

    # ---- Scanner Mode & Camera ----
    "SCANNER_BRAILLE_DETECTED": {
        "en": "Braille detected. Press Enter to read.",
        "fil": "Nakitang Braille. Pindutin ang Enter upang basahin.",
    },
    "SCANNER_SCANNING": {
        "en": "Scanning Braille page.",
        "fil": "Sinusuri ang pahina ng Braille.",
    },
    "SCANNER_NO_BRAILLE": {
        "en": "No Braille detected. Adjust the camera position.",
        "fil": "Walang nakitang Braille. Ayusin ang posisyon ng kamera.",
    },
    "SCANNER_BLURRY": {
        "en": "Image is blurry. Adjust distance or focus.",
        "fil": "Malabo ang larawan. Ayusin ang layo o pokus.",
    },
    "SCANNER_MOVE_CLOSER": {
        "en": "Dots too small. Move the camera closer.",
        "fil": "Masyadong maliit ang mga tuldok. Ilapit ang kamera.",
    },
    "SCANNER_MOVE_FARTHER": {
        "en": "Dots too close. Move the camera slightly farther.",
        "fil": "Masyadong malapit. Ilayo nang kaunti ang kamera.",
    },
    "SCANNER_CAMERA_NOT_CONNECTED": {
        "en": "Camera not connected.",
        "fil": "Hindi nakakonekta ang kamera.",
    },
    "SCANNER_UNKNOWN_CELL": {
        "en": "unknown",
        "fil": "hindi malinaw",
    },
    "SCANNER_CAPITAL": {
        "en": "capital",
        "fil": "malaking titik",
    },
    "SCANNER_READY": {
        "en": "Hold a Braille page under the camera.",
        "fil": "Itapat ang pahina ng Braille sa ilalim ng kamera.",
    },
    "SCANNER_STOPPED": {
        "en": "Readout stopped.",
        "fil": "Itinigil ang pagbasa.",
    },
}


def get_string(key: str, language: str = "en", **kwargs: Any) -> str:
    """
    Fetch a localized string for 'en' or 'fil', formatting any placeholders.
    Falls back to English if the key or language is missing.
    """
    lang = "fil" if (language or "en").lower() == "fil" else "en"
    entry = STRINGS.get(key)
    if entry is None:
        return key
    template = entry.get(lang) or entry.get("en", key)
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


def get_mode_intro(mode: Any, language: str = "en") -> str:
    """Return the localized mode introduction string for AppMode."""
    val = getattr(mode, "name", str(mode)).lower()
    mapping = {
        "free_input": "MODE_INTRO_FREE_INPUT",
        "math": "MODE_INTRO_MATH",
        "ai_assistant": "MODE_INTRO_AI",
        "scanner": "MODE_INTRO_SCANNER",
    }
    key = mapping.get(val, "MODE_INTRO_FREE_INPUT")
    return get_string(key, language=language)


def get_mode_name(mode: Any, language: str = "en") -> str:
    """Return the localized short mode title for LCD headers."""
    val = getattr(mode, "name", str(mode)).lower()
    mapping = {
        "free_input": "MODE_NAME_FREE_INPUT",
        "math": "MODE_NAME_MATH",
        "ai_assistant": "MODE_NAME_AI",
        "scanner": "MODE_NAME_SCANNER",
    }
    key = mapping.get(val, "MODE_NAME_FREE_INPUT")
    return get_string(key, language=language)

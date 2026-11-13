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
        "fil": "Bolyum {vol} porsyento",
    },
    "BUFFER_EMPTY": {
        "en": "Buffer empty",
        "fil": "Walang laman",
    },
    "CLEARED": {
        "en": "Cleared",
        "fil": "Binura",
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
        "fil": "Malayang Pag-type",
    },
    "MODE_NAME_MATH": {
        "en": "Math Mode",
        "fil": "Modong Matematika",
    },
    "MODE_NAME_AI": {
        "en": "AI Assistant Mode",
        "fil": "Modong AI Assistant",
    },
    "MODE_NAME_SCANNER": {
        "en": "Scanner Mode",
        "fil": "Modong Scanner",
    },
    "MODE_INTRO_FREE_INPUT": {
        "en": "Free Input Mode. Type letters and words.",
        "fil": "Malayang Pag-type. Mag-type ng mga titik at salita.",
    },
    "MODE_INTRO_MATH": {
        "en": "Math Mode. Type numbers and operators.",
        "fil": "Modong Matematika. Mag-type ng mga numero at operasyon.",
    },
    "MODE_INTRO_AI": {
        "en": "AI Assistant Mode. Type your question and press Enter.",
        "fil": "Modong AI Assistant. Mag-type ng iyong tanong at pindutin ang Enter.",
    },
    "MODE_INTRO_SCANNER": {
        "en": "Scanner Mode. Hold a Braille page and press Enter to scan.",
        "fil": "Modong Scanner ng Braille. Itapat ang pahina at pindutin ang Enter upang mag-scan.",
    },

    # ---- Math Mode & Arithmetic Card Labels ----
    "MATH_READY": {
        "en": "Ready for math",
        "fil": "Handa para sa matematika",
    },
    "MATH_CLEARED": {
        "en": "Math buffer cleared",
        "fil": "Binura ang math buffer",
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
        "fil": "Handa na ang AI — itype ang iyong tanong.",
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
        "en": "No Braille detected. Adjust the page.",
        "fil": "Walang nakitang Braille. Ayusin ang pahina.",
    },
    "SCANNER_BLURRY": {
        "en": "Image is blurry. Adjust distance or focus.",
        "fil": "Malabo ang larawan. Ayusin ang layo o pokus.",
    },
    "SCANNER_MOVE_CLOSER": {
        "en": "Dots too small. Move the page closer.",
        "fil": "Masyadong maliit ang mga tuldok. Ilapit ang pahina.",
    },
    "SCANNER_MOVE_FARTHER": {
        "en": "Dots too close. Move the page slightly farther.",
        "fil": "Masyadong malapit. Ilayo nang kaunti ang pahina.",
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

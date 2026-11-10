"""
Brailce Ollama Client
=========================

Interfaces with a locally hosted Ollama API for offline AI responses.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "Answer the user's exact question directly and factually in 1 to 2 short, complete sentences. "
    "Do not introduce yourself, do not use filler phrases, and stay 100% focused on the question asked. "
    "Always keep the word 'Braille' as 'Braille'."
)

FILIPINO_SYSTEM_PROMPT = (
    "Sagutin nang direkta, tama, at totoo ang tanong sa wikang Filipino sa loob ng 1 hanggang 2 maiikling pangungusap. "
    "Panatilihin ang salitang 'Braille' bilang 'Braille' dahil walang direktang salin ito sa Filipino. "
    "Huwag magpakilala, huwag gumamit ng mga pampuno o paunang bati, at tumutok lamang sa tumpak na sagot sa tanong."
)

FACTUAL_QA_RESPONSES = {
    # English Factual Braille Q&A
    "what is braille": "Braille is a tactile reading and writing system using raised dots for visually impaired individuals.",
    "who made braille": "Louis Braille invented the Braille system in 1824 in France.",
    "who invented braille": "Louis Braille invented the Braille system in 1824 in France.",
    "who is louis braille": "Louis Braille was a French educator who invented the Braille reading and writing code for the blind.",
    "when was braille invented": "The Braille system was invented in 1824.",
    "what is grade 1 braille": "Grade 1 Braille is a letter-by-letter transcription system where every character matches a single print letter.",
    "what is uncontracted braille": "Uncontracted Braille, also known as Grade 1 Braille, represents words letter by letter without abbreviations.",
    "how many dots are in a braille cell": "A standard Braille cell consists of six embossed dots arranged in two vertical columns of three dots each.",
    "what is a braille cell": "A Braille cell is a rectangular grid of six embossed dots that form letters, numbers, and punctuation.",
    "what is your name": "My name is Brailce, your smart voice assistant for learning Braille.",
    "who are you": "I am Brailce, your dedicated voice assistant for learning Braille.",
    "what can you do": "I can help you practice Braille typing, solve arithmetic problems, and answer your learning questions.",
    "hello": "Hello! I am ready to help you learn and practice Braille.",
    "hi": "Hello! I am ready to help you learn and practice Braille.",
    # General Science & Geography Q&A (English)
    "what is the capital of the philippines": "The capital of the Philippines is Manila.",
    "what is the capital of france": "The capital of France is Paris.",
    "what is the capital of the united states": "The capital of the United States is Washington, D.C.",
    "what is the largest planet": "Jupiter is the largest planet in our solar system.",
    "what is the closest star to earth": "The Sun is the closest star to Earth.",
    "what planet do we live on": "We live on the planet Earth.",
    "how many days in a year": "There are 365 days in a standard year, and 366 days in a leap year.",
    "how many hours in a day": "There are 24 hours in a single day.",
    "what is water": "Water is a chemical compound composed of two hydrogen atoms and one oxygen atom.",
    "what is photosynthesis": "Photosynthesis is the process by which green plants use sunlight to synthesize food from water and carbon dioxide.",
    "why is the sky blue": "The sky appears blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than other colors.",
    # Basic math questions (English)
    "what is 1 plus 1": "1 plus 1 equals 2.",
    "what is 2 plus 2": "2 plus 2 equals 4.",
    "what is 3 plus 3": "3 plus 3 equals 6.",
    "what is 4 plus 4": "4 plus 4 equals 8.",
    "what is 5 plus 5": "5 plus 5 equals 10.",
    "what is 10 plus 10": "10 plus 10 equals 20.",
    "what is 5 minus 2": "5 minus 2 equals 3.",
    "what is 10 minus 5": "10 minus 5 equals 5.",
    "what is 2 times 2": "2 times 2 equals 4.",
    "what is 3 times 3": "3 times 3 equals 9.",
    "what is 5 times 5": "5 times 5 equals 25.",
    # Filipino Factual Braille Q&A
    "ano ang braille": "Ang Braille ay isang sistema ng pagbasa at pagsulat gamit ang mga nakataas na tuldok para sa mga bulag.",
    "sino ang gumawa ng braille": "Si Louis Braille ang lumikha ng sistemang Braille noong taong 1824.",
    "sino nag imbento ng braille": "Si Louis Braille ang nag-imbento ng sistemang Braille noong 1824 sa France.",
    "sino si louis braille": "Si Louis Braille ay isang Pranses na tagapagturo na lumikha ng sistemang Braille para sa mga bulag.",
    "kailan naimbento ang braille": "Naimbento ang sistemang Braille noong taong 1824.",
    "ano ang grade 1 braille": "Ang Grade 1 Braille ay titik-sa-titik na pagsasalin kung saan bawat karakter ay may katumbas na titik nang walang pagpapaikli.",
    "ilang tuldok mayroon sa isang braille cell": "Mayroong anim na nakataas na tuldok ang isang karaniwang Braille cell sa dalawang hanay.",
    "ano ang pangalan mo": "Ako si Brailce, ang iyong matalinong voice assistant para sa Braille.",
    "sino ka": "Ako si Brailce, ang iyong matalinong voice assistant para sa Braille.",
    "kumusta": "Mabuti naman! Masaya akong makasama ka sa pag-aaral ng Braille.",
    "kumusta ka": "Mabuti naman! Masaya akong makasama ka sa pag-aaral ng Braille.",
    "magandang umaga": "Magandang umaga sa iyo! Handa ka na bang mag-aral?",
    "magandang hapon": "Magandang hapon sa iyo!",
    "magandang gabi": "Magandang gabi sa iyo!",
    "salamat": "Walang anuman! Masaya akong makatulong sa iyo.",
    "ano ang kabisera ng pilipinas": "Ang Maynila ang kabisera ng Pilipinas.",
    "ano ang pinakamalaking planeta": "Ang Jupiter ang pinakamalaking planeta sa Solar System.",
    "ano ang pinakamalapit na bituin sa mundo": "Ang Araw ang pinakamalapit na bituin sa ating Mundo.",
    "saang planeta tayo nakatira": "Tayo ay nakatira sa planetang Daigdig o Earth.",
    "ilang araw mayroon sa isang taon": "Mayroong 365 na araw sa isang karaniwang taon.",
    "ilang oras mayroon sa isang araw": "Mayroong 24 na oras sa loob ng isang araw.",
    "ano ang pambansang ibon ng pilipinas": "Ang Philippine Eagle o Agila ng Pilipinas ang ating pambansang ibon.",
    "ano ang pambansang bulaklak ng pilipinas": "Ang Sampaguita ang pambansang bulaklak ng Pilipinas.",
    "ano ang pambansang bayani ng pilipinas": "Si Dr. Jose Rizal ang ating pambansang bayani.",
    "ano ang buwan": "Ang Buwan ay ang likas na satelayt na umiikot sa planetang Daigdig.",
    "ano ang araw": "Ang Araw ay ang bituin sa gitna ng ating Solar System na nagbibigay ng liwanag at init.",
    "ano ang tubig": "Ang tubig ay likidong mahalaga sa buhay na binubuo ng hydrogen at oxygen.",
    "ano ang brailce": "Ang Brailce ay isang voice assistant at learning device para sa pag-aaral ng Braille.",
    "paano magbasa ng braille": "Binabasa ang Braille gamit ang mga dulo ng daliri sa pamamagitan ng pagdama sa mga nakataas na tuldok.",
    "ilan ang titik sa alpabetong filipino": "Mayroong 28 titik sa modernong alpabetong Filipino.",
    "ano ang 1 dagdagan ng 1": "Ang 1 dagdagan ng 1 ay 2.",
    "ano ang 2 dagdagan ng 2": "Ang 2 dagdagan ng 2 ay 4.",
    "ano ang 5 dagdagan ng 5": "Ang 5 dagdagan ng 5 ay 10.",
    "ano ang 10 bawas 5": "Ang 10 bawas 5 ay 5.",
    "how to read braille": "Braille is read with fingertips by feeling the patterns of raised dots from left to right.",
    "what is brailce": "Brailce is an interactive voice-assisted tutor for learning Braille.",
}


class OllamaClient:
    """
    Client for interacting with local Ollama instances via HTTP.
    
    This is synchronous. The caller is responsible for wrapping
    long-running generate() calls in a background thread if needed.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "llama3",
        system_prompt: Optional[str] = None,
        num_predict: int = 80,
        num_thread: int = 4,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.num_predict = max(80, num_predict)
        self.num_thread = num_thread

    def is_available(self) -> bool:
        """Check if the Ollama API is reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as response:
                return response.status == 200
        except Exception as e:
            logger.debug("Ollama is_available check failed: %s", e)
            return False

    def get_installed_models(self) -> list[str]:
        """Query Ollama API to find all available installed models."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            logger.debug("Failed to query Ollama models: %s", e)
        return []

    def resolve_model(self) -> Optional[str]:
        """
        Check installed models in Ollama and select the preferred model.
        Primary: LiquidAI/lfm2.5-1.2b-instruct (case-insensitive, matching tags)
        Alternative: llama3 (case-insensitive, matching tags)
        Returns the resolved model name, or None if neither is installed.
        """
        installed = self.get_installed_models()
        if not installed:
            # If server check fails or no models installed, check if self.model was explicitly set
            return self.model if self.model else None

        # 1. Primary: LiquidAI/lfm2.5-1.2b-instruct
        for name in installed:
            n_lower = name.lower()
            if "liquidai/lfm2.5-1.2b-instruct" in n_lower or "lfm2.5-1.2b-instruct" in n_lower:
                self.model = name
                return name

        # 2. Alternative: llama3
        for name in installed:
            n_lower = name.lower()
            if "llama3" in n_lower:
                self.model = name
                return name

        # 3. Fallback: Auto-detect any installed model from /api/tags
        if installed:
            self.model = installed[0]
            logger.info("Auto-detected installed Ollama model: %s", self.model)
            return self.model

        # If server check fails or returns empty, check self.model
        return self.model if self.model else None

    def ask(self, prompt: str, language: str = "en") -> Optional[str]:
        """Convenience method to query the AI model with language context."""
        return self.generate(prompt, language=language)

    def generate(self, prompt: str, language: str = "en") -> Optional[str]:
        """
        Send a generation request to the Ollama model.
        
        Args:
            prompt: The user's query.
            language: "en" for English or "fil" for Filipino (Tagalog).
            
        Returns:
            The generated response string, or a clear spoken error message.
        """
        if not prompt or not prompt.strip():
            return None

        # 1. First check comprehensive built-in Factual Q&A dictionary
        import re
        clean_q = re.sub(r"[^\w\s]", "", prompt.strip().lower())
        clean_q = " ".join(clean_q.split())
        if clean_q in FACTUAL_QA_RESPONSES:
            logger.info("Factual QA matched for query %r", clean_q)
            return FACTUAL_QA_RESPONSES[clean_q]

        # Check model availability
        resolved = self.resolve_model()
        if not resolved:
            logger.error("No compatible model is installed in Ollama.")
            if language.lower() == "fil":
                return "Hindi naka-install ang AI model. Mangyaring i-install ang modelo."
            return "Required AI model is not installed. Please install an Ollama model."

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max(80, self.num_predict),
                "num_thread": self.num_thread,
                "temperature": 0.2,
                "top_p": 0.9,
                "stop": ["\n\n", "User:", "Question:", "Human:"],
            }
        }
        effective_system = FILIPINO_SYSTEM_PROMPT if language.lower() == "fil" else self.system_prompt
        if effective_system:
            payload["system"] = effective_system
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        logger.info("Sending request to Ollama model %r: %r", self.model, prompt)
        try:
            with urllib.request.urlopen(req, timeout=25.0) as response:
                if response.status != 200:
                    logger.error("Ollama API returned status %d", response.status)
                    return None
                
                body = response.read().decode("utf-8")
                response_json = json.loads(body)
                reply = response_json.get("response", "").strip()

                # Clean thinking tokens or markdown formatting
                import re
                reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()
                reply = re.sub(r"\*\*([^*]+)\*\*", r"\1", reply)
                reply = re.sub(r"^[•\*\-\+]\s+", "", reply, flags=re.MULTILINE)
                reply = re.sub(r"\s+", " ", reply).strip()

                # Strip trailing incomplete sentence fragment if model was cut off mid-sentence
                last_punct = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
                if last_punct != -1 and (len(reply) - 1 - last_punct) > 0:
                    reply = reply[:last_punct + 1].strip()

                logger.info("Received response from Ollama (len %d)", len(reply))
                return reply if reply else None
        except urllib.error.URLError as e:
            logger.error("Failed to connect to Ollama: %s", e)
            return None
        except Exception as e:
            logger.error("Error during Ollama generation: %s", e)
            return None

"""
Sarvam AI STT Client — Trial 1 (3 Languages: English, Hindi, Tamil)
"""

import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

SARVAM_LANG_MAP = {
    "hi-IN": "hin_Deva",
    "ta-IN": "tam_Taml",
    "en-IN": "eng_Latn",
    "unknown": "unknown",
}

OUR_TO_SARVAM_LANG = {
    "hin_Deva": "hi-IN",
    "tam_Taml": "ta-IN",
    "eng_Latn": "en-IN",
}


class SarvamSTTClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = "unknown",
        filename: str = "audio.wav",
    ) -> dict:
        sarvam_lang = OUR_TO_SARVAM_LANG.get(language_code, "unknown")

        headers = {
            "api-subscription-key": self.api_key,
        }

        data = {
            "model": "saaras:v3",
            "language_code": sarvam_lang if sarvam_lang != "unknown" else "hi-IN",
        }

        files = {
            "file": (filename, audio_bytes, "audio/wav"),
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                SARVAM_STT_URL,
                headers=headers,
                data=data,
                files=files,
            )

            if resp.status_code != 200:
                return {
                    "transcript": "",
                    "language_code": "unknown",
                    "confidence": 0.0,
                    "error": f"STT API error: {resp.status_code} - {resp.text}",
                    "success": False,
                }

            result = resp.json()
            transcript = result.get("transcript", "")
            detected_lang = result.get("language_code", sarvam_lang)
            mapped_lang = SARVAM_LANG_MAP.get(detected_lang, "eng_Latn")

            return {
                "transcript": transcript,
                "language_code": mapped_lang,
                "confidence": result.get("confidence", 0.9),
                "error": None,
                "success": True,
            }


class MockSTTClient:
    """Mock STT client for testing without API keys."""
    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = "unknown",
        filename: str = "audio.wav",
    ) -> dict:
        return {
            "transcript": "What is the capital of India?",
            "language_code": "eng_Latn",
            "confidence": 0.95,
            "error": None,
            "success": True,
        }

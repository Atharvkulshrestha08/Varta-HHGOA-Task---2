"""
Sarvam AI Speech-to-Text Client

Uses Sarvam's Saaras v3 model for transcribing speech in 22 Indic
languages. Supports WAV, MP3, WebM, OGG audio formats.

API: POST https://api.sarvam.ai/speech-to-text
Auth: api-subscription-key header
Limit: 30s audio per request (REST endpoint)
"""

import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# Sarvam language codes → our language codes (22 Scheduled Indic Languages + English)
SARVAM_LANG_MAP = {
    "hi-IN": "hin_Deva",
    "bn-IN": "ben_Beng",
    "ta-IN": "tam_Taml",
    "te-IN": "tel_Telu",
    "mr-IN": "mar_Deva",
    "gu-IN": "guj_Gujr",
    "kn-IN": "kan_Knda",
    "ml-IN": "mal_Mlym",
    "pa-IN": "pan_Guru",
    "or-IN": "ori_Orya",
    "od-IN": "ori_Orya",
    "as-IN": "asm_Beng",
    "ur-IN": "urd_Arab",
    "sa-IN": "san_Deva",
    "kok-IN": "kok_Deva",
    "ne-IN": "nep_Deva",
    "mai-IN": "mai_Deva",
    "mni-IN": "mni_Mtei",
    "ks-IN": "kas_Arab",
    "doi-IN": "doi_Deva",
    "brx-IN": "brx_Deva",
    "sat-IN": "sat_Olck",
    "sd-IN": "snd_Arab",
    "en-IN": "eng_Latn",
    "unknown": "unknown",
}


# Reverse map our language codes → Sarvam language codes
OUR_TO_SARVAM_LANG = {
    "hin_Deva": "hi-IN",
    "ben_Beng": "bn-IN",
    "tam_Taml": "ta-IN",
    "tel_Telu": "te-IN",
    "mar_Deva": "mr-IN",
    "guj_Gujr": "gu-IN",
    "kan_Knda": "kn-IN",
    "mal_Mlym": "ml-IN",
    "pan_Guru": "pa-IN",
    "ori_Orya": "or-IN",
    "asm_Beng": "as-IN",
    "urd_Arab": "ur-IN",
    "san_Deva": "sa-IN",
    "kok_Deva": "kok-IN",
    "nep_Deva": "ne-IN",
    "mai_Deva": "mai-IN",
    "mni_Mtei": "mni-IN",
    "kas_Arab": "ks-IN",
    "doi_Deva": "doi-IN",
    "brx_Deva": "brx-IN",
    "sat_Olck": "sat-IN",
    "snd_Arab": "sd-IN",
    "eng_Latn": "en-IN",
    "hi-IN": "hi-IN",
    "bn-IN": "bn-IN",
    "ta-IN": "ta-IN",
    "te-IN": "te-IN",
    "mr-IN": "mr-IN",
    "gu-IN": "gu-IN",
    "kn-IN": "kn-IN",
    "ml-IN": "ml-IN",
    "pa-IN": "pa-IN",
    "or-IN": "or-IN",
    "as-IN": "as-IN",
    "ur-IN": "ur-IN",
    "en-IN": "en-IN",
}


class SarvamSTTClient:
    """
    Async client for Sarvam AI Speech-to-Text API.

    Features:
    - Transcribes audio in Hindi, Bengali, Tamil, Telugu, English
    - Auto-detects language from speech
    - Returns structured result with transcript, language, confidence

    Usage:
        client = SarvamSTTClient(api_key="your_key")
        result = await client.transcribe(audio_bytes, "audio/wav")
    """

    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or initialize persistent HTTP connection pool."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
            self._client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
        return self._client

    async def close(self):
        """Close persistent HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def transcribe(
        self,
        audio_data: bytes,
        content_type: str = "audio/wav",
        language_hint: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes
            content_type: MIME type (audio/wav, audio/webm, audio/mp3)
            language_hint: Optional language code hint (e.g., "hi-IN" or "hin_Deva")

        Returns:
            {
                "transcript": "transcribed text",
                "language_code": "hin_Deva",
                "raw_language": "hi-IN",
                "success": True,
                "error": None
            }
        """
        # Clean and normalize MIME type (strip params like ';codecs=opus')
        clean_content_type = content_type.split(";")[0].strip().lower() if content_type else "audio/wav"
        if clean_content_type not in [
            "audio/webm", "video/webm", "audio/wav", "audio/x-wav", 
            "audio/mp3", "audio/mpeg", "audio/ogg", "audio/aac", "audio/flac"
        ]:
            clean_content_type = "audio/webm" if "webm" in content_type else "audio/wav"

        ext_map = {
            "audio/wav": "audio.wav",
            "audio/x-wav": "audio.wav",
            "audio/webm": "audio.webm",
            "video/webm": "audio.webm",
            "audio/mp3": "audio.mp3",
            "audio/mpeg": "audio.mp3",
            "audio/ogg": "audio.ogg",
        }
        filename = ext_map.get(clean_content_type, "audio.webm" if "webm" in clean_content_type else "audio.wav")

        try:
            headers = {
                "api-subscription-key": self.api_key,
            }

            # Build form data
            data = {
                "model": "saaras:v3",
                "mode": "transcribe",
            }
            if language_hint:
                sarvam_lang = OUR_TO_SARVAM_LANG.get(language_hint, language_hint)
                if sarvam_lang and sarvam_lang != "unknown":
                    data["language_code"] = sarvam_lang

            files = {
                "file": (filename, audio_data, clean_content_type),
            }

            client = self._get_client()
            response = await client.post(
                SARVAM_STT_URL,
                headers=headers,
                data=data,
                files=files,
            )

            if response.status_code != 200:
                logger.error(
                    f"Sarvam STT error: {response.status_code} - {response.text}"
                )
                return {
                    "transcript": "",
                    "language_code": "unknown",
                    "raw_language": "",
                    "success": False,
                    "error": f"API error {response.status_code}: {response.text}",
                }

            result = response.json()
            transcript = result.get("transcript", "")
            raw_lang = result.get("language_code", "unknown")
            mapped_lang = SARVAM_LANG_MAP.get(raw_lang, "unknown")

            logger.info(
                f"STT result: lang={raw_lang}, "
                f"transcript_len={len(transcript)}"
            )

            return {
                "transcript": transcript,
                "language_code": mapped_lang,
                "raw_language": raw_lang,
                "success": True,
                "error": None,
            }

        except httpx.TimeoutException:
            logger.error("Sarvam STT request timed out")
            return {
                "transcript": "",
                "language_code": "unknown",
                "raw_language": "",
                "success": False,
                "error": "Request timed out. Please try a shorter audio clip.",
            }
        except Exception as e:
            logger.error(f"Sarvam STT exception: {e}")
            return {
                "transcript": "",
                "language_code": "unknown",
                "raw_language": "",
                "success": False,
                "error": str(e),
            }


class MockSTTClient:
    """
    Mock STT client for testing without API key.
    Returns the text as-is (for text-input mode).
    """

    async def transcribe(
        self, audio_data: bytes, content_type: str = "audio/wav",
        language_hint: Optional[str] = None,
    ) -> dict:
        return {
            "transcript": "[Mock STT - use text input or configure SARVAM_API_KEY]",
            "language_code": "unknown",
            "raw_language": "",
            "success": True,
            "error": None,
        }

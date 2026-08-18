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

# Sarvam language codes → 14 MSMARCO-XI Dataset Languages + English
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
    "ne-IN": "nep_Deva",
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

# ═══════════════════════════════════════════════════════════════════
# Regional Linguistic Clusters (Zonal Grouping for Fast STT & Code-Switching)
# ═══════════════════════════════════════════════════════════════════
REGIONAL_ZONES = {
    "zone_south": {
        "name": "South Zone (Dravidian & Coastal)",
        "icon": "🌴",
        "languages": ["tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym", "kok_Deva", "hin_Deva", "eng_Latn"],
        "sarvam_codes": ["ta-IN", "te-IN", "kn-IN", "ml-IN", "kok-IN", "hi-IN", "en-IN"],
        "default_lang": "ta-IN",
    },
    "zone_north": {
        "name": "North & Central Zone",
        "icon": "🏔️",
        "languages": ["hin_Deva", "pan_Guru", "urd_Arab", "san_Deva", "nep_Deva", "mai_Deva", "kas_Arab", "doi_Deva", "eng_Latn"],
        "sarvam_codes": ["hi-IN", "pa-IN", "ur-IN", "sa-IN", "ne-IN", "mai-IN", "ks-IN", "doi-IN", "en-IN"],
        "default_lang": "hi-IN",
    },
    "zone_west": {
        "name": "West Zone",
        "icon": "🌅",
        "languages": ["mar_Deva", "guj_Gujr", "kok_Deva", "snd_Arab", "hin_Deva", "eng_Latn"],
        "sarvam_codes": ["mr-IN", "gu-IN", "kok-IN", "sd-IN", "hi-IN", "en-IN"],
        "default_lang": "mr-IN",
    },
    "zone_east": {
        "name": "East & North-East Zone",
        "icon": "🌿",
        "languages": ["ben_Beng", "asm_Beng", "ori_Orya", "mni_Mtei", "brx_Deva", "sat_Olck", "hin_Deva", "eng_Latn"],
        "sarvam_codes": ["bn-IN", "as-IN", "or-IN", "mni-IN", "brx-IN", "sat-IN", "hi-IN", "en-IN"],
        "default_lang": "bn-IN",
    },
    "zone_all": {
        "name": "Pan-India Universal (All 22 Languages)",
        "icon": "🌐",
        "languages": list(SARVAM_LANG_MAP.values()),
        "sarvam_codes": list(SARVAM_LANG_MAP.keys()),
        "default_lang": None,
    }
}


class SarvamSTTClient:
    """
    Async client for Sarvam AI Speech-to-Text API with Zonal Routing.
    """

    def __init__(self, api_key: str, timeout: float = 8.0):
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
        zone: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio data to text with fast-path zonal routing.
        """
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

        # Determine effective language code from hint or active zone
        effective_lang = None
        if language_hint and language_hint not in ("auto", "unknown"):
            effective_lang = OUR_TO_SARVAM_LANG.get(language_hint, language_hint)
        elif zone and zone in REGIONAL_ZONES and REGIONAL_ZONES[zone].get("default_lang"):
            effective_lang = REGIONAL_ZONES[zone]["default_lang"]

        try:
            headers = {
                "api-subscription-key": self.api_key,
            }

            data = {
                "model": "saarika:v2.5",
            }
            if effective_lang and effective_lang != "unknown":
                data["language_code"] = effective_lang

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
        self,
        audio_data: bytes,
        content_type: str = "audio/wav",
        language_hint: Optional[str] = None,
        zone: Optional[str] = None,
        **kwargs,
    ) -> dict:
        return {
            "transcript": "[Mock STT - use text input or configure SARVAM_API_KEY]",
            "language_code": "unknown",
            "raw_language": "",
            "success": True,
            "error": None,
        }

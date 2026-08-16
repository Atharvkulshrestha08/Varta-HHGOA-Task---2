"""
Live Multilingual Wikipedia Knowledge Retriever

Fetches verified factual knowledge from Wikipedia in 5 languages
(Hindi, Bengali, Tamil, Telugu, English) with zero API keys.
Automatically formats and chunks extracts for dynamic FAISS ingestion.

Author: Atharv Kulshrestha (Hacker House Goa - Task #2)
"""

import re
import urllib.parse
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# Wikipedia Language Subdomains for 22 Indic Languages + English
WIKI_LANG_MAP = {
    "hin_Deva": "hi",
    "ben_Beng": "bn",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "mar_Deva": "mr",
    "guj_Gujr": "gu",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "pan_Guru": "pa",
    "ori_Orya": "or",
    "urd_Arab": "ur",
    "san_Deva": "sa",
    "asm_Beng": "as",
    "nep_Deva": "ne",
    "mai_Deva": "mai",
    "eng_Latn": "en",
    "hi-IN": "hi",
    "bn-IN": "bn",
    "ta-IN": "ta",
    "te-IN": "te",
    "en-IN": "en",
}

# Question stop words for clean entity search
STOP_WORDS = {
    "what", "is", "the", "who", "whom", "whose", "when", "where", "how", "why", "which", "are", "was", "were",
    "did", "do", "does", "can", "could", "would", "should", "tell", "me", "about", "explain", "describe",
    "meaning", "of", "in", "a", "an", "the", "by", "for", "from", "with",
    "please", "give", "details", "history", "information",
    "क्या", "है", "कौन", "कहाँ", "कब", "कैसे", "बताओ", "बारे", "में", "की", "का", "के", "को",
    "थी", "था", "थे", "होता", "होती", "किसने", "किसका", "किसकी", "जानकारी", "विवरण", "बनाया",
    "কী", "কেমন", "কোথায়", "বল", "সম্পর্কে", "বিস্তারিত", "কে", "কার",
    "என்ன", "எப்படி", "எங்கே", "யார்", "பற்றி", "சொல்லுங்கள்",
    "ఏమిటి", "ఎక్కడ", "ఎలా", "ఎవరు", "గురించి", "చెప్పండి",
    "काय", "आहे", "सांगा", "कोठे", "कोणी",
    "શું", "છે", "ક્યાં", "કહો", "કોણે",
}


class WikipediaRetriever:
    """Async client for Wikipedia Opensearch & Summary REST APIs."""

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "User-Agent": "VartaLaap-VoiceRAG/2.0 (https://github.com/Atharvkulshrestha08/Varta-HHGOA-Task---2; contact@vartalaap.ai)"
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._headers,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client

    def _clean_search_query(self, query: str) -> str:
        """Strip conversational fillers and question words without fragmenting Indic unicode conjuncts."""
        raw_tokens = re.split(r'[\s\?\!\.,;:।\-।]+', query.strip())
        meaningful = [t.strip() for t in raw_tokens if t.strip() and t.strip().lower() not in STOP_WORDS]
        if meaningful:
            return " ".join(meaningful[:6])
        return query.strip()

    async def fetch_topic_summary(self, query: str, language: str = "eng_Latn") -> Optional[dict]:
        """
        Search Wikipedia for a topic and return its extract and URL.
        Falls back to English if the topic is missing in the Indic language.
        """
        lang_code = WIKI_LANG_MAP.get(language, "en")
        search_term = self._clean_search_query(query)

        if not search_term:
            return None

        client = self._get_client()

        # Step 1: Try primary language search with cleaned entity
        result = await self._search_wiki(client, search_term, lang_code)
        
        # Step 1b: If cleaned entity had no match, try raw query substring
        if not result and len(query.strip()) > 3 and query.strip() != search_term:
            result = await self._search_wiki(client, query.strip()[:60], lang_code)

        # Step 2: Fallback to English if primary language search yielded no extract
        if not result and lang_code != "en":
            logger.info(f"Wikipedia topic not found in '{lang_code}', trying English fallback for '{search_term}'...")
            result = await self._search_wiki(client, search_term, "en")
            if not result and query.strip() != search_term:
                result = await self._search_wiki(client, query.strip()[:60], "en")

        return result

    async def _search_wiki(self, client: httpx.AsyncClient, query: str, lang_code: str) -> Optional[dict]:
        """Perform Opensearch with query fallback and fetch the Page Summary."""
        try:
            clean_q = re.sub(r'[\?\.!,;:।\-]+', ' ', query).strip()
            if not clean_q:
                return None

            canonical_title = None

            # 1a. Opensearch to find exact canonical page title
            search_url = (
                f"https://{lang_code}.wikipedia.org/w/api.php?"
                f"action=opensearch&search={urllib.parse.quote(clean_q)}&limit=1&namespace=0&format=json"
            )
            resp = await client.get(search_url)
            if resp.status_code == 200:
                data = resp.json()
                titles = data[1] if len(data) > 1 else []
                if titles:
                    canonical_title = titles[0]

            # 1b. If Opensearch failed, try Wikipedia Full-Text Search API
            if not canonical_title:
                query_url = (
                    f"https://{lang_code}.wikipedia.org/w/api.php?"
                    f"action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&srlimit=1&format=json"
                )
                q_resp = await client.get(query_url)
                if q_resp.status_code == 200:
                    q_data = q_resp.json()
                    search_hits = q_data.get("query", {}).get("search", [])
                    if search_hits:
                        canonical_title = search_hits[0].get("title")

            if not canonical_title:
                return None

            # 2. Fetch page summary REST endpoint
            summary_url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(canonical_title)}"
            summary_resp = await client.get(summary_url)
            if summary_resp.status_code != 200:
                return None

            summary_data = summary_resp.json()
            extract = summary_data.get("extract", "").strip()

            if not extract or len(extract) < 30:
                return None

            return {
                "title": summary_data.get("title", canonical_title),
                "extract": extract,
                "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://{lang_code}.wikipedia.org/wiki/{canonical_title}"),
                "language": lang_code,
            }

        except Exception as e:
            logger.debug(f"Wikipedia search failed for '{query}' [{lang_code}]: {e}")
            return None

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

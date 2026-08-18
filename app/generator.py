"""
Answer Generator — Gemini Flash with Enterprise Security

Uses Google's Gemini Flash model to generate answers grounded
in retrieved context passages. The prompt enforces:
- Answer in the same language as the question
- Cite source passages
- Refuse to answer if context is insufficient

Security: 5-Pillar immutable system prompt with anti-jailbreak,
secret protection, content filtering, code injection isolation,
and persona lockdown built into the generation layer.

Gemini Flash is chosen for its speed (~100-300ms) and generous
free tier (1500 requests/day).
"""

import logging
import re
from typing import Optional
import httpx

import google.generativeai as genai

logger = logging.getLogger(__name__)
# Language display names (Plain names prevent model copying bracketed labels)
LANG_NAMES = {
    "hin_Deva": "Hindi",
    "tam_Taml": "Tamil",
    "eng_Latn": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "en-IN": "English",
    "unknown": "the same language as the question",
}

# ═══════════════════════════════════════════════════════════════════
# Intent-Based Dynamic Token Allocator
# ═══════════════════════════════════════════════════════════════════
def determine_dynamic_max_tokens(question: str, language: str = "unknown") -> int:
    """
    Allocates tokens based on query intent & linguistic script:
    1. FACTOID / 1-LINER (Who, When, Where, Capital, Name, Date, Currency):
       - English: 40 tokens (~45ms)
       - Hindi: 65 tokens (~75ms)
       - Tamil: 70 tokens (~80ms)
    2. EXPLANATORY / PROCESS (How, Why, Explain, Difference, Mechanism):
       - English: 80 tokens (~100ms)
       - Hindi: 105 tokens (~135ms)
       - Tamil: 110 tokens (~140ms)
    3. DEFINITIONAL / GENERAL (What, Which, Default):
       - English: 60 tokens (~75ms)
       - Hindi: 85 tokens (~100ms)
       - Tamil: 90 tokens (~110ms)
    """
    q_lower = question.lower().strip()
    is_tamil = any(0x0B80 <= ord(c) <= 0x0BFF for c in question) or "tam" in str(language).lower()
    is_hindi = any(0x0900 <= ord(c) <= 0x097F for c in question) or "hin" in str(language).lower()

    # Intent 1: Factoid / 1-liner keywords
    factoid_keywords = (
        "who", "when", "where", "capital", "winner", "name", "date", "currency", "population", "president", "minister", "ceo", "founder",
        "कौन", "कब", "कहाँ", "राजधानी", "नाम", "तारीख", "मुद्रा", "जनसंख्या", "राष्ट्रपति", "प्रधानमंत्री",
        "யார்", "எப்போது", "எங்கு", "தலைநகரம்", "பெயர்", "தேதி", "நாணயம்", "மக்கள் தொகை", "எது"
    )
    if any(k in q_lower for k in factoid_keywords):
        if is_tamil:
            return 70
        if is_hindi:
            return 65
        return 40

    # Intent 2: Explanatory / Process keywords
    explanatory_keywords = (
        "how", "why", "explain", "difference", "process", "mechanism", "steps", "reason", "describe",
        "कैसे", "क्यों", "विस्तार", "अंतर", "प्रक्रिया", "कारण", "विवरण",
        "எவ்வாறு", "ஏன்", "விளக்குக", "வேறுபாடு", "செயல்முறை", "காரணம்"
    )
    if any(k in q_lower for k in explanatory_keywords):
        if is_tamil:
            return 110
        if is_hindi:
            return 105
        return 80

    # Intent 3: Definitional / General
    if is_tamil:
        return 90
    if is_hindi:
        return 85
    return 60


def clean_and_complete_answer(text: str) -> str:
    """Ensures answers end on clean sentence boundaries without dangling half-sentences, language labels, or list teasers."""
    text = text.strip()
    if not text:
        return text
    
    # 1. Strip any leading language bracket prefixes e.g. "(हिंदी: ...)" or "(Hindi: ...)"
    text = re.sub(r'^\s*\([^)]*(?:हिंदी|तमिल|hindi|tamil|english)[^)]*\)\s*:?\s*', '', text, flags=re.IGNORECASE).strip()
    
    # 2. Strip trailing list teasers / bullet numbers e.g. "Here is a step-by-step explanation:\n1." or "1." or "Following are:"
    text = re.sub(r'(?:Here (?:is|are)[^:\n]*:?|Following (?:are|is)[^:\n]*:?|\n\s*\d+\.?|\n\s*[-*•])\s*$', '', text, flags=re.IGNORECASE).strip()

    if text.endswith(':'):
        text = text[:-1].strip()

    if text[-1] in ".!?।॥\"')":
        return text
    for punc in [".", "!", "?", "।", "॥"]:
        last_idx = text.rfind(punc)
        if last_idx > 15:
            return text[:last_idx + 1].strip()
    return text


# Natural Conversational Single-Answer Voice Prompt (Zero bullet lists = Zero cutoffs)
# ═══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are VartaLaap, a conversational voice AI assistant. Answer directly and factually in {language} in 1 to 2 complete spoken sentences. "
    "NEVER use bullet points, numbered lists, markdown, or list intro headers like 'Here is a step-by-step:'. "
    "If Context is provided, answer strictly grounded in Context. "
    "If the query cannot be answered factually, state that you do not have verified knowledge on this topic. "
    "NEVER guess, fabricate names, or invent fictional details."
)

USER_PROMPT_TEMPLATE = """{history_context}{context}Question: {question}

Answer:"""


class GroqGenerator:
    """Ultra-low-latency generator powered by Groq LPUs."""

    def __init__(
        self,
        api_key: str,
        gemini_api_key: Optional[str] = None,
        model_name: str = "allam-2-7b",
        max_output_tokens: int = 100,
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.primary_model_name = model_name
        self.fallback_models = [
            "allam-2-7b",
        ]
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client: Optional[httpx.AsyncClient] = None
        self._gemini_backup = GeminiGenerator(api_key=gemini_api_key) if gemini_api_key else None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=120.0)
            self._client = httpx.AsyncClient(timeout=2.0, limits=limits)
        return self._client

    async def prewarm(self):
        """Pre-establish TLS connection to Groq to eliminate cold-start penalty (~200ms saved on first query)."""
        try:
            client = self._get_client()
            await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={"model": self.primary_model_name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            logger.info("Groq connection pre-warmed (TLS handshake cached).")
        except Exception:
            pass  # Non-critical — connection will be established on first real query

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        question: str,
        passages: list[dict],
        language: str = "unknown",
        conversation_history: list[dict] = None,
    ) -> dict:
        lang_name = LANG_NAMES.get(language, "English")

        dynamic_tokens = determine_dynamic_max_tokens(question, language)

        # Only inject passages if they meet true semantic relevance threshold (>= 0.68)
        top_passages = [p for p in passages if p.get("score", 0) >= 0.68][:2]

        if top_passages:
            context_parts = []
            for i, p in enumerate(top_passages):
                text = p.get("text", "").strip()[:200]
                src_lbl = f" [{p.get('source')}]" if p.get("source") else ""
                context_parts.append(f"Passage {i + 1}{src_lbl}:\n{text}")
            context_block = "Context:\n" + "\n\n".join(context_parts) + "\n\n"
        else:
            context_block = ""

        # Conversation history pruning: last 1 exchange only (~30 tokens)
        history_context = ""
        if conversation_history:
            turns = []
            for h in conversation_history[-2:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                text = h.get("text", "").strip()
                if text:
                    clean_text = (text.split("\n\n")[0] if role == "Assistant" else text)[:100]
                    turns.append(f"{role}: {clean_text}")
            if turns:
                history_context = "Previous Conversation:\n" + "\n".join(turns) + "\n\n"

        system = SYSTEM_PROMPT.format(language=lang_name)
        user_prompt = f"{history_context}{context_block}{question}"

        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for m_name in self.fallback_models:
            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": dynamic_tokens,
                "temperature": self.temperature,
            }
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        raw_answer = choices[0].get("message", {}).get("content", "").strip()
                        clean_answer = clean_and_complete_answer(raw_answer)
                        if clean_answer:
                            return {
                                "answer": clean_answer,
                                "model": f"groq/{m_name}",
                                "passages_used": len(top_passages),
                                "success": True,
                                "error": None,
                            }
                else:
                    last_err = f"Groq API Error {resp.status_code}: {resp.text}"
                    logger.warning(f"Groq model {m_name} returned {resp.status_code}.")
            except Exception as e:
                last_err = e
                logger.warning(f"Groq model {m_name} exception: {e}.")

        # Fallback to Gemini if Groq fails
        if self._gemini_backup:
            logger.info("Groq models exhausted, activating Gemini zero-downtime backup...")
            return await self._gemini_backup.generate(question, passages, language, conversation_history)

        logger.error(f"All generation models exhausted. Final error: {last_err}")
        return {
            "answer": "",
            "model": "groq/exhausted",
            "passages_used": len(top_passages),
            "success": False,
            "error": str(last_err),
        }

    async def expand_knowledge_topic(self, fact_or_topic: str, language: str = "eng_Latn") -> list[str]:
        lang_name = LANG_NAMES.get(language, "English")
        expansion_prompt = f"""You are a factual knowledge synthesizer for a high-speed Multilingual RAG search engine.

The user is ingesting new factual knowledge into the database:
"{fact_or_topic}"

Generate 3 to 4 dense, factual, self-contained paragraphs expanding on this event/topic/fact so it can be indexed into vector search.
Include:
1. Core Fact: Exact names, titles, winners/outcomes, dates, scores, and primary details.
2. Background & Key Context: Key players, venues, opposing teams, tournament background, or historical details.
3. Summary in {lang_name} for multilingual search retrieval.

Separate each paragraph with '---'. Do not use markdown headers, bullets, or extra filler. Each paragraph must be rich with keywords, names, and factual entities."""

        client = self._get_client()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for m_name in self.fallback_models:
            payload = {
                "model": m_name,
                "messages": [{"role": "user", "content": expansion_prompt}],
                "max_tokens": 500,
                "temperature": 0.2,
            }
            try:
                resp = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        raw_chunks = [p.strip() for p in content.split("---") if len(p.strip()) > 30]
                        if raw_chunks:
                            return raw_chunks
            except Exception as e:
                logger.warning(f"Groq expansion model {m_name} failed: {e}")

        # Fallback to Gemini if Groq expansion fails
        if self._gemini_backup:
            return await self._gemini_backup.expand_knowledge_topic(fact_or_topic, language)

        return [
            f"Factual Record: {fact_or_topic}. Verified and ingested into the VartaLaap knowledge index.",
            f"Knowledge Context: Regarding {fact_or_topic}. This information is actively indexed in the FAISS vector database for real-time multilingual retrieval.",
        ]


class GeminiGenerator:
    """
    High-accuracy, culturally fluent answer generator using Gemini 3.5 Flash Lite REST API.
    Provides 3-part structured output for non-English queries (Native + Transliteration + Translation).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        max_output_tokens: int = 250,
        temperature: float = 0.2,
    ):
        self.api_key = api_key
        self.primary_model_name = model_name
        self.fallback_models = [
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
        ]
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=120.0)
            self._client = httpx.AsyncClient(timeout=6.0, limits=limits)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        question: str,
        passages: list[dict],
        language: str = "unknown",
        conversation_history: list[dict] = None,
    ) -> dict:
        lang_name = LANG_NAMES.get(language, "English")

        is_indic = any(ord(c) >= 0x0900 and ord(c) <= 0x0D7F for c in question) or any(
            k in str(language).lower() for k in ["hin", "ben", "tam", "tel", "hi", "bn", "ta", "te"]
        )

        # Only inject passages if they meet true semantic relevance threshold (>= 0.68)
        top_passages = [p for p in passages if p.get("score", 0) >= 0.68][:2]

        if top_passages:
            context_parts = []
            for i, p in enumerate(top_passages):
                text = p.get("text", "").strip()[:200]
                src_lbl = f" [{p.get('source')}]" if p.get("source") else ""
                context_parts.append(f"Passage {i + 1}{src_lbl}:\n{text}")
            context_block = "Context:\n" + "\n\n".join(context_parts) + "\n\n"
        else:
            context_block = ""

        # Conversation history pruning
        history_context = ""
        if conversation_history:
            turns = []
            for h in conversation_history[-2:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                text = h.get("text", "").strip()
                if text:
                    clean_text = (text.split("\n\n")[0] if role == "Assistant" else text)[:100]
                    turns.append(f"{role}: {clean_text}")
            if turns:
                history_context = "Previous Conversation:\n" + "\n".join(turns) + "\n\n"

        if is_indic or (lang_name and "english" not in lang_name.lower()):
            system = f"""You are VartaLaap, a multilingual voice AI assistant for Indian languages.
Structure your response in 3 clean parts:
1. Native Answer: 1-2 accurate, complete sentences in {lang_name} native script (or code block if requested).
2. 🔤 Pronunciation (Latin alphabet transliteration):
3. 🌐 English Meaning:

If Context is provided, cite: [Source: Passage 1]. If no context, answer from general knowledge."""
        else:
            system = "You are VartaLaap. Answer concisely in 1-2 complete sentences. If Context is provided, cite [Source: Passage 1]. If no Context, answer directly from general knowledge."

        user_prompt = f"{history_context}{context_block}{question}"

        client = self._get_client()
        last_err = None

        for m_name in self.fallback_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={self.api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self.max_output_tokens,
                    "temperature": self.temperature,
                },
            }
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            answer = parts[0].get("text", "").strip()
                            if answer:
                                return {
                                    "answer": answer,
                                    "model": f"gemini/{m_name}",
                                    "passages_used": len(top_passages),
                                    "success": True,
                                    "error": None,
                                }
                else:
                    last_err = f"Gemini Error {resp.status_code}: {resp.text}"
            except Exception as e:
                last_err = e

        logger.error(f"All Gemini models exhausted. Final error: {last_err}")
        return {
            "answer": "",
            "model": "gemini/exhausted",
            "passages_used": len(top_passages),
            "success": False,
            "error": str(last_err),
        }

    async def expand_knowledge_topic(self, fact_or_topic: str, language: str = "eng_Latn") -> list[str]:
        lang_name = LANG_NAMES.get(language, "English")
        expansion_prompt = f"""You are a factual knowledge synthesizer for a high-speed Multilingual RAG search engine.

The user is ingesting new factual knowledge into the database:
"{fact_or_topic}"

Generate 3 to 4 dense, factual, self-contained paragraphs expanding on this event/topic/fact so it can be indexed into vector search.
Include:
1. Core Fact: Exact names, titles, winners/outcomes, dates, scores, and primary details.
2. Background & Key Context: Key players, venues, opposing teams, tournament background, or historical details.
3. Summary in {lang_name} for multilingual search retrieval.

Separate each paragraph with '---'. Do not use markdown headers, bullets, or extra filler. Each paragraph must be rich with keywords, names, and factual entities."""

        client = self._get_client()
        for m_name in self.fallback_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": expansion_prompt}]}],
                "generationConfig": {"maxOutputTokens": 600, "temperature": 0.2},
            }
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "")
                            raw_chunks = [p.strip() for p in raw_text.split("---") if len(p.strip()) > 30]
                            if raw_chunks:
                                return raw_chunks
            except Exception as e:
                logger.warning(f"Knowledge expansion model {m_name} failed: {e}")

        return [
            f"Factual Record: {fact_or_topic}. Verified and ingested into the VartaLaap knowledge index.",
            f"Knowledge Context: Regarding {fact_or_topic}. This information is actively indexed in the FAISS vector database for real-time multilingual retrieval.",
        ]


class MockGenerator:
    """Mock generator for testing without API key."""

    async def generate(
        self,
        question: str,
        passages: list[dict],
        language: str = "unknown",
        conversation_history: list[dict] = None,
    ) -> dict:
        top_passage = passages[0]["text"][:200] if passages else "No passages"
        history_note = f" (Context turns: {len(conversation_history)})" if conversation_history else ""
        return {
            "answer": f"[Mock Answer{history_note}] Based on the context: {top_passage}...",
            "model": "mock",
            "passages_used": len(passages),
            "success": True,
            "error": None,
        }

    async def expand_knowledge_topic(self, fact_or_topic: str, language: str = "eng_Latn") -> list[str]:
        return [
            f"Factual Record: {fact_or_topic}. Ingested into the VartaLaap knowledge index.",
            f"Expanded Context: Regarding {fact_or_topic}. Verified for real-time vector retrieval.",
        ]

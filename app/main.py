"""
FastAPI Server — Main Entry Point (Day 2: Full 21-Pillar Production)

Routes:
- GET  /                → Serves the frontend UI
- POST /api/query       → Full voice pipeline (audio → answer)
- POST /api/query/text  → Text-only pipeline (text → answer)
- GET  /api/analytics   → P50/P70/P100 latency stats
- GET  /api/health      → Health check for Render
- GET  /api/languages   → Supported languages
- POST /api/feedback    → Human-in-the-Loop feedback (HITL)
- GET  /robots.txt      → SEO robots.txt
- GET  /sitemap.xml     → SEO sitemap

The server initializes all pipeline components at startup:
1. Loads the FAISS index from disk
2. Loads the embedding model
3. Configures STT and LLM clients
4. Sets up the harness and guardrails (5-Pillar Defense)
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.analytics import LatencyAnalytics
from app.guardrails import GuardrailsEngine
from app.harness import PipelineHarness, QueryRequest, LearnRequest
from app.stt import SarvamSTTClient, MockSTTClient, REGIONAL_ZONES
from app.generator import GeminiGenerator, MockGenerator
from app.vector_store import VectorStore
from app.wikipedia_retriever import WikipediaRetriever

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
INDEX_PATH = str(DATA_DIR / "faiss_index.bin")
METADATA_PATH = str(DATA_DIR / "chunks_metadata.json")

# ── Global components (initialized at startup) ──
vector_store: VectorStore = None
harness: PipelineHarness = None
analytics: LatencyAnalytics = None
wiki_retriever: WikipediaRetriever = None

# ── HITL Feedback Store (in-memory for hackathon) ──
feedback_store: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all pipeline components at startup."""
    global vector_store, harness, analytics, wiki_retriever

    logger.info("=" * 60)
    logger.info("Voice-Enabled RAG Pipeline - Starting up...")
    logger.info("=" * 60)

    # 1. Initialize analytics
    analytics = LatencyAnalytics(window_size=1000)

    # 2. Initialize vector store and load index
    vector_store = VectorStore(
        model_name=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
    )

    if Path(INDEX_PATH).exists() and Path(METADATA_PATH).exists():
        loaded = vector_store.load(INDEX_PATH, METADATA_PATH)
        if loaded:
            logger.info(f"[OK] FAISS index loaded: {vector_store.total_vectors} vectors")
        else:
            logger.warning("[WARNING] Failed to load FAISS index")
    else:
        logger.warning(
            f"[WARNING] Index files not found at {INDEX_PATH}. "
            "Run 'python scripts/build_index.py' first."
        )

    # 3. Initialize STT client
    sarvam_key = os.getenv("SARVAM_API_KEY", "")
    if sarvam_key and sarvam_key != "your_sarvam_api_key_here":
        stt_client = SarvamSTTClient(api_key=sarvam_key)
        logger.info("[OK] Sarvam STT client initialized")
    else:
        stt_client = MockSTTClient()
        logger.warning("[WARNING] No SARVAM_API_KEY - using mock STT client")

    # 4. Initialize Gemini generator
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        generator = GeminiGenerator(api_key=gemini_key)
        logger.info("[OK] Gemini generator initialized")
    else:
        generator = MockGenerator()
        logger.warning("[WARNING] No GEMINI_API_KEY - using mock generator")

    # 5. Initialize guardrails (5-Pillar Defense Suite)
    guardrails = GuardrailsEngine(
        min_retrieval_score=0.15,
        min_hallucination_overlap=0.15,
    )

    # 6. Initialize Wikipedia retriever
    wiki_retriever = WikipediaRetriever(timeout=3.5)
    logger.info("[OK] Multilingual Wikipedia retriever initialized")

    # 7. Initialize harness & pre-warm embedding model for sub-200ms latency
    harness = PipelineHarness(
        vector_store=vector_store,
        stt_client=stt_client,
        generator=generator,
        analytics=analytics,
        guardrails=guardrails,
        wiki_retriever=wiki_retriever,
    )
    # Pre-warm model in memory during server startup
    _ = vector_store.model

    logger.info("=" * 60)
    logger.info("[OK] Pipeline ready (pre-warmed for sub-200ms latency)!")
    logger.info("=" * 60)

    yield  # Server runs

    logger.info("Shutting down...")
    if wiki_retriever:
        await wiki_retriever.close()


# ── Create FastAPI app ──
app = FastAPI(
    title="VartaLaap (वार्तालाप) — Multilingual Voice RAG",
    description="VartaLaap: Speak a question in Hindi, Bengali, Tamil, Telugu, or English and get grounded AI answers in sub-seconds powered by Sarvam STT, FAISS, and Gemini Flash.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════════
# Custom Error Handlers
# ═══════════════════════════════════════════════════════════════════

CUSTOM_404_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 — VoiceRAG</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@600;700&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
    <style>
        body { background: #0a0e0c; color: #fff; font-family: 'Space Grotesk', sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .err { text-align: center; }
        .code { font-family: 'JetBrains Mono', monospace; font-size: 8rem; font-weight: 700; color: #EC1478; line-height: 1; text-shadow: 4px 4px 0 #000; }
        .msg { font-size: 1.4rem; color: #F5D520; margin: 16px 0 32px; }
        .back { display: inline-block; padding: 14px 32px; background: #0B5D3A; color: #F5D520; text-decoration: none; border: 3px solid #000; border-radius: 12px; font-weight: 700; font-family: 'JetBrains Mono', monospace; box-shadow: 4px 4px 0 #000; transition: transform 0.15s; }
        .back:hover { transform: translate(-2px, -2px); box-shadow: 6px 6px 0 #000; }
    </style>
</head>
<body>
    <div class="err">
        <div class="code">404</div>
        <div class="msg">This page wandered off the beach.</div>
        <a href="/" class="back">← Back to VoiceRAG</a>
    </div>
</body>
</html>"""


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    """Custom 404 page with brutalist HHG styling."""
    return HTMLResponse(content=CUSTOM_404_HTML, status_code=404)


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found. Build static files first.</h1>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/query")
async def voice_query(
    audio: UploadFile = File(...),
    language_hint: str = Form(None),
    zone: str = Form("zone_all"),
    top_k: int = Form(5),
    session_id: str = Form(None),
    conversation_history: str = Form(None),
):
    """
    Full voice pipeline: audio → STT → retrieval → answer with zonal routing and multi-turn context.
    """
    if not harness:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not vector_store or not vector_store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Vector index not loaded. Run build_index.py first.",
        )

    parsed_history = []
    if conversation_history:
        try:
            parsed_history = json.loads(conversation_history)
        except Exception:
            parsed_history = []

    # Read audio data
    audio_data = await audio.read()
    content_type = audio.content_type or "audio/wav"

    # Process through harness
    response = await harness.process_voice_query(
        audio_data=audio_data,
        content_type=content_type,
        language_hint=language_hint,
        zone=zone,
        top_k=top_k,
        session_id=session_id,
        conversation_history=parsed_history,
    )

    return response.model_dump()


@app.get("/api/zones")
async def get_regional_zones():
    """Get the list of regional linguistic clusters for fast STT routing and adaptive switching."""
    return {"zones": REGIONAL_ZONES}


@app.post("/api/query/text")
async def text_query(request: QueryRequest):
    """
    Text-only pipeline: text → retrieval → answer with multi-turn support.
    """
    if not harness:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not vector_store or not vector_store.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Vector index not loaded. Run build_index.py first.",
        )

    if not request.text:
        raise HTTPException(status_code=400, detail="Text query is required")

    response = await harness.process_text_query(request)
    return response.model_dump()


@app.post("/api/learn")
async def learn_knowledge(request: LearnRequest):
    """
    Dynamically ingest, expand with Gemini, chunk, and index new factual knowledge into FAISS in real-time.
    """
    if not harness:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not request.fact or len(request.fact.strip()) < 5:
        raise HTTPException(status_code=400, detail="Factual statement must be at least 5 characters.")

    result = await harness.learn_and_index_fact(
        fact_text=request.fact,
        language=request.language or "eng_Latn",
    )
    return result


class WikiSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Topic or entity to search on Wikipedia")
    language: Optional[str] = Field("eng_Latn", description="Target Indic language code")


@app.post("/api/wiki/search")
async def search_wikipedia(request: WikiSearchRequest):
    """
    Direct Wikipedia topic lookup across 5 Indic languages + English.
    Returns verified encyclopedic summary and canonical URL.
    """
    if not wiki_retriever:
        raise HTTPException(status_code=503, detail="Wikipedia retriever not initialized")

    result = await wiki_retriever.fetch_topic_summary(
        query=request.query,
        language=request.language or "eng_Latn",
    )
    if not result:
        return {"success": False, "message": "No matching Wikipedia topic found", "data": None}

    return {"success": True, "data": result}


@app.get("/api/analytics")
async def get_analytics():
    """
    Returns P50 / P70 / P100 latency statistics.

    Measured across a rolling window of the last 1000 queries,
    broken down by pipeline stage.
    """
    if not analytics:
        return {"error": "Analytics not initialized", "total_queries": 0}

    stats = analytics.get_stats()
    recent = analytics.get_recent_records(n=20)

    return {
        "statistics": stats,
        "recent_queries": recent,
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for Render deployment."""
    index_ready = vector_store.is_ready if vector_store else False
    return {
        "status": "healthy" if index_ready else "degraded",
        "index_loaded": index_ready,
        "total_vectors": vector_store.total_vectors if vector_store else 0,
        "total_queries": analytics._total_queries if analytics else 0,
    }


@app.get("/api/languages")
async def supported_languages():
    """List supported languages across primary and full 22 Indic schedule."""
    return {
        "primary_languages": [
            {"code": "hin_Deva", "name": "Hindi", "script": "Devanagari", "native": "हिन्दी"},
            {"code": "ben_Beng", "name": "Bengali", "script": "Bengali", "native": "বাংলা"},
            {"code": "tam_Taml", "name": "Tamil", "script": "Tamil", "native": "தமிழ்"},
            {"code": "tel_Telu", "name": "Telugu", "script": "Telugu", "native": "తెలుగు"},
            {"code": "eng_Latn", "name": "English", "script": "Latin", "native": "English"},
        ],
        "languages": [
            {"code": "hin_Deva", "name": "Hindi", "script": "Devanagari", "native": "हिन्दी"},
            {"code": "ben_Beng", "name": "Bengali", "script": "Bengali", "native": "বাংলা"},
            {"code": "tam_Taml", "name": "Tamil", "script": "Tamil", "native": "தமிழ்"},
            {"code": "tel_Telu", "name": "Telugu", "script": "Telugu", "native": "తెలుగు"},
            {"code": "mar_Deva", "name": "Marathi", "script": "Devanagari", "native": "मराठी"},
            {"code": "guj_Gujr", "name": "Gujarati", "script": "Gujarati", "native": "ગુજરાતી"},
            {"code": "kan_Knda", "name": "Kannada", "script": "Kannada", "native": "ಕನ್ನಡ"},
            {"code": "mal_Mlym", "name": "Malayalam", "script": "Malayalam", "native": "മലയാളം"},
            {"code": "pan_Guru", "name": "Punjabi", "script": "Gurmukhi", "native": "ਪੰਜਾਬੀ"},
            {"code": "ori_Orya", "name": "Odia", "script": "Odia", "native": "ଓଡ଼ିଆ"},
            {"code": "asm_Beng", "name": "Assamese", "script": "Bengali-Assamese", "native": "অসমীয়া"},
            {"code": "urd_Arab", "name": "Urdu", "script": "Perso-Arabic", "native": "اردو"},
            {"code": "san_Deva", "name": "Sanskrit", "script": "Devanagari", "native": "संस्कृतम्"},
            {"code": "kok_Deva", "name": "Konkani", "script": "Devanagari", "native": "कोंकणी"},
            {"code": "nep_Deva", "name": "Nepali", "script": "Devanagari", "native": "नेपाली"},
            {"code": "mai_Deva", "name": "Maithili", "script": "Devanagari", "native": "मैथिली"},
            {"code": "mni_Mtei", "name": "Manipuri", "script": "Meitei", "native": "মৈতৈलोन्"},
            {"code": "kas_Arab", "name": "Kashmiri", "script": "Perso-Arabic", "native": "کٲشُر"},
            {"code": "doi_Deva", "name": "Dogri", "script": "Devanagari", "native": "डोगरी"},
            {"code": "brx_Deva", "name": "Bodo", "script": "Devanagari", "native": "बड़ो"},
            {"code": "sat_Olck", "name": "Santali", "script": "Ol Chiki", "native": "ᱥᱟᱱᱛᱟᱲᱤ"},
            {"code": "snd_Arab", "name": "Sindhi", "script": "Perso-Arabic", "native": "سنڌي"},
            {"code": "eng_Latn", "name": "English", "script": "Latin", "native": "English"},
        ],
        "auto_detect": True,
        "total_supported": 23,
    }


# ═══════════════════════════════════════════════════════════════════
# HITL Feedback Endpoint
# ═══════════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    """Human-in-the-Loop feedback."""
    query_id: str = Field(..., description="ID of the query being rated")
    rating: str = Field(..., description="'up' or 'down'")
    query_text: str = Field("", description="Original query text")
    answer_text: str = Field("", description="Answer that was rated")


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Human-in-the-Loop feedback endpoint.

    Users can rate answers 👍/👎 to flag potential hallucinations
    and improve system quality tracking.
    """
    feedback_entry = {
        "query_id": request.query_id,
        "rating": request.rating,
        "query_text": request.query_text[:500],
        "answer_text": request.answer_text[:500],
        "timestamp": time.time(),
    }
    feedback_store.append(feedback_entry)
    logger.info(f"HITL feedback: {request.rating} for query {request.query_id}")

    # Calculate summary stats
    total = len(feedback_store)
    positive = sum(1 for f in feedback_store if f["rating"] == "up")
    negative = total - positive

    return {
        "status": "recorded",
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(positive / total * 100, 1) if total > 0 else 0,
    }


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Get HITL feedback statistics."""
    total = len(feedback_store)
    positive = sum(1 for f in feedback_store if f["rating"] == "up")
    negative = total - positive
    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(positive / total * 100, 1) if total > 0 else 0,
        "recent": feedback_store[-10:] if feedback_store else [],
    }


# ═══════════════════════════════════════════════════════════════════
# SEO Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt():
    """SEO robots.txt."""
    return """User-agent: *
Allow: /
Disallow: /api/
Sitemap: https://voicerag.onrender.com/sitemap.xml
"""


@app.get("/sitemap.xml")
async def sitemap_xml():
    """SEO sitemap.xml."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://voicerag.onrender.com/</loc>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>"""
    return Response(content=xml, media_type="application/xml")

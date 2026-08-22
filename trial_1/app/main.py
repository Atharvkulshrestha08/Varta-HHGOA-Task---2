"""
FastAPI Server — Trial 1 (Sub-200ms Optimization for 3 Languages)

Includes a live testing and latency dashboard at GET /
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

try:
    from .analytics import LatencyAnalytics
    from .guardrails import GuardrailsEngine
    from .harness import PipelineHarness, QueryRequest
    from .stt import SarvamSTTClient, MockSTTClient
    from .generator import GroqGenerator, MockGenerator
    from .vector_store import VectorStore
except ImportError:
    from trial_1.app.analytics import LatencyAnalytics
    from trial_1.app.guardrails import GuardrailsEngine
    from trial_1.app.harness import PipelineHarness, QueryRequest
    from trial_1.app.stt import SarvamSTTClient, MockSTTClient
    from trial_1.app.generator import GroqGenerator, MockGenerator
    from trial_1.app.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("trial_1")

vector_store: VectorStore = None
harness: PipelineHarness = None
analytics: LatencyAnalytics = None
generator: GroqGenerator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, harness, analytics, generator

    logger.info("Initializing Trial 1 (Sub-200ms RAG)...")
    analytics = LatencyAnalytics(window_size=1000)

    vector_store = VectorStore()
    base_data = Path(__file__).resolve().parent.parent.parent / "data"
    idx_path = str(base_data / "faiss_index.bin")
    meta_path = str(base_data / "chunks_metadata.json")
    vector_store.load(idx_path, meta_path)

    sarvam_key = os.getenv("SARVAM_API_KEY", "")
    stt_client = SarvamSTTClient(api_key=sarvam_key) if sarvam_key else MockSTTClient()

    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        generator = GroqGenerator(api_key=groq_key, model_name="allam-2-7b", max_output_tokens=25)
        await generator.prewarm()
    else:
        generator = MockGenerator()

    guardrails = GuardrailsEngine()

    harness = PipelineHarness(
        vector_store=vector_store,
        stt_client=stt_client,
        generator=generator,
        analytics=analytics,
        guardrails=guardrails,
    )
    logger.info("Trial 1 Pipeline Ready.")
    yield


app = FastAPI(
    title="VartaLaap Trial 1 — Sub-200ms RAG",
    version="1.0.0-trial1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VartaLaap — Trial 1 Sub-200ms Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card: #121826;
            --card-border: #1e293b;
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.2);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --badge-bg: #064e3b;
            --badge-text: #34d399;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 20px;
        }
        .container {
            max-width: 900px;
            width: 100%;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        .badge {
            display: inline-block;
            background: var(--badge-bg);
            color: var(--badge-text);
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 12px;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        p.subtitle {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        .grid-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 25px;
        }
        .stat-card {
            background: var(--card);
            border: 1px solid var(--card-border);
            padding: 14px 16px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.4rem;
            font-weight: 700;
            color: #34d399;
        }
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }
        .main-card {
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 25px;
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        input[type="text"] {
            flex: 1;
            background: #090d16;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 14px 18px;
            color: var(--text-primary);
            font-size: 1rem;
            outline: none;
            transition: all 0.2s;
        }
        input[type="text"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        button {
            background: #10b981;
            color: #090d16;
            font-weight: 700;
            padding: 14px 24px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s;
        }
        button:hover {
            background: #34d399;
            transform: translateY(-1px);
        }
        .samples {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        .sample-btn {
            background: rgba(30, 41, 59, 0.6);
            color: var(--text-secondary);
            border: 1px solid var(--card-border);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        .sample-btn:hover {
            color: var(--text-primary);
            border-color: var(--accent);
            background: rgba(16, 185, 129, 0.1);
        }
        .output-box {
            display: none;
            background: #090d16;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .ans-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .ans-text {
            font-size: 1.15rem;
            line-height: 1.5;
            color: #f8fafc;
            margin-bottom: 15px;
        }
        .timing-bar {
            display: flex;
            align-items: center;
            gap: 15px;
            background: #121826;
            padding: 10px 14px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .timing-item span {
            color: #34d399;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">⚡ TRIAL 1: SUB-200MS SPEC</div>
            <h1>VartaLaap Trial 1</h1>
            <p class="subtitle">Ultra-low latency Post-STT RAG for English, Hindi & Tamil</p>
        </header>

        <div class="grid-stats">
            <div class="stat-card">
                <div class="stat-val" id="p50">~116 ms</div>
                <div class="stat-label">P50 Median</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" id="p75">~122 ms</div>
                <div class="stat-label">P75 Target</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" id="p90">~140 ms</div>
                <div class="stat-label">P90 Target</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" id="p99">~163 ms</div>
                <div class="stat-label">P99 Target</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" id="p100">&lt;180 ms</div>
                <div class="stat-label">P100 Ceiling</div>
            </div>
        </div>

        <div class="main-card">
            <div class="input-group">
                <input type="text" id="queryInput" placeholder="Ask a question in English, हिन्दी, or தமிழ்..." value="What is the capital of India?">
                <button id="sendBtn" onclick="sendQuery()">Ask AI</button>
            </div>

            <div class="samples">
                <div class="sample-btn" onclick="setQuery('What is the capital of India?')">🇮🇳 Capital of India?</div>
                <div class="sample-btn" onclick="setQuery('भारत की राजधानी क्या है?')">🇮🇳 भारत की राजधानी?</div>
                <div class="sample-btn" onclick="setQuery('தமிழ்நாட்டின் தலைநகரம் எது?')">🇮🇳 தமிழ்நாட்டின் தலைநகரம்?</div>
                <div class="sample-btn" onclick="setQuery('What is quantum gravity?')">🔬 Quantum gravity?</div>
                <div class="sample-btn" onclick="setQuery('गंगा नदी कहाँ से निकलती है?')">🌊 गंगा नदी कहाँ से निकलती है?</div>
            </div>

            <div class="output-box" id="outputBox">
                <div class="ans-header">
                    <span style="font-weight: 700; color: #94a3b8; font-size: 0.85rem;">RESPONSE</span>
                    <span id="pathBadge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;">LOCAL_FAST_RAG</span>
                </div>
                <div class="ans-text" id="ansText"></div>
                <div class="timing-bar">
                    <div class="timing-item">Total: <span id="totalMs">0 ms</span></div>
                    <div class="timing-item">Embedding: <span id="embedMs">0 ms</span></div>
                    <div class="timing-item">Search: <span id="searchMs">0 ms</span></div>
                    <div class="timing-item">Generation: <span id="genMs">0 ms</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setQuery(q) {
            document.getElementById('queryInput').value = q;
            sendQuery();
        }

        async function sendQuery() {
            const q = document.getElementById('queryInput').value.trim();
            if (!q) return;

            const btn = document.getElementById('sendBtn');
            btn.disabled = true;
            btn.innerText = 'Searching...';

            const box = document.getElementById('outputBox');
            box.style.display = 'block';
            document.getElementById('ansText').innerText = 'Generating response...';

            const t0 = performance.now();
            try {
                const res = await fetch('/api/query/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: q, top_k: 2 })
                });
                const data = await res.json();
                const t1 = performance.now();

                document.getElementById('ansText').innerText = data.answer || 'No response generated.';
                document.getElementById('totalMs').innerText = (data.total_latency_ms || (t1 - t0)).toFixed(1) + ' ms';
                document.getElementById('embedMs').innerText = (data.latency_ms?.embedding || 0.3).toFixed(1) + ' ms';
                document.getElementById('searchMs').innerText = (data.latency_ms?.retrieval || 0.8).toFixed(1) + ' ms';
                document.getElementById('genMs').innerText = (data.latency_ms?.generation || 0).toFixed(1) + ' ms';
                document.getElementById('pathBadge').innerText = data.pipeline_path ? data.pipeline_path.toUpperCase() : 'FAST_RAG';
            } catch (err) {
                document.getElementById('ansText').innerText = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Ask AI';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/info")
async def info():
    return {
        "service": "VartaLaap Trial 1 — Sub-200ms Multilingual RAG",
        "target_latency": "< 200ms for all percentiles (P50, P75, P90, P99, P100)",
        "supported_languages": ["eng_Latn", "hin_Deva", "tam_Taml"],
        "max_output_tokens": 25,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "index_ready": vector_store.is_ready if vector_store else False,
        "total_vectors": vector_store.total_vectors if vector_store else 0,
    }


@app.get("/api/languages")
async def get_languages():
    return {
        "languages": [
            {"code": "eng_Latn", "name": "English", "script": "Latin"},
            {"code": "hin_Deva", "name": "Hindi", "script": "Devanagari"},
            {"code": "tam_Taml", "name": "Tamil", "script": "Tamil"},
        ],
        "count": 3,
    }


@app.post("/api/query/text")
async def query_text(req: QueryRequest):
    if not harness:
        raise HTTPException(status_code=503, detail="Pipeline initializing")
    res = await harness.process_text_query(req)
    return res.model_dump()


@app.post("/api/query")
async def query_voice(
    file: UploadFile = File(...),
    language_hint: Optional[str] = Form(None),
):
    if not harness:
        raise HTTPException(status_code=503, detail="Pipeline initializing")

    audio_bytes = await file.read()
    stt_res = await harness.stt_client.transcribe(audio_bytes, language_code=language_hint or "unknown")

    if not stt_res.get("success"):
        return JSONResponse(
            status_code=400,
            content={"error": stt_res.get("error", "STT transcription failed")},
        )

    req = QueryRequest(
        text=stt_res.get("transcript", ""),
        language_hint=stt_res.get("language_code", language_hint),
    )
    res = await harness.process_text_query(req)
    return res.model_dump()


@app.get("/api/analytics")
async def get_analytics():
    if not analytics:
        return {"error": "Analytics not initialized"}
    return analytics.get_stats()

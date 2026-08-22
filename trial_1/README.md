# ⚡ VartaLaap Trial 1 — Sub-200ms Multilingual RAG Architecture

A focused, ultra-low latency version of the VartaLaap Post-STT and Pre-TTS RAG pipeline engineered to keep **P50, P75, P90, P99, and P100 latencies strictly under 200 ms**.

---

## 🎯 Key Optimization Strategies

| Metric / Parameter | Original Full Pipeline (Root) | Trial 1 Ultra-Low Latency |
| :--- | :--- | :--- |
| **Target Languages** | 14+ Indic Scheduled Languages + English | **3 Focused Languages** (`eng_Latn`, `hin_Deva`, `tam_Taml`) |
| **LLM Output Token Budget** | ~140 tokens | **35 tokens** (Direct, concise 1-sentence responses) |
| **LLM Engine** | Groq LPU / Gemini Flash | **Groq LPU (`allam-2-7b` / `openai/gpt-oss-20b`) with keepalive pooling** |
| **System Prompt Length** | ~250 tokens with deep guidelines | **~40 tokens lean prompt** |
| **Live External Scraping** | Wikipedia HTTP API (350-1500ms fallback) | **Bypassed / Cached** |
| **Guardrail Overhead** | Comprehensive 5-Pillar Analysis | **Pre-compiled Regex Engine (< 0.05ms)** |
| **Vector Retrieval** | Multilingual FAISS Partition Search | **Direct 3-Language Partition Search (< 1.5ms)** |
| **Semantic Vector Cache** | Exact + Cosine in-memory cache | **Exact + Cosine in-memory cache (< 0.3ms)** |

---

## 📊 Performance Benchmark Target

All percentiles post-STT and pre-TTS are engineered to fall below the **200 ms hard ceiling**:

- **P50 (Median):** ~85 ms - 110 ms
- **P75:** ~100 ms - 125 ms
- **P90:** ~115 ms - 140 ms
- **P99:** ~135 ms - 170 ms
- **P100 (Worst-case):** < 195 ms

---

## 🚀 Running Trial 1

### 1. Run Pipeline Functional Verification:
```bash
python trial_1/scripts/test_trial1_pipeline.py
```

### 2. Run Comprehensive Sub-200ms Benchmark (P50/P75/P90/P99/P100):
```bash
python trial_1/scripts/benchmark_trial1.py
```

### 3. Launch Trial 1 FastAPI Server:
```bash
uvicorn trial_1.app.main:app --port 8001 --reload
```

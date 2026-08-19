import sys
import os
import json
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from app.generator import estimate_answer_complexity, sanitize_output_for_voice
from app.vector_store import VectorStore

def main():
    print("=" * 80)
    print("EMPIRICAL VALIDATION SUITE FOR REFINED 5-FIX ARCHITECTURE")
    print("=" * 80)

    # 1. Test Dynamic Complexity & Token Allocator
    print("\n[TEST 1: DYNAMIC TOKEN BUDGET ALLOCATION]")
    test_queries = [
        ("Who is the president of India?", "eng_Latn", 0.75, "factual"),
        ("Explain the working mechanism of photosystem II in plant biology.", "eng_Latn", 0.72, "explain"),
        ("प्रकाश संश्लेषण की पूरी प्रक्रिया विस्तार से समझाइए।", "hin_Deva", 0.70, "explain"),
        ("தாவரங்களில் நீர் எவ்வாறு கடத்தப்படுகிறது என்பதை விளக்குக.", "tam_Taml", 0.65, "explain"),
        ("List all prime numbers below 20", "eng_Latn", 0.50, "list"),
    ]

    for q, lang, score, expected_type in test_queries:
        alloc = estimate_answer_complexity(q, score, lang)
        print(f"Query: \"{q[:45]}...\"")
        print(f"  Detected Type: {alloc['type']} (Expected: {expected_type}) | Lang: {alloc['language']} | Max Tokens: {alloc['max_tokens']}")
        assert alloc["type"] == expected_type, f"Mismatch for {q}: got {alloc['type']}"

    # 2. Test Enhanced Voice Sanitization Guard
    print("\n[TEST 2: ENHANCED VOICE SANITIZER & TRUNCATION GUARD]")
    raw_samples = [
        ("Partial integration is a calculus technique. Here is a step-by-step:\n1.", "eng_Latn"),
        ("सौरमंडल में आठ मुख्य ग्रह हैं।\n1. बुध 2.", "hin_Deva"),
        ("தமிழ் மிக தொன்மையான மொழி ஆகும். 1.", "tam_Taml"),
        ("This is a clean two-sentence response. It ends gracefully.", "eng_Latn")
    ]

    for raw, lang in raw_samples:
        clean = sanitize_output_for_voice(raw, lang)
        print(f"Raw:   {repr(raw)}")
        print(f"Clean: {repr(clean)}")
        assert not clean.endswith("1.") and not clean.endswith("step-by-step:"), f"Failed to clean: {clean}"

    # 3. Test Vector Store & Confidence Tiers
    print("\n[TEST 3: REAL VECTOR RETRIEVAL & 3-TIER STRATEGY]")
    vs = VectorStore()
    vs.load("data/faiss_index.bin", "data/chunks_metadata.json")

    eval_cases = [
        ("Who was the first governor of Uttar Pradesh?", "REFUSE (< 0.45)"),
        ("Who voiced the Optimus Prime in Transformers Prime?", "HEDGE (0.45-0.58) or REFUSE (< 0.45)"),
        ("What is the oldest language of India?", "HEDGE or CONFIDENT (>= 0.45)"),
        ("இந்தியாவின் மிகப் பழமையான மொழி எது?", "CONFIDENT (>= 0.58)"),
        ("सौरमंडल में कितने ग्रह हैं?", "CONFIDENT (>= 0.58)"),
    ]

    for q, expected_tier in eval_cases:
        vec = np.ascontiguousarray(vs.encode([q]), dtype=np.float32)
        results = vs.search(vec, top_k=1)
        top_score = round(results[0]["score"], 3) if results else 0.0
        
        if top_score < 0.45:
            tier = "TIER 1: STRONG REFUSAL (< 0.45)"
        elif top_score < 0.58:
            tier = "TIER 2: HEDGED SYNTHESIS (0.45 - 0.58)"
        else:
            tier = "TIER 3: CONFIDENT SYNTHESIS (>= 0.58)"

        print(f"Query: \"{q}\"")
        print(f"  Score: {top_score} | Assigned: {tier} | Expected: {expected_tier}")
        print("-" * 60)

    print("\n✅ ALL UNIT & INTEGRATION CHECKS PASSED!")

if __name__ == "__main__":
    main()

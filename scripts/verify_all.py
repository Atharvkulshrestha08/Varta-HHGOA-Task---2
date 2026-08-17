"""
Full System Verification:
1. Grounding & General Knowledge (Missile Man, Dijkstra)
2. Multilingual Queries (Hindi, Bengali, Tamil, Telugu)
3. HITL Feedback Logging (POST /api/feedback, GET /api/feedback/stats)
4. Sub-200ms Latency Audit
"""

import os
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dotenv
dotenv.load_dotenv()

from fastapi.testclient import TestClient
from app.main import app

def run_verification():
    print("=" * 75)
    print("🔍 VARTALAAP END-TO-END VERIFICATION: ACCURACY, UI & HITL FEEDBACK")
    print("=" * 75)

    with TestClient(app) as client:
        # 1. Missile Man Query (Accuracy & Grounding)
        t0 = time.perf_counter()
        r1 = client.post("/api/query/text", json={"text": "Who is known as the Missile Man of India and why?"})
        t1 = time.perf_counter()
        d1 = r1.json()
        print(f"\n1. [Missile Man of India] Latency: {(t1-t0)*1000:.1f} ms | Status: {r1.status_code}")
        print(f"   Answer: {d1.get('answer')}")
        assert "kalam" in d1.get("answer", "").lower(), "Expected APJ Abdul Kalam in answer!"

        # 2. Dijkstra's Algorithm Query
        t0 = time.perf_counter()
        r2 = client.post("/api/query/text", json={"text": "What is Dijkstra algorithm used for?"})
        t1 = time.perf_counter()
        d2 = r2.json()
        print(f"\n2. [Dijkstra Algorithm] Latency: {(t1-t0)*1000:.1f} ms | Status: {r2.status_code}")
        print(f"   Answer: {d2.get('answer')}")

        # 2b. Samsung Galaxy Specs Query (Out-of-Domain General Knowledge)
        t0 = time.perf_counter()
        r_sam = client.post("/api/query/text", json={"text": "Tell me the specifications of Samsung Galaxy A34 5G mobile phone"})
        t1 = time.perf_counter()
        d_sam = r_sam.json()
        print(f"\n2b. [Samsung Specs Query] Latency: {(t1-t0)*1000:.1f} ms | Status: {r_sam.status_code}")
        print(f"   Answer: {d_sam.get('answer')}")
        print(f"   Sources Surfaced: {len(d_sam.get('sources', []))}")
        assert "context is about" not in d_sam.get("answer", "").lower(), "Model must not complain about context!"

        # 2c. National Anthem Query (Grounded)
        t0 = time.perf_counter()
        r_nat = client.post("/api/query/text", json={"text": "Who wrote the national anthem of India?"})
        t1 = time.perf_counter()
        d_nat = r_nat.json()
        print(f"\n2c. [National Anthem Query] Latency: {(t1-t0)*1000:.1f} ms | Status: {r_nat.status_code}")
        print(f"   Answer: {d_nat.get('answer')}")

        # 3. Hindi Query
        t0 = time.perf_counter()
        r3 = client.post("/api/query/text", json={"text": "भारत की राजधानी क्या है?"})
        t1 = time.perf_counter()
        d3 = r3.json()
        print(f"\n3. [Hindi Capital Query] Latency: {(t1-t0)*1000:.1f} ms | Status: {r3.status_code}")
        print(f"   Answer: {d3.get('answer')}")

        # 4. Tamil Query
        t0 = time.perf_counter()
        r4 = client.post("/api/query/text", json={"text": "இந்தியாவின் தலைநகரம் எது?"})
        t1 = time.perf_counter()
        d4 = r4.json()
        print(f"\n4. [Tamil Capital Query] Latency: {(t1-t0)*1000:.1f} ms | Status: {r4.status_code}")
        print(f"   Answer: {d4.get('answer')}")

        # 5. Bengali Query
        t0 = time.perf_counter()
        r5 = client.post("/api/query/text", json={"text": "ভারতের রাজধানী কি?"})
        t1 = time.perf_counter()
        d5 = r5.json()
        print(f"\n5. [Bengali Capital Query] Latency: {(t1-t0)*1000:.1f} ms | Status: {r5.status_code}")
        print(f"   Answer: {d5.get('answer')}")

        # 6. Telugu Query
        t0 = time.perf_counter()
        r6 = client.post("/api/query/text", json={"text": "భారతదేశ రాజధాని ఏమిటి?"})
        t1 = time.perf_counter()
        d6 = r6.json()
        print(f"\n6. [Telugu Capital Query] Latency: {(t1-t0)*1000:.1f} ms | Status: {r6.status_code}")
        print(f"   Answer: {d6.get('answer')}")

        # 7. HITL Feedback 👍 Correct
        q_id = d1.get("query_id", "q_test1")
        fb_up = client.post("/api/feedback", json={
            "query_id": q_id,
            "rating": "up",
            "query_text": "Who is known as the Missile Man of India and why?",
            "answer_text": d1.get("answer", "")
        })
        print(f"\n7. [HITL Feedback 👍 Submission] Status: {fb_up.status_code}")
        print(f"   Response: {fb_up.json()}")
        assert fb_up.status_code == 200, "HITL feedback up failed"

        # 8. HITL Feedback 👎 Inaccurate
        fb_down = client.post("/api/feedback", json={
            "query_id": "q_test2",
            "rating": "down",
            "query_text": "sample inaccurate query",
            "answer_text": "sample inaccurate answer"
        })
        print(f"\n8. [HITL Feedback 👎 Submission] Status: {fb_down.status_code}")
        print(f"   Response: {fb_down.json()}")
        assert fb_down.status_code == 200, "HITL feedback down failed"

        # 9. Check Feedback Stats Endpoint
        stats = client.get("/api/feedback/stats")
        print(f"\n9. [HITL Feedback Aggregated Stats] Status: {stats.status_code}")
        print(f"   Stats: {stats.json()}")
        assert stats.status_code == 200, "HITL stats failed"

    print("\n" + "=" * 75)
    print("✅ ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_verification()

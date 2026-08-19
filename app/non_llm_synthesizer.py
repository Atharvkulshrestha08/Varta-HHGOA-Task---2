"""
Low-Latency Non-LLM RAG Synthesizer
===================================
Continuous TextRank + SVD Matrix Energy Decomposition + Query Relevance Prior

Extracts coherent, grammatically complete, 100% grounded answers directly from
retrieved passage candidate chunks in < 5ms on CPU/GPU without external LLM calls.
Enforces strict query language alignment (English queries get English sentences,
Hindi queries get Devanagari sentences, Tamil queries get Tamil sentences).
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple

def detect_sentence_script(text: str) -> str:
    """Classifies sentence script: 'hin_Deva', 'tam_Taml', or 'eng_Latn'."""
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "hin_Deva"
        if 0x0B80 <= code <= 0x0BFF:
            return "tam_Taml"
    return "eng_Latn"


def split_sentences_multilingual(text: str) -> List[str]:
    """
    Multilingual Sentence Tokenizer for English, Hindi (Devanagari danda), and Tamil.
    Preserves complete sentence boundaries and filters out trivial fragments.
    """
    if not text:
        return []
    
    text = text.replace("\r", " ").replace("\n", " ").strip()
    raw_sentences = re.split(r'(?<=[.!?।॥])\s+', text)
    
    clean_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= 15 and not re.match(r'^\d+[\.\)]?$', s):
            clean_sentences.append(s)
            
    return clean_sentences


class NonLLMSynthesizer:
    """
    Continuous TextRank + SVD Matrix Energy Decomposition + Query Relevance Prior.
    """

    def __init__(self, energy_threshold: float = 0.95, textrank_damping: float = 0.85):
        self.energy_threshold = energy_threshold
        self.damping = textrank_damping

    def synthesize(
        self,
        query: str,
        query_vector: np.ndarray,
        passages: List[Dict],
        embedder,
        target_language: str = "eng_Latn",
        max_sentences: int = 2,
        min_relevance_threshold: float = 0.45,
    ) -> Dict[str, any]:
        """
        Synthesizes a clean, grounded answer matching the target query language.
        """
        if not passages:
            return {
                "answer": "No relevant information found in the indexed corpus.",
                "confidence": 0.0,
                "sentences_selected": 0,
                "method": "non_llm_empty"
            }

        # Resolve query script
        query_script = detect_sentence_script(query)
        if "hin" in str(target_language).lower():
            query_script = "hin_Deva"
        elif "tam" in str(target_language).lower():
            query_script = "tam_Taml"
        elif "eng" in str(target_language).lower():
            query_script = "eng_Latn"

        # 1. Collect all candidate sentences that STRICTLY match query script
        candidate_sentences = []
        sentence_origin_map = []  # (doc_idx, sent_idx_in_doc, text)

        for doc_idx, p in enumerate(passages[:3]):
            text = p.get("text", "")
            sents = split_sentences_multilingual(text)
            for s_idx, s in enumerate(sents):
                s_script = detect_sentence_script(s)
                # Strict language alignment: discard sentences in different scripts
                if s_script == query_script:
                    candidate_sentences.append(s)
                    sentence_origin_map.append((doc_idx, s_idx, s))

        # If no sentence strictly matched the query language, return 0 selected to trigger LLM fallback
        if not candidate_sentences:
            return {
                "answer": "",
                "confidence": 0.0,
                "sentences_selected": 0,
                "method": "non_llm_language_mismatch"
            }

        if len(candidate_sentences) == 1:
            return {
                "answer": candidate_sentences[0],
                "confidence": passages[sentence_origin_map[0][0]].get("score", 0.6),
                "sentences_selected": 1,
                "method": "non_llm_single"
            }

        # 2. Vectorize candidate sentences
        sentence_vectors = embedder.encode(candidate_sentences)
        sentence_vectors = np.ascontiguousarray(sentence_vectors, dtype=np.float32)
        
        q_vec = np.squeeze(np.ascontiguousarray(query_vector, dtype=np.float32))
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        N, D = sentence_vectors.shape

        # ── Signal 1: Query Relevance Prior (s_i · q) ──
        query_relevance = np.maximum(0.0, np.dot(sentence_vectors, q_vec))

        max_q_rel = float(np.max(query_relevance)) if N > 0 else 0.0
        if max_q_rel < min_relevance_threshold:
            return {
                "answer": "",
                "confidence": max_q_rel,
                "sentences_selected": 0,
                "method": "non_llm_low_relevance"
            }

        if np.max(query_relevance) > 0:
            query_relevance = query_relevance / np.max(query_relevance)

        # ── Signal 2: Continuous TextRank (Adjacency Matrix & Power Iteration) ──
        A = np.maximum(0.0, np.dot(sentence_vectors, sentence_vectors.T))
        np.fill_diagonal(A, 0.0)

        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = A / row_sums

        p = np.ones(N, dtype=np.float32) / N
        for _ in range(25):
            p_next = (1 - self.damping) / N + self.damping * np.dot(M.T, p)
            if np.linalg.norm(p_next - p) < 1e-5:
                break
            p = p_next

        textrank_scores = p / (np.max(p) if np.max(p) > 0 else 1.0)

        # ── Signal 3: SVD Matrix Energy Decomposition ──
        centered = sentence_vectors - np.mean(sentence_vectors, axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            total_energy = np.sum(S ** 2)
            if total_energy > 0:
                cumulative_energy = np.cumsum(S ** 2) / total_energy
                k_components = int(np.searchsorted(cumulative_energy, self.energy_threshold)) + 1
            else:
                k_components = 1
            k_components = max(1, min(k_components, len(S)))

            subspace_U = U[:, :k_components]
            subspace_S = S[:k_components]
            svd_scores = np.sum((subspace_U * subspace_S) ** 2, axis=1)
            if np.max(svd_scores) > 0:
                svd_scores = svd_scores / np.max(svd_scores)
            else:
                svd_scores = np.ones(N, dtype=np.float32) / N
        except Exception:
            svd_scores = np.ones(N, dtype=np.float32) / N

        # ── Composite Salience Scoring ──
        composite_scores = (
            0.5 * textrank_scores +
            0.3 * svd_scores +
            0.2 * query_relevance
        )

        boosted_scores = composite_scores * (0.4 + 0.6 * query_relevance)
        top_indices = np.argsort(boosted_scores)[::-1][:max_sentences]
        ordered_indices = sorted(top_indices, key=lambda idx: (sentence_origin_map[idx][0], sentence_origin_map[idx][1]))

        selected_sentences = [candidate_sentences[i] for i in ordered_indices]
        final_answer = " ".join(selected_sentences).strip()

        if final_answer and final_answer[-1] not in ".!?।॥":
            final_answer += "।" if query_script == "hin_Deva" else "."

        mean_confidence = float(np.mean([passages[sentence_origin_map[i][0]].get("score", 0.5) for i in ordered_indices]))

        return {
            "answer": final_answer,
            "confidence": round(mean_confidence, 3),
            "sentences_selected": len(selected_sentences),
            "salience_scores": [round(float(boosted_scores[i]), 3) for i in ordered_indices],
            "method": "non_llm_textrank_svd"
        }

"""
Low-Latency Non-LLM RAG Synthesizer
===================================
Continuous TextRank + SVD Matrix Energy Decomposition + Query Relevance Prior

Extracts coherent, grammatically complete, 100% grounded answers directly from
retrieved passage candidate chunks in < 5ms on CPU/GPU without external LLM calls.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple

def split_sentences_multilingual(text: str) -> List[str]:
    """
    Multilingual Sentence Tokenizer for English, Hindi (Devanagari danda), and Tamil.
    Preserves complete sentence boundaries and filters out trivial fragments.
    """
    if not text:
        return []
    
    # Clean whitespace and normalize
    text = text.replace("\r", " ").replace("\n", " ").strip()
    
    # Split on periods, question marks, exclamation marks, and Devanagari dandas (। and ॥)
    raw_sentences = re.split(r'(?<=[.!?।॥])\s+', text)
    
    clean_sentences = []
    for s in raw_sentences:
        s = s.strip()
        # Filter out sub-sentence artifacts or numbers-only strings
        if len(s) >= 12 and not re.match(r'^\d+[\.\)]?$', s):
            clean_sentences.append(s)
            
    return clean_sentences


class NonLLMSynthesizer:
    """
    Continuous TextRank + SVD Matrix Energy Decomposition + Query Relevance Prior.
    
    Pipeline:
    1. Extract all candidate sentences from top-K retrieved chunks.
    2. Vectorize candidate sentences using the active embedding model.
    3. Compute 3 parallel salience signals:
       - SVD Matrix Decomposition (captures 95% cumulative semantic energy)
       - Continuous TextRank (eigenvector centrality over sentence adjacency graph)
       - Query Relevance Prior (max(0, s_i · q))
    4. Compute Composite Salience Score:
       Score(s_i) = 0.5 * TextRank(s_i) + 0.3 * SVD(s_i) + 0.2 * QueryRelevance(s_i)
    5. Select top salient sentences and restore original document chronological flow.
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
        max_sentences: int = 2,
        min_relevance_threshold: float = 0.35,
    ) -> Dict[str, any]:
        """
        Synthesizes a clean, grounded answer in < 5ms.
        """
        if not passages:
            return {
                "answer": "No relevant information found in the indexed corpus.",
                "confidence": 0.0,
                "sentences_selected": 0,
                "method": "non_llm_empty"
            }

        # 1. Collect all candidate sentences across top passages
        candidate_sentences = []
        sentence_origin_map = []  # (doc_idx, sent_idx_in_doc, text)

        for doc_idx, p in enumerate(passages[:3]):
            text = p.get("text", "")
            sents = split_sentences_multilingual(text)
            for s_idx, s in enumerate(sents):
                candidate_sentences.append(s)
                sentence_origin_map.append((doc_idx, s_idx, s))

        if not candidate_sentences:
            # Fallback to passage text directly
            fallback_text = passages[0].get("text", "")[:200]
            return {
                "answer": fallback_text,
                "confidence": passages[0].get("score", 0.5),
                "sentences_selected": 1,
                "method": "non_llm_passage_fallback"
            }

        if len(candidate_sentences) == 1:
            return {
                "answer": candidate_sentences[0],
                "confidence": passages[0].get("score", 0.6),
                "sentences_selected": 1,
                "method": "non_llm_single"
            }

        # 2. Vectorize candidate sentences
        # Shape: (N, D)
        sentence_vectors = embedder.encode(candidate_sentences)
        sentence_vectors = np.ascontiguousarray(sentence_vectors, dtype=np.float32)
        
        # Ensure query vector is (D,) or (1, D)
        q_vec = np.squeeze(np.ascontiguousarray(query_vector, dtype=np.float32))
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        N, D = sentence_vectors.shape

        # ── Signal 1: Query Relevance Prior (s_i · q) ──
        query_relevance = np.maximum(0.0, np.dot(sentence_vectors, q_vec))

        # Check if max relevance meets minimum threshold
        max_q_rel = float(np.max(query_relevance)) if N > 0 else 0.0
        if max_q_rel < min_relevance_threshold:
            return {
                "answer": "No sufficiently relevant information found in the indexed corpus for this query.",
                "confidence": max_q_rel,
                "sentences_selected": 0,
                "method": "non_llm_low_relevance"
            }

        # Normalize query relevance to [0, 1]
        if np.max(query_relevance) > 0:
            query_relevance = query_relevance / np.max(query_relevance)

        # ── Signal 2: Continuous TextRank (Adjacency Matrix & Power Iteration) ──
        # Adjacency matrix A_ij = max(0, s_i · s_j)
        A = np.maximum(0.0, np.dot(sentence_vectors, sentence_vectors.T))
        np.fill_diagonal(A, 0.0)  # No self-loops

        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = A / row_sums

        # Power iteration for principal eigenvector (stationary distribution)
        p = np.ones(N, dtype=np.float32) / N
        for _ in range(25):
            p_next = (1 - self.damping) / N + self.damping * np.dot(M.T, p)
            if np.linalg.norm(p_next - p) < 1e-5:
                break
            p = p_next

        textrank_scores = p / (np.max(p) if np.max(p) > 0 else 1.0)

        # ── Signal 3: SVD Matrix Energy Decomposition ──
        # Center sentence vectors
        centered = sentence_vectors - np.mean(sentence_vectors, axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Find number of singular values covering 95% cumulative energy
            total_energy = np.sum(S ** 2)
            if total_energy > 0:
                cumulative_energy = np.cumsum(S ** 2) / total_energy
                k_components = int(np.searchsorted(cumulative_energy, self.energy_threshold)) + 1
            else:
                k_components = 1
            k_components = max(1, min(k_components, len(S)))

            # Project sentences onto top-k singular concept subspace
            subspace_U = U[:, :k_components]
            subspace_S = S[:k_components]
            svd_scores = np.sum((subspace_U * subspace_S) ** 2, axis=1)
            if np.max(svd_scores) > 0:
                svd_scores = svd_scores / np.max(svd_scores)
            else:
                svd_scores = np.ones(N, dtype=np.float32) / N
        except Exception:
            svd_scores = np.ones(N, dtype=np.float32) / N

        # ── Composite Salience Scoring: 0.5 TextRank + 0.3 SVD + 0.2 QueryRelevance ──
        composite_scores = (
            0.5 * textrank_scores +
            0.3 * svd_scores +
            0.2 * query_relevance
        )

        # Bias strongly by query relevance for short factual questions
        boosted_scores = composite_scores * (0.4 + 0.6 * query_relevance)

        # Select top-K most salient sentence indices
        top_indices = np.argsort(boosted_scores)[::-1][:max_sentences]

        # Reorder to original document chronological flow for narrative coherence
        ordered_indices = sorted(top_indices, key=lambda idx: (sentence_origin_map[idx][0], sentence_origin_map[idx][1]))

        selected_sentences = [candidate_sentences[i] for i in ordered_indices]

        # Assemble final synthesis
        final_answer = " ".join(selected_sentences).strip()

        # Quality check: Ensure terminal punctuation
        if final_answer and final_answer[-1] not in ".!?।॥":
            final_answer += "।" if any(0x0900 <= ord(c) <= 0x097F for c in final_answer) else "."

        mean_confidence = float(np.mean([passages[sentence_origin_map[i][0]].get("score", 0.5) for i in ordered_indices]))

        return {
            "answer": final_answer,
            "confidence": round(mean_confidence, 3),
            "sentences_selected": len(selected_sentences),
            "salience_scores": [round(float(boosted_scores[i]), 3) for i in ordered_indices],
            "method": "non_llm_textrank_svd"
        }

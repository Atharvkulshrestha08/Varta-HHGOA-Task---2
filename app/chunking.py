"""
Multi-Strategy Chunking Engine

Implements 4 distinct chunking strategies to demonstrate thoughtful
text splitting for RAG retrieval:

1. Semantic Sentence Chunking - splits on sentence boundaries
2. Fixed-Size Sliding Window - uniform token-based chunks with overlap
3. Paragraph-Aware Chunking - respects document structure
4. Metadata-Enriched Chunking - attaches source metadata to each chunk

The ChunkEngine selects the best strategy per passage and always
applies metadata enrichment on top.
"""

import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    chunk_id: str = ""
    strategy: str = ""
    language: str = ""
    source_passage_index: int = -1
    query_type: str = ""
    is_selected: bool = False
    overlap_with_prev: bool = False
    char_count: int = 0
    token_estimate: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]
        self.char_count = len(self.text)
        # Rough token estimate: ~4 chars per token for English, ~2 for Indic
        self.token_estimate = max(1, self.char_count // 3)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Sentence boundary patterns for different scripts ──────────────
SENTENCE_ENDINGS = {
    "default": re.compile(r'(?<=[.!?])\s+'),
    "devanagari": re.compile(r'(?<=[।!?\.])\s+'),     # Hindi
    "bengali": re.compile(r'(?<=[।!?\.])\s+'),          # Bengali
    "tamil": re.compile(r'(?<=[.!?\.])\s+'),             # Tamil
    "telugu": re.compile(r'(?<=[.!?\.])\s+'),            # Telugu
}

def _detect_script(text: str) -> str:
    """Detect the primary script of the text."""
    sample = text[:200]
    devanagari = len(re.findall(r'[\u0900-\u097F]', sample))
    bengali = len(re.findall(r'[\u0980-\u09FF]', sample))
    tamil = len(re.findall(r'[\u0B80-\u0BFF]', sample))
    telugu = len(re.findall(r'[\u0C00-\u0C7F]', sample))

    counts = {
        "devanagari": devanagari,
        "bengali": bengali,
        "tamil": tamil,
        "telugu": telugu,
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 5 else "default"


# ═══════════════════════════════════════════════════════════════════
# Strategy 1: Semantic Sentence Chunking
# ═══════════════════════════════════════════════════════════════════

def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
    language: str = "default"
) -> list[str]:
    """
    Split text on sentence boundaries, group N sentences per chunk
    with M-sentence overlap between consecutive chunks.

    WHY: Preserves meaning boundaries. A sentence is the smallest
    unit of complete thought — splitting mid-sentence loses context.
    Overlap ensures retrieval doesn't miss answers that span chunk edges.
    """
    script = _detect_script(text) if language == "default" else language
    pattern = SENTENCE_ENDINGS.get(script, SENTENCE_ENDINGS["default"])

    sentences = [s.strip() for s in pattern.split(text) if s.strip()]

    if len(sentences) <= sentences_per_chunk:
        return [text.strip()]

    chunks = []
    i = 0
    while i < len(sentences):
        end = min(i + sentences_per_chunk, len(sentences))
        chunk_text = " ".join(sentences[i:end])
        chunks.append(chunk_text)

        # Move forward by (sentences_per_chunk - overlap)
        i += max(1, sentences_per_chunk - overlap_sentences)

    return chunks


# ═══════════════════════════════════════════════════════════════════
# Strategy 2: Fixed-Size Sliding Window
# ═══════════════════════════════════════════════════════════════════

def chunk_by_fixed_size(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64
) -> list[str]:
    """
    Split text into fixed-size character chunks with sliding window overlap.

    WHY: Guarantees uniform chunk sizes for consistent embedding quality.
    The embedding model performs best on inputs of similar length.
    Overlap prevents hard cuts that lose meaning at boundaries.

    Note: We use character-based splitting (not tokens) because it's
    10x faster and the embedding model's tokenizer handles the conversion.
    """
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a word boundary (space) if possible
        if end < len(text):
            # Look back up to 30 chars for a space
            space_pos = text.rfind(' ', end - 30, end)
            if space_pos > start:
                end = space_pos

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = end - overlap
        if start <= 0 and end >= len(text):
            break

    return chunks


# ═══════════════════════════════════════════════════════════════════
# Strategy 3: Paragraph-Aware Chunking
# ═══════════════════════════════════════════════════════════════════

def chunk_by_paragraphs(
    text: str,
    max_chunk_chars: int = 512,
    fallback_chunk_size: int = 256,
    fallback_overlap: int = 64
) -> list[str]:
    """
    Split on paragraph boundaries (double newlines), then sub-split
    oversized paragraphs using fixed-size strategy.

    WHY: Respects document structure. Paragraphs are authored units
    of thought — splitting within them is a last resort. This strategy
    produces the most semantically coherent chunks for well-structured text.
    """
    # Split on paragraph breaks
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If paragraph itself is too long, sub-split it
        if len(para) > max_chunk_chars:
            # Flush current accumulator first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Sub-split the long paragraph
            sub_chunks = chunk_by_fixed_size(
                para, fallback_chunk_size, fallback_overlap
            )
            chunks.extend(sub_chunks)
        elif len(current_chunk) + len(para) + 1 > max_chunk_chars:
            # Adding this para would exceed limit — flush
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Accumulate
            current_chunk = f"{current_chunk}\n{para}" if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ═══════════════════════════════════════════════════════════════════
# Strategy 4: Metadata-Enriched Chunking (decorator-style)
# ═══════════════════════════════════════════════════════════════════

def enrich_chunks_with_metadata(
    raw_chunks: list[str],
    strategy_name: str,
    language: str = "",
    passage_index: int = -1,
    query_type: str = "",
    is_selected: bool = False,
) -> list[Chunk]:
    """
    Wraps raw text chunks into Chunk objects with full metadata.

    WHY: Metadata enables filtering at retrieval time. For example,
    we can boost chunks from passages that were human-selected as
    relevant in the original MSMARCO dataset. Language metadata
    enables language-specific retrieval.
    """
    enriched = []
    for i, text in enumerate(raw_chunks):
        chunk = Chunk(
            text=text,
            strategy=strategy_name,
            language=language,
            source_passage_index=passage_index,
            query_type=query_type,
            is_selected=is_selected,
            overlap_with_prev=(i > 0),
        )
        enriched.append(chunk)
    return enriched


# ═══════════════════════════════════════════════════════════════════
# ChunkEngine: Selects best strategy per passage
# ═══════════════════════════════════════════════════════════════════

class ChunkEngine:
    """
    Intelligent chunking engine that selects the best strategy
    per passage based on its structure and characteristics.

    Decision logic:
    1. Try Paragraph-Aware first (if text has paragraph breaks)
    2. If single paragraph → use Semantic Sentence chunking
    3. Fixed-Size as fallback for very long or structureless text
    4. Always apply Metadata-Enrichment on top
    """

    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1,
        fixed_chunk_size: int = 256,
        fixed_overlap: int = 64,
        paragraph_max_chars: int = 512,
    ):
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.fixed_chunk_size = fixed_chunk_size
        self.fixed_overlap = fixed_overlap
        self.paragraph_max_chars = paragraph_max_chars

    def chunk_passage(
        self,
        text: str,
        language: str = "",
        passage_index: int = -1,
        query_type: str = "",
        is_selected: bool = False,
    ) -> list[Chunk]:
        """
        Chunk a single passage using the best strategy, then
        enrich with metadata.
        """
        text = text.strip()
        if not text:
            return []

        # Decide strategy based on text structure
        has_paragraphs = bool(re.search(r'\n\s*\n', text))
        is_very_long = len(text) > 1000

        if has_paragraphs:
            # Strategy 3: Paragraph-Aware
            raw_chunks = chunk_by_paragraphs(
                text,
                max_chunk_chars=self.paragraph_max_chars,
                fallback_chunk_size=self.fixed_chunk_size,
                fallback_overlap=self.fixed_overlap,
            )
            strategy = "paragraph_aware"
        elif is_very_long:
            # Strategy 2: Fixed-Size for very long single-paragraph text
            raw_chunks = chunk_by_fixed_size(
                text,
                chunk_size=self.fixed_chunk_size,
                overlap=self.fixed_overlap,
            )
            strategy = "fixed_size_sliding_window"
        else:
            # Strategy 1: Semantic Sentence
            raw_chunks = chunk_by_sentences(
                text,
                sentences_per_chunk=self.sentences_per_chunk,
                overlap_sentences=self.sentence_overlap,
                language=language,
            )
            strategy = "semantic_sentence"

        # Strategy 4: Always enrich with metadata
        enriched = enrich_chunks_with_metadata(
            raw_chunks,
            strategy_name=strategy,
            language=language,
            passage_index=passage_index,
            query_type=query_type,
            is_selected=is_selected,
        )

        return enriched

    def chunk_passages(
        self,
        passages: list[dict],
    ) -> list[Chunk]:
        """
        Chunk multiple passages. Each passage dict should have:
        {
            "text": str,
            "language": str,
            "index": int,
            "query_type": str,
            "is_selected": bool
        }
        """
        all_chunks = []
        for p in passages:
            chunks = self.chunk_passage(
                text=p["text"],
                language=p.get("language", ""),
                passage_index=p.get("index", -1),
                query_type=p.get("query_type", ""),
                is_selected=p.get("is_selected", False),
            )
            all_chunks.extend(chunks)
        return all_chunks

    def get_strategy_stats(self, chunks: list[Chunk]) -> dict:
        """Return stats about which strategies were used."""
        stats = {}
        for c in chunks:
            stats[c.strategy] = stats.get(c.strategy, 0) + 1
        return {
            "total_chunks": len(chunks),
            "by_strategy": stats,
            "avg_char_count": (
                sum(c.char_count for c in chunks) / len(chunks)
                if chunks else 0
            ),
            "avg_token_estimate": (
                sum(c.token_estimate for c in chunks) / len(chunks)
                if chunks else 0
            ),
        }

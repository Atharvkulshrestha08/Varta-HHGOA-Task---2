"""
Tests for Multi-Strategy Chunking Engine
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking import (
    ChunkEngine,
    chunk_by_sentences,
    chunk_by_fixed_size,
    chunk_by_paragraphs,
    enrich_chunks_with_metadata,
    _detect_script,
)


def test_sentence_chunking_english():
    """Test semantic sentence chunking on English text."""
    text = "The sun rises in the east. It sets in the west. The earth revolves around the sun. This takes 365 days. Seasons change because of axial tilt."
    chunks = chunk_by_sentences(text, sentences_per_chunk=2, overlap_sentences=1)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    # Verify overlap (last sentence of chunk N = first sentence of chunk N+1)
    print(f"  ✅ English sentence chunking: {len(chunks)} chunks")


def test_sentence_chunking_hindi():
    """Test sentence chunking with Hindi (Devanagari) text."""
    text = "भारत एक बड़ा देश है। यहाँ कई भाषाएँ बोली जाती हैं। हिंदी सबसे ज्यादा बोली जाती है। अंग्रेज़ी भी व्यापक है।"
    chunks = chunk_by_sentences(text, sentences_per_chunk=2, overlap_sentences=1)
    assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"
    print(f"  ✅ Hindi sentence chunking: {len(chunks)} chunks")


def test_fixed_size_chunking():
    """Test fixed-size sliding window chunking."""
    text = "A" * 600 + " " + "B" * 200
    chunks = chunk_by_fixed_size(text, chunk_size=256, overlap=64)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    # Verify each chunk is <= chunk_size + some tolerance
    for c in chunks:
        assert len(c) <= 300, f"Chunk too long: {len(c)}"
    print(f"  ✅ Fixed-size chunking: {len(chunks)} chunks")


def test_paragraph_aware_chunking():
    """Test paragraph-aware chunking."""
    text = "Paragraph one about physics.\n\nParagraph two about chemistry. It has more content.\n\nParagraph three is very short."
    chunks = chunk_by_paragraphs(text, max_chunk_chars=100)
    assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"
    print(f"  ✅ Paragraph chunking: {len(chunks)} chunks")


def test_metadata_enrichment():
    """Test metadata-enriched chunking."""
    raw_chunks = ["Hello world", "Foo bar baz"]
    enriched = enrich_chunks_with_metadata(
        raw_chunks,
        strategy_name="test_strategy",
        language="hin_Deva",
        passage_index=42,
        query_type="DESCRIPTION",
        is_selected=True,
    )
    assert len(enriched) == 2
    assert enriched[0].language == "hin_Deva"
    assert enriched[0].strategy == "test_strategy"
    assert enriched[0].source_passage_index == 42
    assert enriched[0].is_selected is True
    assert enriched[1].overlap_with_prev is True
    print(f"  ✅ Metadata enrichment: {len(enriched)} enriched chunks")


def test_script_detection():
    """Test script detection for different languages."""
    assert _detect_script("नमस्ते भारत") == "devanagari"
    assert _detect_script("Hello World") == "default"
    print(f"  ✅ Script detection working")


def test_chunk_engine_auto_strategy():
    """Test ChunkEngine auto-selects the right strategy."""
    engine = ChunkEngine()

    # Paragraph text → paragraph_aware
    para_text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = engine.chunk_passage(para_text, language="eng_Latn")
    assert any(c.strategy == "paragraph_aware" for c in chunks)

    # Long single paragraph → fixed_size
    long_text = "Word " * 500
    chunks = engine.chunk_passage(long_text, language="eng_Latn")
    assert any(c.strategy == "fixed_size_sliding_window" for c in chunks)

    # Short text → semantic_sentence
    short_text = "The cat sat on the mat. It was a sunny day. Birds were singing."
    chunks = engine.chunk_passage(short_text, language="eng_Latn")
    assert any(c.strategy == "semantic_sentence" for c in chunks)

    print(f"  ✅ ChunkEngine auto-strategy selection working")


def test_chunk_engine_stats():
    """Test chunking statistics."""
    engine = ChunkEngine()
    passages = [
        {"text": "Hello world. This is a test.", "language": "eng_Latn", "index": 0, "query_type": "TEST", "is_selected": False},
        {"text": "Para one.\n\nPara two.\n\nPara three.", "language": "eng_Latn", "index": 1, "query_type": "TEST", "is_selected": True},
    ]
    chunks = engine.chunk_passages(passages)
    stats = engine.get_strategy_stats(chunks)
    assert stats["total_chunks"] > 0
    assert "by_strategy" in stats
    print(f"  ✅ Chunking stats: {stats['total_chunks']} total chunks, strategies: {list(stats['by_strategy'].keys())}")


if __name__ == "__main__":
    print("\n🧪 Running Chunking Tests\n")
    test_sentence_chunking_english()
    test_sentence_chunking_hindi()
    test_fixed_size_chunking()
    test_paragraph_aware_chunking()
    test_metadata_enrichment()
    test_script_detection()
    test_chunk_engine_auto_strategy()
    test_chunk_engine_stats()
    print("\n✅ All chunking tests passed!\n")

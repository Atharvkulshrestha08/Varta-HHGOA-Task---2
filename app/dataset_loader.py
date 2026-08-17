"""
Dataset Loader for MSMARCO-XI

Loads the ai4bharat/MSMARCO-XI dataset from HuggingFace for
4 Indic languages: Hindi, Bengali, Tamil, Telugu.

Each record contains:
- query: The search query (in target language)
- Eng_Query: English version of the query
- passages: { English_passages: [...], Translated_passages: [...], is_selected: [...] }
- Answer: Answer in target language
- Eng_Answer: English answer
- query_type: Type of query (e.g., "DESCRIPTION", "NUMERIC", etc.)
- source_lang / target_lang: Language codes
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Language code mapping for the 14 official MSMARCO-XI dataset languages
LANGUAGE_MAP = {
    "hin_Deva": {"name": "Hindi", "hf_prefix": "hin", "filter_code": "hin_Deva"},
    "ben_Beng": {"name": "Bengali", "hf_prefix": "ben", "filter_code": "ben_Beng"},
    "tam_Taml": {"name": "Tamil", "hf_prefix": "tam", "filter_code": "tam_Taml"},
    "tel_Telu": {"name": "Telugu", "hf_prefix": "tel", "filter_code": "tel_Telu"},
    "mar_Deva": {"name": "Marathi", "hf_prefix": "mar", "filter_code": "mar_Deva"},
    "guj_Gujr": {"name": "Gujarati", "hf_prefix": "guj", "filter_code": "guj_Gujr"},
    "kan_Knda": {"name": "Kannada", "hf_prefix": "kan", "filter_code": "kan_Knda"},
    "mal_Mlym": {"name": "Malayalam", "hf_prefix": "mal", "filter_code": "mal_Mlym"},
    "pan_Guru": {"name": "Punjabi", "hf_prefix": "pan", "filter_code": "pan_Guru"},
    "ori_Orya": {"name": "Odia", "hf_prefix": "ori", "filter_code": "ori_Orya"},
    "asm_Beng": {"name": "Assamese", "hf_prefix": "asm", "filter_code": "asm_Beng"},
    "urd_Arab": {"name": "Urdu", "hf_prefix": "urd", "filter_code": "urd_Arab"},
    "san_Deva": {"name": "Sanskrit", "hf_prefix": "san", "filter_code": "san_Deva"},
    "nep_Deva": {"name": "Nepali", "hf_prefix": "nep", "filter_code": "nep_Deva"},
}


def load_dataset_from_huggingface(
    languages: list[str] = None,
    max_per_language: int = 5000,
    split: str = "train",
) -> list[dict]:
    """
    Load MSMARCO-XI passages from HuggingFace.

    Returns a flat list of passage dicts:
    [
        {
            "text": "passage text in target language",
            "language": "hin_Deva",
            "index": 0,
            "query_type": "DESCRIPTION",
            "is_selected": True,
            "source_query": "original query",
            "english_text": "English version of passage"
        },
        ...
    ]
    """
    from datasets import load_dataset

    if languages is None:
        languages = list(LANGUAGE_MAP.keys())

    all_passages = []
    global_index = 0

    for lang_code in languages:
        if lang_code not in LANGUAGE_MAP:
            logger.warning(f"Unknown language code: {lang_code}, skipping")
            continue

        lang_info = LANGUAGE_MAP[lang_code]
        logger.info(f"Loading {lang_info['name']} ({lang_code}) from MSMARCO-XI...")

        try:
            # Load the dataset using direct parquet URL to bypass config resolution bugs
            parquet_url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/{lang_code[:3]}train.parquet"
            logger.info(f"Downloading direct parquet from: {parquet_url}")
            
            ds = load_dataset(
                "parquet",
                data_files={"train": parquet_url},
                split=split,
                streaming=False,
            )

            count = 0
            for row in ds:
                # Target language is already filtered by the file, but we keep this just in case
                if row.get("target_lang") != lang_code:
                    continue

                if count >= max_per_language:
                    break

                passages_data = row.get("passages", {})
                translated_passages = passages_data.get("Translated_passages", [])
                english_passages = passages_data.get("English_passages", [])
                is_selected_list = passages_data.get("is_selected", [])

                # Extract each passage from this query's passage set
                for p_idx, passage_text in enumerate(translated_passages):
                    if not passage_text or not passage_text.strip():
                        continue

                    is_selected = (
                        bool(is_selected_list[p_idx])
                        if p_idx < len(is_selected_list)
                        else False
                    )
                    english_text = (
                        english_passages[p_idx]
                        if p_idx < len(english_passages)
                        else ""
                    )

                    all_passages.append({
                        "text": passage_text.strip(),
                        "language": lang_code,
                        "index": global_index,
                        "query_type": row.get("query_type", ""),
                        "is_selected": is_selected,
                        "source_query": row.get("query", ""),
                        "english_text": english_text,
                    })
                    global_index += 1

                count += 1

            logger.info(
                f"  Loaded {count} queries → "
                f"{global_index - (global_index - len([p for p in all_passages if p['language'] == lang_code]))} passages "
                f"for {lang_info['name']}"
            )

        except Exception as e:
            logger.error(f"Failed to load {lang_info['name']}: {e}")
            continue

    logger.info(f"Total passages loaded: {len(all_passages)}")
    return all_passages


def save_passages_to_disk(passages: list[dict], output_path: str):
    """Save loaded passages to JSON for reuse."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(passages)} passages to {output_path}")


def load_passages_from_disk(input_path: str) -> Optional[list[dict]]:
    """Load previously saved passages from JSON."""
    path = Path(input_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        passages = json.load(f)
    logger.info(f"Loaded {len(passages)} passages from {input_path}")
    return passages

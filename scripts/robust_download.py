import os
import json
import logging
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

LANGUAGES = {
    "hin_Deva": "hin",
    "ben_Beng": "ben",
    "tam_Taml": "tam",
    "tel_Telu": "tel"
}

MAX_PER_LANGUAGE = 5000
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_JSON = DATA_DIR / "raw_passages.json"

def download_file(url, local_path):
    logger.info(f"Downloading {url} to {local_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    logger.info(f"Downloaded {local_path}")

def build():
    all_passages = []
    global_index = 0
    
    for lang_full, lang_short in LANGUAGES.items():
        parquet_file = DATA_DIR / f"{lang_short}train.parquet"
        url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/{lang_short}train.parquet"
        
        if not parquet_file.exists():
            try:
                download_file(url, parquet_file)
            except Exception as e:
                logger.error(f"Failed to download {lang_short}: {e}")
                continue
                
        logger.info(f"Parsing {parquet_file} with pandas...")
        try:
            df = pd.read_parquet(parquet_file)
            count = 0
            for idx, row in df.iterrows():
                if count >= MAX_PER_LANGUAGE:
                    break
                    
                if row.get("target_lang") != lang_full:
                    continue
                    
                passages_data = row.get("passages", {})
                if not isinstance(passages_data, dict):
                    continue
                    
                translated = passages_data.get("Translated_passages", [])
                english = passages_data.get("English_passages", [])
                selected = passages_data.get("is_selected", [])
                
                for p_idx, text in enumerate(translated):
                    if not text or not str(text).strip():
                        continue
                        
                    is_sel = bool(selected[p_idx]) if p_idx < len(selected) else False
                    eng_text = str(english[p_idx]) if p_idx < len(english) else ""
                    
                    all_passages.append({
                        "text": str(text).strip(),
                        "language": lang_full,
                        "index": global_index,
                        "query_type": str(row.get("query_type", "")),
                        "is_selected": is_sel,
                        "source_query": str(row.get("query", "")),
                        "english_text": eng_text
                    })
                    global_index += 1
                count += 1
                
            logger.info(f"Loaded {count} queries for {lang_full}")
        except Exception as e:
            logger.error(f"Error parsing {parquet_file}: {e}")
            
    with open(RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_passages, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_passages)} passages to {RAW_JSON}")

if __name__ == '__main__':
    build()

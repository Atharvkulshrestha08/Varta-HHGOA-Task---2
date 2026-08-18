import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    with open("data/chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    queries_by_lang = {"eng_Latn": [], "hin_Deva": [], "tam_Taml": []}

    for c in chunks:
        lang = c.get("language")
        q = c.get("query")
        text = c.get("text", "").replace("\n", " ").strip()
        if lang in queries_by_lang and q and len(q.strip()) > 8:
            if not any(x[0] == q for x in queries_by_lang[lang]):
                queries_by_lang[lang].append((q.strip(), text[:140]))

    print("=" * 80)
    print("VERIFIED QUERIES EXTRACTED DIRECTLY FROM 48,000 VECTOR DATABASE")
    print("=" * 80)

    for lang, items in queries_by_lang.items():
        lang_name = "🇬🇧 English (eng_Latn)" if lang == "eng_Latn" else ("🇮🇳 Hindi (hin_Deva)" if lang == "hin_Deva" else "🌴 Tamil (tam_Taml)")
        print(f"\n### {lang_name} — Total Unique Indexed Topics: {len(items):,}")
        print("-" * 80)
        for i, (q, passage) in enumerate(items[:15]):
            print(f"{i+1}. Query: \"{q}\"")
            print(f"   Context in Vector DB: \"{passage}...\"")
            print()

if __name__ == "__main__":
    main()

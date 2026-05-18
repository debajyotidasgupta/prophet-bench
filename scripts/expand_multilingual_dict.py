#!/usr/bin/env python3
"""EMNLP-6: Expand multilingual_dict.json from 6 to 10 languages.

Adds Arabic (ar), Chinese Mandarin (zh), Russian (ru), Korean (ko)
using romanized transliterations (consistent with the existing Hindi /
Japanese convention in the dict). Native-script aliases are added to
language_aliases so the verifier accepts both ``arabic`` and
``العربية`` for the language-identification task.

This script is idempotent: running it twice is a no-op.

Run once:
  python scripts/expand_multilingual_dict.py
Verify with a smoke generation:
  python -c "from prophet.families.multilingual_family import MultilingualFamily; \\
             print({t.metadata.get('lang_code', t.metadata.get('lang')) \\
                    for t in MultilingualFamily().generate(n=200, seed=42)})"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Romanized translations for the 4 new languages.
# Convention follows the existing dict: lowercase ASCII, no diacritics.
# Verified against standard romanization tables (Hepburn for jp, Pinyin for
# zh, common Cyrillic-to-Latin for ru, Hangul-to-Latin for ko).
# ---------------------------------------------------------------------------

NEW_LANGS = ["ar", "zh", "ru", "ko"]
LANGUAGE_NAMES = {
    "ar": "arabic",
    "zh": "chinese",
    "ru": "russian",
    "ko": "korean",
}
LANGUAGE_ALIASES_NEW = {
    "arabic": ["arabic", "ar", "العربية", "arabe"],
    "chinese": ["chinese", "zh", "mandarin", "中文", "汉语"],
    "russian": ["russian", "ru", "русский", "russe"],
    "korean": ["korean", "ko", "한국어", "coreen"],
}

# 33 phrases × 4 new languages. Each entry must already exist in the
# `translations` list keyed by 'en'; we just augment it with the new lang codes.
TRANSLATIONS_NEW: dict[str, dict[str, str]] = {
    "thank you": {"ar": "shukran", "zh": "xiexie", "ru": "spasibo", "ko": "gomawo"},
    "hello":     {"ar": "marhaba", "zh": "nihao",  "ru": "privet",  "ko": "annyeong"},
    "goodbye":   {"ar": "maasalama", "zh": "zaijian", "ru": "dosvidaniya", "ko": "annyeonghi"},
    "yes":       {"ar": "naam",  "zh": "shide",  "ru": "da",      "ko": "ne"},
    "no":        {"ar": "la",    "zh": "bushi",  "ru": "net",     "ko": "aniyo"},
    "water":     {"ar": "ma",    "zh": "shui",   "ru": "voda",    "ko": "mul"},
    "bread":     {"ar": "khubz", "zh": "mianbao","ru": "khleb",   "ko": "bbang"},
    "house":     {"ar": "bayt",  "zh": "fangzi", "ru": "dom",     "ko": "jib"},
    "friend":    {"ar": "sadiq", "zh": "pengyou","ru": "drug",    "ko": "chingu"},
    "book":      {"ar": "kitab", "zh": "shu",    "ru": "kniga",   "ko": "chaek"},
    "school":    {"ar": "madrasa","zh": "xuexiao","ru": "shkola", "ko": "hakgyo"},
    "cat":       {"ar": "qit",   "zh": "mao",    "ru": "koshka",  "ko": "goyangi"},
    "dog":       {"ar": "kalb",  "zh": "gou",    "ru": "sobaka",  "ko": "gae"},
    "sun":       {"ar": "shams", "zh": "taiyang","ru": "solntse", "ko": "haetbit"},
    "moon":      {"ar": "qamar", "zh": "yueliang","ru": "luna",   "ko": "dal"},
    "tree":      {"ar": "shajara","zh": "shu",   "ru": "derevo",  "ko": "namu"},
    "fire":      {"ar": "nar",   "zh": "huo",    "ru": "ogon",    "ko": "bul"},
    "river":     {"ar": "nahr",  "zh": "he",     "ru": "reka",    "ko": "gang"},
    "mountain":  {"ar": "jabal", "zh": "shan",   "ru": "gora",    "ko": "san"},
    "country":   {"ar": "balad", "zh": "guojia", "ru": "strana",  "ko": "nara"},
    "city":      {"ar": "madina","zh": "chengshi","ru": "gorod",  "ko": "dosi"},
    "love":      {"ar": "hubb",  "zh": "ai",     "ru": "lyubov",  "ko": "sarang"},
    "language":  {"ar": "lugha", "zh": "yuyan",  "ru": "yazyk",   "ko": "eoneo"},
    "tea":       {"ar": "shay",  "zh": "cha",    "ru": "chay",    "ko": "cha"},
    "rice":      {"ar": "aruz",  "zh": "mifan",  "ru": "ris",     "ko": "bap"},
    "fish":      {"ar": "samaka","zh": "yu",     "ru": "ryba",    "ko": "saengseon"},
    "bird":      {"ar": "tair",  "zh": "niao",   "ru": "ptitsa",  "ko": "sae"},
    "flower":    {"ar": "zahra", "zh": "hua",    "ru": "tsvetok", "ko": "ggot"},
    "horse":     {"ar": "hisan", "zh": "ma",     "ru": "loshad",  "ko": "mal"},
    "good":      {"ar": "jayyid","zh": "hao",    "ru": "khoroshiy","ko": "joha"},
    "bad":       {"ar": "sayyi", "zh": "huai",   "ru": "plokhoy", "ko": "nappeun"},
    "happy":     {"ar": "saeed", "zh": "kaixin", "ru": "schastlivy","ko": "haengbok"},
    "sad":       {"ar": "hazin", "zh": "shangxin","ru": "grustny", "ko": "seulpeun"},
}

NUMBERS_NEW: dict[int, dict[str, str]] = {
    0:  {"ar": "sifr",      "zh": "ling",   "ru": "nol",     "ko": "yeong"},
    1:  {"ar": "wahid",     "zh": "yi",     "ru": "odin",    "ko": "il"},
    2:  {"ar": "ithnan",    "zh": "er",     "ru": "dva",     "ko": "i"},
    3:  {"ar": "thalatha",  "zh": "san",    "ru": "tri",     "ko": "sam"},
    4:  {"ar": "arbaa",     "zh": "si",     "ru": "chetyre", "ko": "sa"},
    5:  {"ar": "khamsa",    "zh": "wu",     "ru": "pyat",    "ko": "o"},
    6:  {"ar": "sitta",     "zh": "liu",    "ru": "shest",   "ko": "yuk"},
    7:  {"ar": "sabaa",     "zh": "qi",     "ru": "sem",     "ko": "chil"},
    8:  {"ar": "thamaniya", "zh": "ba",     "ru": "vosem",   "ko": "pal"},
    9:  {"ar": "tisaa",     "zh": "jiu",    "ru": "devyat",  "ko": "gu"},
    10: {"ar": "ashara",    "zh": "shi",    "ru": "desyat",  "ko": "sip"},
}

PLUS_WORDS_NEW = {"ar": "zaid",    "zh": "jia",  "ru": "plyus", "ko": "deohagi"}

# Cognate-style groups: a shared concept word, romanized in each lang.
COGNATE_GROUPS_NEW = [
    {"root": "computer", "members": [
        {"word": "kombyuter", "lang": "ar"},  # كومبيوتر
        {"word": "diannao",   "lang": "zh"},  # 电脑
        {"word": "kompyuter", "lang": "ru"},
        {"word": "keompyuteo","lang": "ko"},
    ], "distractors": ["bayt", "shu", "dosi", "nehri", "gae", "namu"]},
    {"root": "internet", "members": [
        {"word": "intirnit", "lang": "ar"},
        {"word": "wangluo",  "lang": "zh"},
        {"word": "internet", "lang": "ru"},
        {"word": "inteoneat","lang": "ko"},
    ], "distractors": ["shukran", "ai", "kniga", "sae", "saeed", "haetbit"]},
    {"root": "coffee", "members": [
        {"word": "qahwa",   "lang": "ar"},
        {"word": "kafei",   "lang": "zh"},
        {"word": "kofe",    "lang": "ru"},
        {"word": "keopi",   "lang": "ko"},
    ], "distractors": ["mifan", "shams", "kalb", "gora", "joha", "yangchi"]},
]

ORDERING_SENTENCES_NEW = [
    # ar / zh / ru / ko ordering sentences using romanization; 5 tokens each.
    {"lang": "ar", "ordered": ["al", "qit", "yara", "al", "kalb"], "key": "aqyak"},  # The cat sees the dog (romanized)
    {"lang": "zh", "ordered": ["wo", "xihuan", "chi", "mi", "fan"], "key": "wxcmf"},  # I like to eat rice
    {"lang": "ru", "ordered": ["ya", "lyublyu", "khleb", "i", "syr"], "key": "ylkis"},  # I love bread and cheese
    {"lang": "ko", "ordered": ["nae", "chingu", "neun", "chaek", "ida"], "key": "ncnci"},  # My friend [is] a book
]

GRAMMAR_CHOICES_NEW = [
    {"correct": "anaa ureed al kitab.", "wrong": [
        "kitab ureed anaa al.", "al kitab anaa ureed.", "ureed al anaa kitab."
    ], "lang": "ar"},
    {"correct": "wo xihuan chi mifan.", "wrong": [
        "mifan chi xihuan wo.", "chi xihuan wo mifan.", "wo mifan chi xihuan."
    ], "lang": "zh"},
    {"correct": "ya chitayu knigu.", "wrong": [
        "knigu chitayu ya.", "chitayu ya knigu.", "ya knigu chitayu."
    ], "lang": "ru"},
    {"correct": "naneun chaegeul ilkneunda.", "wrong": [
        "chaegeul naneun ilkneunda.", "ilkneunda naneun chaegeul.", "naneun ilkneunda chaegeul."
    ], "lang": "ko"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", type=Path,
                    default=Path("data/seeds/multilingual_dict.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = json.loads(args.dict.read_text(encoding="utf-8"))

    # Idempotency check
    existing_langs = set(d.get("language_names", {}).keys())
    if existing_langs >= set(NEW_LANGS):
        print("All new languages already present. No-op.")
        return 0

    # 1. language_names
    for code in NEW_LANGS:
        d["language_names"][code] = LANGUAGE_NAMES[code]

    # 2. language_aliases
    for name, aliases in LANGUAGE_ALIASES_NEW.items():
        d.setdefault("language_aliases", {})[name] = aliases

    # 3. translations: augment in-place
    for entry in d["translations"]:
        en = entry.get("en")
        if en in TRANSLATIONS_NEW:
            for code in NEW_LANGS:
                entry[code] = TRANSLATIONS_NEW[en][code]

    # 4. numbers
    for entry in d["numbers"]:
        v = entry.get("value")
        if v in NUMBERS_NEW:
            for code in NEW_LANGS:
                entry[code] = NUMBERS_NEW[v][code]

    # 5. plus_words
    for code in NEW_LANGS:
        d.setdefault("plus_words", {})[code] = PLUS_WORDS_NEW[code]

    # 6. cognate_groups: append new groups
    d.setdefault("cognate_groups", []).extend(COGNATE_GROUPS_NEW)

    # 7. ordering_sentences
    d.setdefault("ordering_sentences", []).extend(ORDERING_SENTENCES_NEW)

    # 8. grammar_choices
    d.setdefault("grammar_choices", []).extend(GRAMMAR_CHOICES_NEW)

    # 9. _meta
    meta = d.setdefault("_meta", {})
    meta["languages"] = sorted(d["language_names"].keys())
    meta["version"] = "2.0"
    meta["description"] = (
        str(meta.get("description", ""))
        + " | v2: expanded to 10 languages (ar, zh, ru, ko added) for EMNLP coverage"
    ).strip(" |")

    if args.dry_run:
        print("DRY RUN — would have written to", args.dict)
        print(f"  language_names now: {sorted(d['language_names'].keys())}")
        print(f"  translations rows: {len(d['translations'])}")
        print(f"  cognate_groups: {len(d['cognate_groups'])}")
        return 0

    args.dict.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote expanded dict -> {args.dict}")
    print(f"  language_names: {sorted(d['language_names'].keys())} ({len(d['language_names'])} langs)")
    print(f"  translations rows: {len(d['translations'])}")
    print(f"  numbers rows: {len(d['numbers'])}")
    print(f"  cognate_groups: {len(d['cognate_groups'])}")
    print(f"  ordering_sentences: {len(d['ordering_sentences'])}")
    print(f"  grammar_choices: {len(d['grammar_choices'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

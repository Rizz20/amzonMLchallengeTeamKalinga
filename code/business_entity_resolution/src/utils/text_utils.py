"""
text_utils.py — Country-agnostic text normalization for Business Entity Resolution.

Handles: English (US/UK), Hindi (Devanagari), French, and any unseen language.
Design principle: never hard-code country logic; use parameterized/generic transforms.
"""

import re
import unicodedata
from typing import Optional, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Abbreviation & suffix dictionaries
# ──────────────────────────────────────────────────────────────────────────────

# Legal / corporate suffixes — these are STRIPPED when building "core name"
LEGAL_SUFFIXES = {
    # English
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "lp", "llp", "plc",
    "pvt", "private", "pte",
    "grp", "group", "holdings", "holding", "enterprises", "enterprise",
    "ventures", "venture", "solutions", "services", "systems",
    "international", "intl", "global", "worldwide",
    # French
    "sa", "sarl", "sas", "sasu", "sci", "eurl", "snc", "sca", "scs",
    "ei", "eirl",
    # India
    "pvtltd",
    # Generic
    "associates", "association", "foundation", "trust",
}

# Name token expansions applied BEFORE suffix stripping
NAME_TOKEN_EXPAND = {
    "&": "and",
    "intl": "international",
    "natl": "national",
    "dept": "department",
    "mgmt": "management",
    "svcs": "services",
    "svc": "service",
    "mfg": "manufacturing",
    "tech": "technology",
    "assoc": "associates",
    "assn": "association",
    "univ": "university",
    "hosp": "hospital",
    "med": "medical",
    "sys": "systems",
    "bldg": "building",
    "ctr": "center",
    "bros": "brothers",
    "hldgs": "holdings",
    "entrp": "enterprises",
    "pvt": "private",
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
}

# Address token expansions (country-agnostic; covers US, India, France common abbreviations)
ADDR_TOKEN_EXPAND = {
    # Street types
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ct": "court",
    "ln": "lane",
    "pl": "place",
    "hwy": "highway",
    "pkwy": "parkway",
    "fwy": "freeway",
    "expy": "expressway",
    "trl": "trail",
    "ter": "terrace",
    "cir": "circle",
    # Directions
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    # Unit types
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "rm": "room",
    "bldg": "building",
    "po": "post office",
    # French street types
    "rue": "rue",
    "bd": "boulevard",
    "av": "avenue",
    "pl": "place",
    "imp": "impasse",
    "all": "allee",
    "bis": "bis",
}

# Characters to strip from names/addresses
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_LEADING_NOISE_RE = re.compile(r"^[\-\*<>\.\s]+")   # leading symbols like "-- ", "<< "


# ──────────────────────────────────────────────────────────────────────────────
# Core normalization helpers
# ──────────────────────────────────────────────────────────────────────────────

def unicode_normalize(text: str) -> str:
    """NFC normalize unicode; handles mixed Devanagari/Latin/French diacritics."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def to_ascii_safe(text: str) -> str:
    """
    Transliterate accented Latin characters to ASCII (e.g., é → e, ñ → n).
    Non-Latin scripts (Hindi, etc.) are left unchanged (don't forcibly transliterate).
    Only converts characters with ASCII decomposition.
    """
    normalized = unicodedata.normalize("NFD", text)
    result = []
    for ch in normalized:
        cat = unicodedata.category(ch)
        # Keep letters (Ll, Lu, Lt, Lm, Lo) and marks from non-latin scripts
        # Drop combining accent marks that have a base ASCII char
        if cat.startswith("M") and unicodedata.category(unicodedata.normalize("NFC", ch)) == cat:
            # Combining mark on a non-ASCII base — keep it
            result.append(ch)
        elif cat.startswith("M"):
            # Combining mark on an ASCII base — drop it (accents on Latin letters)
            pass
        else:
            result.append(ch)
    return "".join(result)


def strip_leading_noise(text: str) -> str:
    """Remove leading symbols like '-- ', '<< ', '** ', etc."""
    return _LEADING_NOISE_RE.sub("", text).strip()


def tokenize(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split into tokens.
    Works for Latin and non-Latin scripts (Devanagari stays as words).
    """
    text = unicode_normalize(text)
    text = text.lower()
    text = strip_leading_noise(text)
    # Replace punctuation with space (keep alphanumeric + unicode letters + digits)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return [t for t in text.split() if t]


# ──────────────────────────────────────────────────────────────────────────────
# Business name normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_name(name: str, expand_abbrevs: bool = True) -> str:
    """
    Full normalized name: lowercase, expand abbreviations, strip punctuation.
    Used for TF-IDF and string similarity features.
    """
    if not name or not isinstance(name, str):
        return ""
    tokens = tokenize(name)
    if expand_abbrevs:
        tokens = [NAME_TOKEN_EXPAND.get(t, t) for t in tokens]
    return " ".join(tokens)


def core_name(name: str) -> str:
    """
    Core name: normalized name with legal suffixes stripped.
    E.g., "Starbucks Corp Inc Ltd" → "starbucks"
    Used for exact/near-exact matching.
    """
    tokens = normalize_name(name, expand_abbrevs=True).split()
    # Strip trailing legal suffixes (they appear at end or beginning)
    # Also strip from beginning (some records have "Pvt. ABC Ltd.")
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else " ".join(tokens)


def name_prefix(name: str, n: int = 4) -> str:
    """First n characters of normalized name (for blocking key)."""
    cn = core_name(name)
    return cn[:n] if cn else ""


def name_trigrams(name: str) -> List[str]:
    """Character trigrams of normalized name — used for inverted-index blocking."""
    s = core_name(name).replace(" ", "")
    if len(s) < 3:
        return [s] if s else []
    return [s[i:i+3] for i in range(len(s) - 2)]


def name_bigrams(name: str) -> List[str]:
    """Word bigrams of normalized name — alternative blocking signal."""
    tokens = normalize_name(name).split()
    if len(tokens) < 2:
        return tokens
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


# ──────────────────────────────────────────────────────────────────────────────
# Address normalization & field extraction
# ──────────────────────────────────────────────────────────────────────────────

# Regex patterns for postal code extraction — generic, not country-specific
_US_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_IN_PIN_RE = re.compile(r"\b(\d{6})\b")
_FR_CP_RE = re.compile(r"\b(\d{5})\b")
_GENERIC_NUM_RE = re.compile(r"\b(\d{4,6})\b")  # fallback

# Street number at start of address
_STREET_NUM_RE = re.compile(r"^\s*(\d+(?:\s*[A-Za-z])?(?:/\d+)?)\b")


def extract_postal_code(address: str, country: str = "") -> Optional[str]:
    """
    Extract postal code from address. Country-aware but falls back gracefully.
    Returns the first match or None.
    """
    if not address:
        return None
    country = (country or "").upper()
    if country == "US":
        m = _US_ZIP_RE.search(address)
        return m.group(1) if m else None
    elif country == "INDIA" or country == "IN":
        m = _IN_PIN_RE.search(address)
        return m.group(1) if m else None
    elif country == "FRANCE" or country == "FR":
        m = _FR_CP_RE.search(address)
        return m.group(1) if m else None
    else:
        # Generic: try 5-6 digit patterns in order
        for pattern in [_IN_PIN_RE, _US_ZIP_RE, _GENERIC_NUM_RE]:
            m = pattern.search(address)
            if m:
                return m.group(1)
    return None


def extract_street_number(address: str) -> Optional[str]:
    """Extract leading street/house number from address string."""
    if not address:
        return None
    m = _STREET_NUM_RE.match(address.strip())
    return m.group(1).strip() if m else None


def normalize_address(address: str, expand_abbrevs: bool = True) -> str:
    """
    Normalize address: lowercase, expand abbreviations, remove punctuation.
    Country-agnostic.
    """
    if not address or not isinstance(address, str):
        return ""
    tokens = tokenize(address)
    if expand_abbrevs:
        tokens = [ADDR_TOKEN_EXPAND.get(t, t) for t in tokens]
    return " ".join(tokens)


def address_tokens(address: str) -> List[str]:
    """Tokenized & normalized address words — used for Jaccard overlap."""
    return normalize_address(address).split()


def address_trigrams(address: str) -> List[str]:
    """Character trigrams of normalized address — for n-gram blocking."""
    s = normalize_address(address).replace(" ", "")
    if len(s) < 3:
        return [s] if s else []
    return [s[i:i+3] for i in range(len(s) - 2)]


# ──────────────────────────────────────────────────────────────────────────────
# Phonetic encodings
# ──────────────────────────────────────────────────────────────────────────────

def soundex(text: str) -> Optional[str]:
    """Soundex encoding of first meaningful token in text (Latin only)."""
    try:
        import jellyfish
        tokens = core_name(text).split()
        if not tokens:
            return None
        # Use first token that is ASCII-ish (skip non-Latin scripts)
        for t in tokens:
            if all(ord(c) < 128 for c in t) and len(t) >= 2:
                return jellyfish.soundex(t)
        return None
    except Exception:
        return None


def nysiis(text: str) -> Optional[str]:
    """NYSIIS phonetic encoding — better than Soundex for names."""
    try:
        import jellyfish
        tokens = core_name(text).split()
        if not tokens:
            return None
        for t in tokens:
            if all(ord(c) < 128 for c in t) and len(t) >= 2:
                return jellyfish.nysiis(t)
        return None
    except Exception:
        return None


def full_name_soundex(name: str) -> str:
    """Soundex of all tokens joined — for multi-word name phonetic key."""
    try:
        import jellyfish
        tokens = [t for t in core_name(name).split()
                  if all(ord(c) < 128 for c in t) and len(t) >= 2 and t not in LEGAL_SUFFIXES]
        if not tokens:
            return ""
        return "".join(jellyfish.soundex(t) for t in tokens[:3])
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Blocking key generators
# ──────────────────────────────────────────────────────────────────────────────

def blocking_keys(row: dict) -> List[str]:
    """
    Generate all blocking keys for a single record.
    Returns a list of string keys; candidates that share ANY key are a pair.

    Keys are prefixed with type to avoid cross-type collisions.
    Country is included in most keys to reduce cross-country false positives
    (except BK-6, the country-agnostic fallback).
    """
    entity_id = row.get("entity_id", "")
    name = row.get("business_name", "") or ""
    address = row.get("business_address", "") or ""
    country = (row.get("country", "") or "").strip()

    keys = []

    # BK-1: Country + postal code prefix + name prefix (3 chars)
    postal = extract_postal_code(address, country)
    nprefix = name_prefix(name, 3)
    if postal and nprefix:
        keys.append(f"BK1|{country}|{postal[:4]}|{nprefix}")

    # BK-2: Country + soundex of name
    sdx = soundex(name)
    if sdx:
        keys.append(f"BK2|{country}|{sdx}")

    # BK-3: Country + NYSIIS of name (catches more variations than Soundex)
    nys = nysiis(name)
    if nys:
        keys.append(f"BK3|{country}|{nys}")

    # BK-4: Country + street number + soundex of first address token
    street_num = extract_street_number(address)
    addr_toks = address_tokens(address)
    if street_num and addr_toks and len(addr_toks) >= 2:
        first_meaningful_tok = addr_toks[1] if addr_toks[0].isdigit() else addr_toks[0]
        addr_sdx = None
        try:
            import jellyfish
            if all(ord(c) < 128 for c in first_meaningful_tok):
                addr_sdx = jellyfish.soundex(first_meaningful_tok)
        except Exception:
            pass
        if addr_sdx:
            keys.append(f"BK4|{country}|{street_num}|{addr_sdx}")

    # BK-5: Country + normalized core name (exact after normalization)
    cn = core_name(name)
    if cn and len(cn) >= 3:
        keys.append(f"BK5|{country}|{cn}")

    # BK-6: Country + name prefix 5 chars (wider net)
    nprefix5 = name_prefix(name, 5)
    if nprefix5:
        keys.append(f"BK6|{country}|{nprefix5}")

    # BK-7: Country + postal code (no name constraint — catches same location, different name spelling)
    if postal:
        keys.append(f"BK7|{country}|{postal}")

    return keys


# ──────────────────────────────────────────────────────────────────────────────
# Utility: record completeness
# ──────────────────────────────────────────────────────────────────────────────

def name_completeness(name: str) -> int:
    """Number of meaningful name tokens (after normalization)."""
    return len(core_name(name).split())


def address_completeness(address: str) -> int:
    """Number of address tokens (proxy for how complete the address is)."""
    return len(address_tokens(address))


def has_digits_in_name(name: str) -> bool:
    """True if name contains digits (e.g., '7-Eleven', 'H&R Block 123')."""
    return bool(re.search(r"\d", name or ""))


def is_latin_script(text: str) -> bool:
    """True if majority of alphabetic characters are Latin (ASCII range)."""
    if not text:
        return True
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return True
    latin = sum(1 for c in alpha if ord(c) < 256)
    return latin / len(alpha) >= 0.5

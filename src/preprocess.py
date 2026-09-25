import csv
import re
import unicodedata
from typing import Dict, Optional
import pandas as pd

# Comprehensive dictionary mapping Indian states and major territories from regional scripts (Devanagari, Tamil, Kannada, Telugu, Bengali, Gujarati) to English
INDIC_STATE_MAP = {
    # Devanagari / Hindi / Marathi
    "महाराष्ट्र": "maharashtra",
    "उत्तर प्रदेश": "uttar pradesh",
    "मध्य प्रदेश": "madhya pradesh",
    "गुजरात": "gujarat",
    "कर्नाटक": "karnataka",
    "तमिलनाडु": "tamil nadu",
    "तमिल नाडु": "tamil nadu",
    "राजस्थान": "rajasthan",
    "पश्चिम बंगाल": "west bengal",
    "आंध्र प्रदेश": "andhra pradesh",
    "तेलंगाना": "telangana",
    "केरल": "kerala",
    "पंजाब": "punjab",
    "हरियाणा": "haryana",
    "बिहार": "bihar",
    "ओडिशा": "odisha",
    "उड़ीसा": "odisha",
    "झारखंड": "jharkhand",
    "असम": "assam",
    "उत्तराखंड": "uttarakhand",
    "गोवा": "goa",
    "दिल्ली": "delhi",
    "मुंबई": "mumbai",
    "चेन्नई": "chennai",
    "कोलकाता": "kolkata",
    "बेंगलुरु": "bengaluru",
    "हैदराबाद": "hyderabad",
    # Tamil
    "தமிழ்நாடு": "tamil nadu",
    "சென்னை": "chennai",
    "கேரளா": "kerala",
    "கர்நாடகா": "karnataka",
    # Kannada
    "ಕರ್ನಾಟಕ": "karnataka",
    "ಬೆಂಗಳೂರು": "bengaluru",
    "ಮೈಸೂರು": "mysore",
    # Telugu
    "ఆంధ్ర ప్రదేశ్": "andhra pradesh",
    "తెలంగాణ": "telangana",
    "హైదరాబాద్": "hyderabad",
    # Gujarati
    "ગુજરાત": "gujarat",
    "અમદાવાદ": "ahmedabad",
    "સુરત": "surat",
    # Bengali
    "পশ্চিমবঙ্গ": "west bengal",
    "কলকাতা": "kolkata",
}

# French address abbreviations
FRENCH_ABBR_MAP = {
    r"\br\.\b": "rue",
    r"\br\b": "rue",
    r"\bav\.\b": "avenue",
    r"\bav\b": "avenue",
    r"\bave\.\b": "avenue",
    r"\bbd\.\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bblvd\.\b": "boulevard",
    r"\ball\.\b": "allee",
    r"\bpl\.\b": "place",
    r"\brte\b": "route",
    r"\bb\.?p\.?\b": "boite postale",
    r"\bcedex\b": "cedex",
    r"\bs\.?a\.?r\.?l\.?\b": "sarl",
    r"\bs\.?a\.?s\.?\b": "sas",
    r"\bs\.?a\.?\b": "sa",
    r"\bs\.?c\.?\b": "sc",
}

# Indian address abbreviations & landmarks
INDIAN_ABBR_MAP = {
    r"\bb/h\b": "behind",
    r"\bb\.h\.\b": "behind",
    r"\bopp\.?\b": "opposite",
    r"\bo/p\b": "opposite",
    r"\bnr\.?\b": "near",
    r"\bn/r\b": "near",
    r"\badj\.?\b": "adjacent",
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bapt\.?\b": "apartment",
    r"\bflr\.?\b": "floor",
    r"\bfl\.?\b": "floor",
    r"\bsec\.?\b": "sector",
    r"\bind\.?\b": "industrial",
    r"\bindl\.?\b": "industrial",
    r"\best\.?\b": "estate",
    r"\bsoc\.?\b": "society",
    r"\bpvt\.?\b": "private",
    r"\bp\.ltd\.?\b": "private limited",
    r"\bltd\.?\b": "limited",
    r"\bcorp\.?\b": "corporation",
    r"\bco\.?\b": "company",
    r"\bent\.?\b": "enterprise",
    r"\bmfg\.?\b": "manufacturing",
}

# US address abbreviations
US_ABBR_MAP = {
    r"\bst\.\b": "street",
    r"\bave\.\b": "avenue",
    r"\brd\.\b": "road",
    r"\bblvd\.\b": "boulevard",
    r"\bdr\.\b": "drive",
    r"\bln\.\b": "lane",
    r"\bct\.\b": "court",
    r"\bpl\.\b": "place",
    r"\bpkwy\.\b": "parkway",
    r"\bhwy\.\b": "highway",
    r"\bste\.\b": "suite",
    r"\bapt\.\b": "apartment",
    r"\bfl\.\b": "floor",
    r"\bcorp\.\b": "corporation",
    r"\binc\.\b": "incorporated",
    r"\bllc\b": "llc",
}


def strip_accents_and_normalize(text: str) -> str:
    """
    Applies Unicode NFKD decomposition and removes non-spacing mark accents.
    Converts French characters like 'Àrt' -> 'Art', 'Frères' -> 'Freres', 'Lège' -> 'Lege'.
    Preserves Indic scripts (Devanagari, Tamil, etc.) while stripping European accents.
    """
    if not text or not isinstance(text, str):
        return ""
    # NFKD decomposition
    nfkd_form = unicodedata.normalize("NFKD", text)
    # Strip combining diacritical marks (accents)
    stripped = "".join(c for c in nfkd_form if unicodedata.category(c) != "Mn")
    return stripped


def deduplicate_consecutive_tokens(text: str) -> str:
    """
    Removes adjacent repeated tokens (stuttering).
    E.g. 'Pvt Pvt Ltd Services' -> 'Pvt Ltd Services'
         'Unit Unit 2' -> 'Unit 2'
    """
    if not text:
        return ""
    # Matches case-insensitively duplicated words separated by spaces
    pattern = re.compile(r"\b(\w+)(?:\s+\1\b)+", flags=re.IGNORECASE)
    return pattern.sub(r"\1", text)


def normalize_indic_scripts(text: str) -> str:
    """
    Substitutes regional script representations of states and cities with their standard English tokens.
    """
    if not text:
        return ""
    for indic_token, english_token in INDIC_STATE_MAP.items():
        if indic_token in text:
            text = text.replace(indic_token, english_token)
    return text


def expand_abbreviations(text: str, country: Optional[str] = None) -> str:
    """
    Applies country-aware regex abbreviation expansions.
    """
    if not text:
        return ""
    
    country_lower = str(country).lower() if country else ""

    # Indian expansions
    if "india" in country_lower or "in" == country_lower:
        for pattern, replacement in INDIAN_ABBR_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # French expansions
    elif "france" in country_lower or "fr" == country_lower:
        for pattern, replacement in FRENCH_ABBR_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # US expansions
    elif "us" in country_lower or "united states" in country_lower or "usa" == country_lower:
        for pattern, replacement in US_ABBR_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Apply general cleanups regardless of country
    for pattern, replacement in INDIAN_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for pattern, replacement in FRENCH_ABBR_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def clean_text(text: str, country: Optional[str] = None) -> str:
    """
    End-to-end cleaning pipeline for business names and addresses.
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    # 1. Map Indic state names in local scripts
    text = normalize_indic_scripts(text)
    
    # 2. Unicode NFKD normalization and accent stripping
    text = strip_accents_and_normalize(text)
    
    # 3. Lowercase
    text = text.lower()
    
    # 4. Expand abbreviations
    text = expand_abbreviations(text, country)
    
    # 5. Consecutive token deduplication
    text = deduplicate_consecutive_tokens(text)
    
    # 6. Replace multiple spaces and punctuation noise with single space
    text = re.sub(r"[^\w\s\d]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def extract_numeric_tokens(text: str) -> set:
    """
    Extracts all numeric sequences (house numbers, postal codes, unit numbers).
    Useful for digit Jaccard overlap on inverted or noisy addresses.
    """
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", str(text)))


def load_tsv(filepath: str) -> pd.DataFrame:
    """
    Loads TSV file with strict sep='\t' and quoting=csv.QUOTE_NONE as specified in the competition prompt.
    """
    return pd.read_csv(
        filepath,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )


def save_tsv(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves DataFrame to TSV with strict sep='\t', index=False, quoting=csv.QUOTE_NONE.
    """
    df.to_csv(
        filepath,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        index=False,
        encoding="utf-8"
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive preprocessing to an entity DataFrame:
    Adds clean_business_name, clean_business_address, and combined_text columns.
    """
    df = df.copy()
    
    # Standardize column strings
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str).str.strip()

    # Clean name and address
    clean_names = []
    clean_addrs = []
    combined_texts = []

    for _, row in df.iterrows():
        c = row["country"]
        cn = clean_text(row["business_name"], country=c)
        ca = clean_text(row["business_address"], country=c)
        comb = f"{cn} {ca}".strip()
        clean_names.append(cn)
        clean_addrs.append(ca)
        combined_texts.append(comb)

    df["clean_business_name"] = clean_names
    df["clean_business_address"] = clean_addrs
    df["combined_text"] = combined_texts
    
    return df

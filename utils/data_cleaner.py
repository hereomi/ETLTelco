import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Union, List, Any, Dict, Optional, Set
from dataclasses import dataclass, field

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    strip_bom: bool = True
    normalize_whitespace: bool = True
    standardize_smart_chars: bool = True
    remove_control_chars: bool = True
    normalize_unicode: Optional[str] = 'NFKC'  # NFKC, NFKD, NFC, NFD or None
    handle_na: bool = True
    trim: bool = True
    lowercase_headers: bool = False
    remove_special_chars: bool = False  # If True, removes everything except \w and \s
    
    # Custom replacement map for specific characters
    custom_map: Dict[str, str] = field(default_factory=lambda: {
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\u200c': '',    # Zero-width non-joiner
        '\u200d': '',    # Zero-width joiner
        '\u2060': '',    # Word joiner
        '\ufeff': '',    # BOM
        '\u180e': '',    # Mongolian vowel separator (invisible)
        '\u202f': ' ',   # Narrow non-breaking space
        
        # Smart Punctuation
        '\u201c': '"',   # Smart Double Left
        '\u201d': '"',   # Smart Double Right
        '\u2018': "'",   # Smart Single Left
        '\u2019': "'",   # Smart Single Right
        '\u201b': "'",   # Single high-reversed-9 quote
        '\u201e': '"',   # Double low-9 quote
        '\u201f': '"',   # Double high-reversed-9 quote
        
        # Dashes
        '\u2013': '-',   # en dash
        '\u2014': '-',   # em dash
        '\u2015': '-',   # horizontal bar
        
        # Others
        '\u2026': '...', # ellipsis
        
        # Homoglyphs (Cyrillic to Latin)
        '\u0410': 'A', '\u0430': 'a', # A
        '\u0412': 'B',               # B
        '\u0415': 'E', '\u0435': 'e', # E
        '\u041a': 'K', '\u043a': 'k', # K
        '\u041c': 'M',               # M
        '\u041d': 'H',               # H
        '\u041e': 'O', '\u043e': 'o', # O
        '\u0420': 'P', '\u0440': 'p', # P
        '\u0421': 'C', '\u0441': 'c', # C
        '\u0422': 'T',               # T
        '\u0425': 'X', '\u0445': 'x', # X
        '\u0443': 'y',               # y
        
        # Homoglyphs (Greek to Latin)
        '\u0391': 'A', '\u03b1': 'a', # Alpha
        '\u0392': 'B', '\u03b2': 'b', # Beta
        '\u0395': 'E', '\u03b5': 'e', # Epsilon
        '\u0397': 'H',               # Eta
        '\u0399': 'I', '\u03b9': 'i', # Iota
        '\u039a': 'K', '\u03ba': 'k', # Kappa
        '\u039d': 'N', '\u03bd': 'v', # Nu
        '\u039f': 'O', '\u03bf': 'o', # Omicron
        '\u03a1': 'P', '\u03c1': 'p', # Rho
        '\u03a4': 'T', '\u03c4': 't', # Tau
        '\u03a5': 'Y', '\u03c5': 'y', # Upsilon
        '\u03a7': 'X', '\u03c7': 'x', # Chi
    })

def clean_string(text: Any, config: Optional[CleaningConfig] = None) -> Any:
    """
    Robustly cleans a single value.
    
    Args:
        text: The value to clean (can be any type).
        config: Optional CleaningConfig instance.
    """
    if config is None:
        config = CleaningConfig()
        
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None if config.handle_na else text
        
    if not isinstance(text, str):
        return text
    
    # 1. Unicode Normalization (NFKC is usually best for data engineering)
    # It converts homoglyphs like the Cyrillic 'А' to Latin 'A' if they are canonicallly equivalent
    if config.normalize_unicode:
        text = unicodedata.normalize(config.normalize_unicode, text)
        
    # 2. Custom Map Replacements
    for old, new in config.custom_map.items():
        text = text.replace(old, new)
        
    # 3. Control Characters
    if config.remove_control_chars:
        # Keep \n, \r, \t
        text = "".join(ch for ch in text if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C")
        
    # 4. Remove Special Characters (Aggressive)
    if config.remove_special_chars:
        text = re.sub(r'[^\w\s\n\r\t]', '', text)
        
    # 5. Trim
    if config.trim:
        text = text.strip()
        
    return text

def clean_series(s: pd.Series, config: Optional[CleaningConfig] = None) -> pd.Series:
    """
    Robustly cleans a pandas Series using vectorized operations where possible.
    """
    if config is None:
        config = CleaningConfig()
        
    # Only process strings
    if not pd.api.types.is_object_dtype(s.dtype) and not pd.api.types.is_string_dtype(s.dtype):
        return s
        
    # 1. Basic preprocessing (convert to string if mixed, handle NA)
    working_s = s.astype(str).replace(['nan', 'None', 'NAT'], np.nan) if config.handle_na else s.astype(str)
    
    # 2. Vectorized Replacements from Map
    for old, new in config.custom_map.items():
        working_s = working_s.str.replace(old, new, regex=False)
        
    # 3. Unicode Normalization (not directly vectorized in pandas, but can be done via map)
    if config.normalize_unicode:
        working_s = working_s.apply(lambda x: unicodedata.normalize(config.normalize_unicode, x) if isinstance(x, str) else x)
        
    # 4. Control Characters and Regex steps
    if config.remove_control_chars:
        # Regex to remove non-printable control chars, preserving whitespace
        # [^\x20-\x7E] covers basic ASCII non-printable, but we want wider Unicode C category
        # Since Unicode 'C' is hard in regex, we keep the apply logic or use a specific regex
        working_s = working_s.apply(lambda x: "".join(ch for ch in x if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C") if isinstance(x, str) else x)

    if config.remove_special_chars:
        working_s = working_s.str.replace(r'[^\w\s\n\r\t]', '', regex=True)
        
    # 5. Trim
    if config.trim:
        working_s = working_s.str.strip()
        
    return working_s

def clean_headers(df: pd.DataFrame, config: Optional[CleaningConfig] = None) -> pd.DataFrame:
    """
    Robustly cleans DataFrame headers.
    """
    if config is None:
        config = CleaningConfig()
        
    new_cols = []
    for col in df.columns:
        c = clean_string(str(col), config)
        if config.lowercase_headers:
            c = c.lower()
        # Ensure header is valid for most DBs (no spaces, alphanumeric)
        c = re.sub(r'[^a-zA-Z0-9_]', '_', c)
        c = re.sub(r'__+', '_', c).strip('_')
        new_cols.append(c)
        
    df.columns = new_cols
    return df

def robust_clean(df: pd.DataFrame, config: Optional[CleaningConfig] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Perform a robust clean of the entire DataFrame headers and data.
    """
    if config is None:
        config = CleaningConfig()
        
    # Clean Headers first
    df = clean_headers(df, config)
    
    # Clean Data
    target_cols = columns if columns else df.select_dtypes(include=['object', 'string']).columns
    
    for col in target_cols:
        df[col] = clean_series(df[col], config)
        
    return df

# Shortcut for standard data engineering cleanliness
def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standard robust cleanup: BOM, Trim, Smart Chars, Controls."""
    config = CleaningConfig(lowercase_headers=False)
    return robust_clean(df, config)

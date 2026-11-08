"""Text normalization for Arabic, Darija (Arabic script + Arabizi) and French.

Two things live here:
  * `normalize()`     -> canonical form used for keyword / entity matching
  * `skeleton()`      -> script-independent consonant key, so that
                         "Doliprane", "doliprane" and "دوليبران" collide.
"""
from __future__ import annotations

import re
import unicodedata

ARABIC_RE = re.compile(r"[؀-ۿ]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # diacritics + tatweel
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_AR_CHARS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي",
                           "ڭ": "ك", "ݣ": "ك", "گ": "ك", "ڤ": "ف", "ﻻ": "لا", "پ": "ب"})

# Arabic clitics that commonly glue onto a word: و ف ب ل ال وال فال بال لل
AR_PREFIX = r"(?:و|ف|ب|ل)?(?:ال|ل)?"


def has_arabic(s: str) -> bool:
    return bool(ARABIC_RE.search(s))


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Lowercase, unify Arabic letter variants, strip diacritics/accents, western digits,
    split glued numbers/units ("500mg" -> "500 mg"), collapse spaces."""
    t = text.translate(_ARABIC_INDIC)
    t = _TASHKEEL.sub("", t).translate(_AR_CHARS)
    t = strip_accents(t.lower())
    t = re.sub(r"[’'`´]", " ", t)                     # l'aspegic -> l aspegic
    t = re.sub(r"(\d+)\s*(mg|g|gr|ml|mcg|ug|ui|%)\b", r"\1 \2", t)
    t = re.sub(r"[^\w\s%.,]", " ", t)                  # keep letters/digits (\w covers Arabic)
    t = re.sub(r"(?<!\d)[.,]|[.,](?!\d)", " ", t)      # punctuation, but keep 2,5 / 0.5
    return re.sub(r"\s+", " ", t).strip()


def tokens(text_norm: str) -> list[str]:
    return text_norm.split()


def ar_variants(tok: str) -> list[str]:
    """All readings of an Arabic token with/without clitic prefixes: 'فلاجيل' must stay a
    candidate even though it starts with ف."""
    if not has_arabic(tok):
        return [tok]
    out = [tok]
    for p in ("وبال", "فال", "بال", "وال", "لل", "ال", "ول", "ب", "ف", "و", "ل"):
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            out.append(tok[len(p):])
    return out


def strip_ar_prefix(tok: str) -> str:
    """'بالدوليبران' -> 'دوليبران' (only for Arabic-script tokens long enough to survive)."""
    if not has_arabic(tok):
        return tok
    for p in ("وبال", "فال", "بال", "وال", "لل", "ال", "ول", "ب", "ف", "و", "ل"):
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            return tok[len(p):]
    return tok


# --------------------------------------------------------------------------- skeleton
_AR_SKEL = {
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "خ": "k", "د": "d", "ذ": "d", "ر": "r", "ز": "z",
    "س": "s", "ش": "c", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "غ": "k", "ف": "f", "ق": "k",
    "ك": "k", "ل": "l", "م": "m", "ن": "n",
    # vowels / glides / gutturals vanish, as latin vowels and silent h do
    "ا": "", "و": "", "ي": "", "ه": "", "ة": "", "ء": "", "ع": "", "ح": "",
}


def _latin_skeleton(w: str) -> str:
    w = strip_accents(w.lower())
    w = re.sub(r"[^a-z]", "", w)
    w = w.replace("ph", "f").replace("ch", "0").replace("sh", "0")   # park the "sh" sound (Arabic ش)
    w = w.replace("ck", "k").replace("qu", "k").replace("x", "ks")
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = re.sub(r"g(?=[eiy])", "j", w)
    w = w.replace("c", "k").replace("q", "k").replace("g", "k")
    w = w.replace("p", "b").replace("v", "f")
    return w.replace("0", "c")


def skeleton(word: str) -> str:
    """Consonant skeleton shared by Latin and Arabic spellings of a drug name."""
    if has_arabic(word):
        raw = "".join(_AR_SKEL.get(ch, "") for ch in word)
    else:
        raw = _latin_skeleton(word)
        raw = re.sub(r"[aeiouyhw]", "", raw)
    raw = raw.replace("z", "s")              # intervocalic French s is written ز in Arabic (Lasilix/لازيليكس)
    return re.sub(r"(.)\1+", r"\1", raw)  # collapse doubles


_AR_VOWELS = {"ا": "a", "و": "o", "ي": "i", "ى": "i", "ة": "a"}


def loose_latin(word: str) -> str:
    """Vowel-preserving romanization, used only to break ties between drugs sharing a skeleton
    (e.g. BRUFEN vs IBUPROFENE -> both 'brfn')."""
    if has_arabic(word):
        return "".join(_AR_VOWELS.get(ch, _AR_SKEL.get(ch, "")) for ch in word)
    return _latin_skeleton(word)

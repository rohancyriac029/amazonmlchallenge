"""Text normalization for business names and addresses.

Produces several representations per record (conservative, aggressive/core,
compact, consonant skeleton, digit tokens).  Everything here is rule based and
country agnostic: Indic scripts are transliterated with a single offset table
(all ISCII-derived Unicode blocks share the same layout), Latin text is
accent-folded, and legal/filler vocabulary covers common forms across
jurisdictions.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Indic -> Latin transliteration (Devanagari, Bengali, Gurmukhi, Gujarati,
# Oriya, Tamil, Telugu, Kannada, Malayalam share a parallel 128-codepoint layout)
# ---------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
           0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
         0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
         0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
         0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
         0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
         0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r",
         0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
           0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
           0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au", 0x55: "", 0x56: "ai"}
_VIRAMA = 0x4D
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}


def _indic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp < 0x0D80:
        return cp & 0x7F
    return None


def transliterate_indic(text):
    """Rule-based Indic->Latin transliteration with inherent-vowel handling."""
    if not any(0x0900 <= ord(c) < 0x0D80 for c in text):
        return text
    out = []
    pending_a = False  # a consonant was emitted and awaits its inherent vowel
    conjunct = False   # the pending consonant closes a conjunct (keeps final schwa)
    after_virama = False
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if ch in "‌‍":
                continue
            if pending_a and conjunct:
                out.append("a")  # aditya, mitra: final schwa kept after a conjunct
            pending_a = False
            after_virama = False
            out.append(ch)
            continue
        if off in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[off])
            conjunct = after_virama and _CONS[off] in ("y", "r", "v")
            pending_a = True
            after_virama = False
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending_a = False
            after_virama = False
        elif off == _VIRAMA:
            pending_a = False
            after_virama = True
        elif off in _VOWELS:
            # schwa before an independent vowel is not pronounced (e.g. es-es)
            pending_a = False
            after_virama = False
            out.append(_VOWELS[off])
        elif off in _NASAL:
            if pending_a:
                out.append("a")
                pending_a = False
            after_virama = False
            out.append(_NASAL[off])
        elif 0x66 <= off <= 0x6F:
            pending_a = False
            out.append(str(off - 0x66))
        else:  # nukta, avagraha, misc signs
            continue
    if pending_a and conjunct:
        out.append("a")
    return "".join(out)


# ---------------------------------------------------------------------------
# Generic cleanup
# ---------------------------------------------------------------------------
_NULL_TOKENS_RE = re.compile(r"(?<![a-z0-9])(?:<null>|null|none|nan|n/a|na)(?![a-z0-9])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def fold(text):
    """Transliterate, strip accents, lowercase, collapse whitespace."""
    if not text:
        return ""
    text = transliterate_indic(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("�", "")
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
# legal forms / generic corporate vocabulary (multi-jurisdiction, canonical form)
LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co", "ltd": "ltd", "limited": "ltd", "llc": "llc",
    "llp": "llp", "lp": "lp", "pvt": "pvt", "private": "pvt", "plc": "plc", "pc": "pc",
    "pa": "pa", "pllc": "pllc", "gmbh": "gmbh", "sa": "sa", "sas": "sas", "sasu": "sasu",
    "sarl": "sarl", "eurl": "eurl", "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl",
    "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa", "esq": "esq",
    # transliterated Indic spellings of legal suffixes
    "praivet": "pvt", "praivhet": "pvt", "praiveta": "pvt", "limited": "ltd", "limitad": "ltd",
    "limited.": "ltd", "limitedd": "ltd", "kampani": "co", "kanpani": "co", "elelpi": "llp",
    "ielelpi": "llp", "elalpi": "llp", "incorporeted": "inc",
}
# generic words that noise generators and real sources add/drop freely
FILLER = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "dba", "d", "b", "a",
          "services", "service", "partners", "center", "centre", "group", "holdings",
          "enterprises", "solutions", "international", "smt", "shri", "sri", "m", "s", "ms"}
_DBA_RE = re.compile(r"\b(?:d\.?\s?b\.?\s?a\.?|doing business as|t/a|a\.k\.a\.?|aka)\b")
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)((?:\.[a-z]{2,4}){1,2})/?$")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_HASHNUM_RE = re.compile(r"#\s?\d+")


# English letter names as they appear after transliteration (acronyms such as
# LLP / SS are commonly written phonetically in Indic scripts)
_LETTER_NAMES = {
    "e": "a", "ai": "i", "bi": "b", "si": "c", "di": "d", "i": "e", "ef": "f", "eph": "f",
    "ji": "g", "ech": "h", "eich": "h", "aich": "h", "je": "j", "ke": "k", "el": "l",
    "em": "m", "en": "n", "o": "o", "pi": "p", "kyu": "q", "ar": "r", "aar": "r", "es": "s",
    "ti": "t", "yu": "u", "vi": "v", "dablyu": "w", "dabalyu": "w", "eks": "x", "vai": "y",
    "jed": "z", "jhed": "z", "zed": "z",
}
# consonant-final letter names may carry an epenthetic schwa (el-a-pi)
_LETTER_NAMES.update({k + "a": v for k, v in list(_LETTER_NAMES.items())
                      if k[-1] not in "aeiou" and len(k) >= 2})
_LETTER_KEYS = sorted(_LETTER_NAMES, key=len, reverse=True)


def decode_spelled_acronym(tok):
    """'elelpi' -> 'llp', 'eses' -> 'ss'; returns None if not fully decodable."""
    n = len(tok)
    if n < 3 or n > 24:
        return None
    best = [None] * (n + 1)
    best[0] = ""
    for i in range(n):
        if best[i] is None:
            continue
        for k in _LETTER_KEYS:
            if tok.startswith(k, i):
                j = i + len(k)
                cand = best[i] + _LETTER_NAMES[k]
                if best[j] is None or len(cand) < len(best[j]):
                    best[j] = cand
    res = best[n]
    if res is None or len(res) < 2:
        return None
    return res


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "$": "s"})
_WEB_JUNK = {"www", "com", "net", "org", "http", "https"}


def deleet(tok):
    """'5even' -> 'seven', 'ne0tech' -> 'neotech' (only mixed alpha-digit tokens)."""
    n_alpha = sum(c.isalpha() for c in tok)
    if n_alpha >= 2 and n_alpha < len(tok) and not tok.isdigit():
        return tok.translate(_LEET)
    return tok


_TRANSLIT_DICT = {}


def set_translit_dict(d):
    """Install a learned {transliterated token: english token} dictionary."""
    _TRANSLIT_DICT.clear()
    _TRANSLIT_DICT.update(d)


def has_indic(text):
    return any(0x0900 <= ord(c) < 0x0D80 for c in text or "")


def name_repr(raw):
    """Return dict of name representations."""
    s = fold(raw)
    is_domain = 0
    m = _DOMAIN_RE.match(s.replace(" ", "")) if s and " " not in s.strip() else None
    if m and "." in s:
        s = m.group(1).replace("-", " ")
        is_domain = 1
    has_dba = 1 if _DBA_RE.search(s) else 0
    has_phone = 1 if _PHONE_RE.search(s) else 0
    s = _PHONE_RE.sub(" ", s)
    s = _HASHNUM_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " ")
    s = _DBA_RE.sub(" dba ", s)
    # join dotted initialisms (l.l.c. -> llc, c.i.t. -> cit)
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda mm: mm.group(1).replace(".", ""), s)
    toks = [t for t in _NON_ALNUM_RE.split(s) if t]
    indic = has_indic(raw)
    if indic:
        fixed = []
        for t in toks:
            if t in _TRANSLIT_DICT:
                fixed.append(_TRANSLIT_DICT[t])
                continue
            acr = decode_spelled_acronym(t)
            fixed.append(acr if acr else t)
        toks = fixed
    toks = [deleet(t) for t in toks]
    toks = [t for t in toks if t not in _WEB_JUNK] or toks
    conservative = " ".join(toks)
    canon = [LEGAL.get(t, t) for t in toks]
    core = [t for t in canon if t not in FILLER and t not in LEGAL.values() and not t.isdigit()]
    if not core:
        core = [t for t in canon if not t.isdigit()] or canon
    legal = sorted({t for t in canon if t in LEGAL.values()})
    return {
        "n_cons": conservative,
        "n_core": " ".join(core),
        "n_sorted": " ".join(sorted(core)),
        "n_compact": "".join(core),
        "n_skel": " ".join(skeleton(t) for t in core),
        "n_legal": " ".join(legal),
        "n_digits": " ".join(t for t in toks if t.isdigit()),
        "n_is_domain": is_domain,
        "n_has_dba": has_dba,
        "n_has_phone": has_phone,
        "n_indic": int(indic),
    }


_SKEL_SUBS = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"), ("bh", "b"),
              ("kh", "k"), ("gh", "g"), ("jh", "j"), ("ck", "k"), ("w", "v"), ("q", "k"),
              ("z", "j"), ("x", "ks"), ("y", "i")]
_VOWEL_RE = re.compile(r"[aeiou]+")
_REPEAT_RE = re.compile(r"(.)\1+")


def skeleton(tok):
    """Phonetic consonant skeleton: robust to vowel typos and transliteration."""
    if not tok:
        return ""
    t = tok
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = t.replace("c", "k").replace("g", "k").replace("d", "t").replace("b", "p")
    head = t[0]
    rest = _VOWEL_RE.sub("", t[1:])
    return _REPEAT_RE.sub(r"\1", head + rest)


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
ADDR_ABBR = {
    "st": "st", "street": "st", "str": "st", "saint": "st", "ste": "ste", "suite": "ste",
    "rd": "rd", "road": "rd", "ave": "ave", "av": "ave", "avenue": "ave", "blvd": "blvd",
    "boulevard": "blvd", "bd": "blvd", "dr": "dr", "drive": "dr", "ln": "ln", "lane": "ln",
    "ct": "ct", "court": "ct", "pl": "pl", "place": "pl", "sq": "sq", "square": "sq",
    "hwy": "hwy", "highway": "hwy", "pkwy": "pkwy", "parkway": "pkwy", "ter": "ter",
    "terrace": "ter", "cir": "cir", "circle": "cir", "trl": "trl", "trail": "trl",
    "apt": "apt", "apartment": "apt", "apartments": "apt", "fl": "fl", "floor": "fl",
    "no": "no", "number": "no", "nr": "near", "opp": "opp", "opposite": "opp",
    "bldg": "bldg", "building": "bldg", "n": "n", "north": "n", "s": "s", "south": "s",
    "e": "e", "east": "e", "w": "w", "west": "w", "ne": "ne", "nw": "nw", "se": "se",
    "sw": "sw", "mt": "mt", "mount": "mt", "ft": "ft", "fort": "ft", "twp": "twp",
    "township": "twp", "r": "rue", "rue": "rue", "imp": "imp", "impasse": "imp",
    "ch": "chemin", "chemin": "chemin", "rte": "rte", "route": "rte", "all": "allee",
    "allee": "allee", "hno": "hno", "h": "h", "sec": "sector", "sector": "sector",
    "nagar": "nagar", "ngr": "nagar", "mkt": "market", "market": "market",
    "dist": "dist", "district": "dist", "tq": "taluk", "taluk": "taluk", "po": "po",
    "box": "box", "unit": "unit", "ph": "ph", "phase": "phase", "blk": "block", "block": "block",
}
ADDR_STOP = {"null", "none", "nan", "na", "near", "opp", "and", "of", "the", "de", "du",
             "des", "la", "le", "les", "no", "d", "l"}
_ORD_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
              "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
              "eleventh": "11", "twelfth": "12", "thirteenth": "13", "fourteenth": "14",
              "fifteenth": "15", "sixteenth": "16", "seventeenth": "17", "eighteenth": "18",
              "nineteenth": "19", "twentieth": "20"}
_ORD_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_NUMALPHA_RE = re.compile(r"(\d+)([a-z]+)|([a-z]+)(\d+)")


def addr_repr(raw):
    s = fold(raw)
    s = _NULL_TOKENS_RE.sub(" ", s)
    s = s.replace("&", " and ")
    s = _ORD_RE.sub(r"\1", s)
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p]
    toks = []
    for t in _NON_ALNUM_RE.split(s):
        if not t:
            continue
        t = _ORD_WORDS.get(t, t)
        mm = _NUMALPHA_RE.fullmatch(t)
        if mm and len(t) <= 12:
            toks.extend([x for x in mm.groups() if x])
        else:
            toks.append(t)
    canon = []
    for t in toks:
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_ABBR.get(t, t)
        canon.append(t)
    words = [t for t in canon if not t.isdigit() and t not in ADDR_STOP]
    nums = [t for t in canon if t.isdigit()]
    first_num = ""
    for t in canon:
        if t.isdigit():
            first_num = t
            break
    pin = [t for t in nums if len(t) in (5, 6)]
    return {
        "a_norm": " ".join(canon),
        "a_words": " ".join(words),
        "a_sorted": " ".join(sorted(set(words))),
        "a_nums": " ".join(nums),
        "a_first_num": first_num,
        "a_pin": " ".join(pin),
        "a_ncomp": len(parts),
        "a_empty": int(len(canon) == 0),
    }

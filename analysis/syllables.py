"""Syllable and phoneme counts, ported from Lenka's thesis (appendix C.9).

Kept identical to the thesis code so that results are comparable with hers,
including two quirks: normalization maps 'í' and 'ī' to uppercase 'I' (such
words then fall back to the heuristic, whose vowel regex is lowercase), and
the heuristic adds a syllable for a consonant + 'le' ending after counting the
'e' already (e.g. 'table' -> 3) for words missing from the CMU dictionary.

Requires NLTK's CMU Pronouncing Dictionary (nltk.download('cmudict')).
"""
import re

import numpy as np
from nltk.corpus import cmudict

cmu_dict = cmudict.dict()


def normalise_token_for_syllables(word):
    word = str(word).lower()
    word = re.sub(r"[^a-zàáâäæãåāèéêëēėęìíîïīòóôöõøōùúûüūýÿ' -]", "", word)
    word = (word.replace("à", "a").replace("á", "a").replace("â", "a").replace("ä", "a")
            .replace("æ", "ae").replace("ã", "a").replace("å", "a").replace("ā", "a")
            .replace("è", "e").replace("é", "e").replace("ê", "e").replace("ë", "e")
            .replace("ē", "e").replace("ė", "e").replace("ę", "e")
            .replace("ì", "i").replace("í", "I")
            .replace("î", "i").replace("ï", "i").replace("ī", "I")
            .replace("ò", "o").replace("ó", "o").replace("ô", "o").replace("ö", "o")
            .replace("õ", "o").replace("ø", "o").replace("ō", "o")
            .replace("ù", "u").replace("ú", "u").replace("û", "u")
            .replace("ü", "u").replace("ū", "u")
            .replace("ý", "y").replace("ÿ", "y"))
    word = word.replace("'", "").replace(" ", "").replace("-", "")
    return word


def is_vowel(phoneme):
    return phoneme[-1].isdigit()


def count_syllables(word):
    normalized = normalise_token_for_syllables(word)
    if not normalized:
        return 0
    if normalized in cmu_dict:
        counts = [sum(1 for p in pron if is_vowel(p)) for pron in cmu_dict[normalized]]
        if counts:
            return int(round(float(np.median(counts))))
    count = len(re.findall(r"[aeiouy]+", normalized)) or 1
    if normalized.endswith("e") and not normalized.endswith(("le", "ye")) and count > 1:
        count -= 1
    if normalized.endswith("ed") and len(normalized) > 3:
        if not re.search(r"[td]ed$", normalized) and count > 1:
            count -= 1
    if normalized.endswith("es") and len(normalized) > 3:
        if not re.search(r"(ses|xes|zes|ches|shes)$", normalized) and count > 1:
            count -= 1
    if re.search(r"[^aeiouy]le$", normalized):
        count += 1
    return max(1, count)


def count_phonemes(word):
    normalized = normalise_token_for_syllables(word)
    if normalized in cmu_dict:
        counts = [len(pron) for pron in cmu_dict[normalized]]
        if counts:
            return int(round(float(np.median(counts))))
    patterns = sorted([
        "augh", "igh", "tch", "dge", "ch", "sh", "th", "ph", "gh", "ng", "ck",
        "qu", "wh", "ee", "oo", "ou", "ow", "oi", "oy", "ai", "ay",
        "ea", "oa", "ie"], key=len, reverse=True)
    w = normalized.lower()
    count = 0
    i = 0
    while i < len(w):
        match = next((p for p in patterns if w.startswith(p, i)), None)
        if match:
            count += 1
            i += len(match)
        else:
            count += int(w[i].isalpha())
            i += 1
    if w.endswith("e") and not w.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def thesis_tokens(sentence):
    """Sentence-level MAL tokens as in the thesis: whitespace split, then
    normalization; empty results are dropped. Returns (token, start, end) with
    character offsets of the whitespace token in the sentence."""
    out = []
    for m in re.finditer(r"\S+", sentence):
        norm = normalise_token_for_syllables(m.group())
        if norm:
            out.append((norm, m.start(), m.end()))
    return out

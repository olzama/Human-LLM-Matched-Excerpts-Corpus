#!/usr/bin/env python3
"""Turn the corpus's sentence_list fields into parser-ready items.

Reads the corpus directly from llm_dataset_LL.zip. Segmentation is the
corpus's own `sentence_list`, refined in one way only: an entry that spans a
paragraph break (blank line) is split there. Non-prose pieces (headings,
titles, scene breaks, illustration captions) and excerpt-boundary fragments
cut off by the excerpting are dropped, each with a logged reason.

Outputs (in --out):
  items/<corpus>.jsonl   one line per kept item
  dropped.tsv            every dropped piece with its reason
"""
import argparse
import ast
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CORPORA = {
    'human': 'human corpus sample/',
    'claude_sonnet_5_5': 'AI corpus/claude_sonnet_5_5/run_01/',
    'gemini_3_1_pro': 'AI corpus/gemini_3_1_pro/run_01/',
    'gpt_6_sol': 'AI corpus/gpt_6_sol/run_01/',
    'mistral_large_4': 'AI corpus/mistral_large_4/run_01/',
}
# Corpus index used in globally unique integer item ids:
# i-id = corpus_idx * 10_000_000 + excerpt_idx * 1_000 + item_idx
CORPUS_IDX = {c: i + 1 for i, c in enumerate(CORPORA)}

PARA_BREAK = re.compile(r'\n[ \t]*\n')
TERMINAL = re.compile(r'[.!?…]["\'”’)\]_*]*$|[.!?…][”’"\']?\s*[—-]*$')
SCENE_BREAK = re.compile(r'^[\s*\-_=~·•#]+$')
MD_HEADING = re.compile(r'^#{1,6}(\s|$)')
BOLD_LINE = re.compile(r'^\*\*[^*]+\*\*$')
CHAPTER = re.compile(r'^(chapter|book|part|volume|canto)\b[\s\w.:,\'’-]*$', re.I)
ROMAN = re.compile(r'^[IVXLC]+\.?$')
ILLUSTRATION = re.compile(r'\[Illustration[^\]]*\]', re.I)
FOOTNOTE_MARK = re.compile(r'\[\d+\]')

CLOSING_PUNCT_ONLY = re.compile(r'^\s*["\'”’_*]+\s*$')

PUNCT_MAP = str.maketrans({'“': '"', '”': '"', '„': '"', '‘': "'", '’': "'"})


def clean(text):
    """Remove layout and markup that is not part of the sentence."""
    text = ILLUSTRATION.sub(' ', text)
    text = FOOTNOTE_MARK.sub('', text)
    text = text.replace('*', '').replace('_', '')   # markdown / Gutenberg emphasis
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_punct(text):
    """One convention for quotes and dashes across all subcorpora."""
    text = text.translate(PUNCT_MAP)
    text = text.replace('--', '—')
    return text


def heading_reason(raw):
    s = raw.strip()
    if SCENE_BREAK.match(s):
        return 'scene-break'
    if MD_HEADING.match(s):
        return 'md-heading'
    if BOLD_LINE.match(s):
        return 'bold-heading'
    if ILLUSTRATION.fullmatch(s):
        return 'illustration'
    c = clean(s)
    if CHAPTER.match(c) and len(c.split()) <= 12:
        return 'chapter-heading'
    if ROMAN.match(c):
        return 'roman-numeral'
    if not re.search(r'[A-Za-z]', c):
        return 'no-letters'
    return None


def is_title(c):
    """Short, unpunctuated, title-cased or all-caps line without quotes."""
    if TERMINAL.search(c) or re.search(r'[,:;"“”—-]$|["“”]', c):
        return False
    words = re.findall(r"[A-Za-z][\w'’-]*", c)
    if not words or len(words) > 12:
        return False
    content = [w for w in words if len(w) > 3] or words
    return sum(w[0].isupper() for w in content) / len(content) >= 0.7


def pieces_of(entry):
    return [p for p in PARA_BREAK.split(entry) if p.strip()]


def process_excerpt(corpus, sentence_list):
    """Return (kept, dropped): lists of dicts."""
    pieces = []
    for s_idx, entry in enumerate(sentence_list):
        for p_idx, p in enumerate(pieces_of(entry)):
            # The corpus segmenter puts a closing quote that ends one paragraph
            # at the start of the next sentence; give it back to its sentence.
            if CLOSING_PUNCT_ONLY.match(p) and pieces:
                pieces[-1]['raw'] = pieces[-1]['raw'].rstrip() + p.strip()
                continue
            pieces.append({'sent_idx': s_idx, 'piece_idx': p_idx, 'raw': p})
    kept, dropped = [], []
    for p in pieces:
        r = heading_reason(p['raw'])
        if r:
            dropped.append({**p, 'reason': r})
        else:
            kept.append(p)
    # Titles: short unpunctuated pieces before the first real prose item.
    while kept:
        c = clean(kept[0]['raw'])
        if is_title(c):
            dropped.append({**kept.pop(0), 'reason': 'title'})
        else:
            break
    # Excerpt boundaries: the excerpting cuts text mid-sentence.
    if kept and corpus == 'human':
        c = clean(kept[0]['raw'])
        if c[:1].islower():
            dropped.append({**kept.pop(0), 'reason': 'truncated-start'})
    if kept and not TERMINAL.search(clean(kept[-1]['raw'])):
        dropped.append({**kept.pop(), 'reason': 'truncated-end'})
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', default='llm_dataset_LL.zip')
    ap.add_argument('--out', default='data')
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'items').mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.zip)
    names = zf.namelist()
    stats = defaultdict(Counter)
    with open(out / 'dropped.tsv', 'w', newline='') as dfh:
        dw = csv.writer(dfh, delimiter='\t')
        dw.writerow(['corpus', 'excerpt_id', 'sent_idx', 'piece_idx', 'reason', 'raw'])
        for corpus, prefix in CORPORA.items():
            files = sorted(n for n in names
                           if n.startswith(prefix) and n.endswith('.json')
                           and '/' not in n[len(prefix):])
            with open(out / 'items' / f'{corpus}.jsonl', 'w') as ofh:
                for x_idx, fn in enumerate(files, start=1):
                    j = json.loads(zf.read(fn))
                    sl = j['sentence_list']
                    if isinstance(sl, str):
                        sl = ast.literal_eval(sl)
                    excerpt_id = Path(fn).stem
                    kept, dropped = process_excerpt(corpus, sl)
                    for d in dropped:
                        dw.writerow([corpus, excerpt_id, d['sent_idx'], d['piece_idx'],
                                     d['reason'], d['raw'].replace('\n', '\\n')])
                        stats[corpus]['dropped:' + d['reason']] += 1
                    for k, p in enumerate(kept, start=1):
                        text = clean(p['raw'])
                        rec = {
                            'i-id': CORPUS_IDX[corpus] * 10_000_000 + x_idx * 1_000 + k,
                            'corpus': corpus,
                            'excerpt_id': excerpt_id,
                            'sent_idx': p['sent_idx'],
                            'piece_idx': p['piece_idx'],
                            'raw': p['raw'],
                            'text': text,
                            'text_norm': normalize_punct(text),
                        }
                        assert k < 1000, 'item index overflows i-id scheme'
                        ofh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        stats[corpus]['items'] += 1
                        if p['piece_idx'] > 0 or len(pieces_of(sl[p['sent_idx']])) > 1:
                            stats[corpus]['from-para-split'] += 1
                    stats[corpus]['excerpts'] += 1
    for corpus, c in stats.items():
        print(corpus, dict(sorted(c.items())))


if __name__ == '__main__':
    main()

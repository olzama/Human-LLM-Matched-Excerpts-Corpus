#!/usr/bin/env python3
"""Build one empty [incr tsdb()] profile per excerpt from data/items/*.jsonl.

  profiles/skeletons/<corpus>/<excerpt_id>/

i-input is the punctuation-normalized text (`text_norm`); i-comment keeps
the position in the corpus's sentence_list as "<sent_idx>.<piece_idx>".

With --sample N, instead builds one pilot profile per corpus with N items
drawn evenly across sentence-length bins (profiles/pilot-skeletons/<corpus>).
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from delphin import tsdb

DEFAULT_RELATIONS = Path.home() / 'delphin/erg/erg-github-latest/tsdb/skeletons/Relations'
LENGTH_BINS = [(1, 10), (11, 20), (21, 30), (31, 45), (46, 60), (61, 80), (81, 120), (121, 10_000)]


def item_record(rec):
    text = rec['text_norm']
    return {
        'i-id': rec['i-id'],
        'i-origin': rec['corpus'],
        'i-register': rec['excerpt_id'],
        'i-format': 'none',
        'i-difficulty': 1,
        'i-category': 'S',
        'i-input': text,
        'i-wf': 1,
        'i-length': len(text.split()),
        'i-comment': f"{rec['sent_idx']}.{rec['piece_idx']}",
        'i-author': rec['corpus'],
    }


def write_profile(path, schema, records, overwrite):
    if path.exists():
        if not overwrite:
            raise SystemExit(f'{path} exists; use --overwrite')
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tsdb.initialize_database(path, schema, files=True)
    tsdb.write(path, 'item', [tsdb.make_record(item_record(r), schema['item']) for r in records],
               fields=schema['item'])


def length_bin(n):
    for lo, hi in LENGTH_BINS:
        if lo <= n <= hi:
            return (lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--items', default='data/items')
    ap.add_argument('--out', default='profiles')
    ap.add_argument('--relations', default=str(DEFAULT_RELATIONS))
    ap.add_argument('--sample', type=int, help='pilot: N items per corpus, stratified by length')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    schema = tsdb.read_schema(args.relations)
    for fn in sorted(Path(args.items).glob('*.jsonl')):
        corpus = fn.stem
        recs = [json.loads(line) for line in open(fn)]
        if args.sample:
            rng = random.Random(f'{args.seed}-{corpus}')
            by_bin = defaultdict(list)
            for r in recs:
                by_bin[length_bin(len(r['text_norm'].split()))].append(r)
            per_bin = args.sample // len(LENGTH_BINS)
            chosen = []
            for b in LENGTH_BINS:
                pool = by_bin.get(b, [])
                chosen += rng.sample(pool, min(per_bin, len(pool)))
            chosen.sort(key=lambda r: r['i-id'])
            write_profile(Path(args.out) / 'pilot-skeletons' / corpus, schema, chosen, args.overwrite)
            print(corpus, len(chosen), 'pilot items')
            continue
        by_excerpt = defaultdict(list)
        for r in recs:
            by_excerpt[r['excerpt_id']].append(r)
        for ex, rs in by_excerpt.items():
            write_profile(Path(args.out) / 'skeletons' / corpus / ex, schema, rs, args.overwrite)
        print(corpus, len(by_excerpt), 'profiles,', len(recs), 'items')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Equal-size samples for comparing subcorpora while parsing is incomplete.

1. Matched excerpts: keep only excerpts present in every included subcorpus
   (excerpts are matched across subcorpora by novel, so all keep the same works).
2. Sentences (per-sentence TSV from sentence_measures.py): within the matched
   excerpts, randomly sample each subcorpus's sentences down to the smallest
   subcorpus's number of sentences.
3. Type counts (counts-by-excerpt.jsonl from count_types.py): within the
   matched excerpts, thin each subcorpus's counts at random (binomial thinning,
   per phenomenon) so that every subcorpus has the same expected number of type
   tokens as the smallest one. Shares are unchanged in expectation; only the
   amount of data is equalized.

--exclude drops subcorpora (e.g. one that has barely been parsed yet). Writes
the equalized files plus equalize.json describing the sample.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

PHENOMENA = ('constr', 'lexrule', 'lextype', 'lexentries')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--counts', required=True)
    ap.add_argument('--sentences', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    recs = [r for r in (json.loads(l) for l in open(a.counts, encoding='utf-8')) if r['corpus'] not in a.exclude]
    rows = [r for r in csv.DictReader(open(a.sentences, encoding='utf-8'), delimiter='\t')
            if r['corpus'] not in a.exclude]
    corpora = sorted({r['corpus'] for r in recs})
    exc = defaultdict(set)
    for r in recs:
        exc[r['corpus']].add(r['excerpt'])
    for r in rows:
        exc[r['corpus'] + '#tsv'].add(r['excerpt'])
    matched = set.intersection(*(exc[c] & exc[c + '#tsv'] for c in corpora))

    # sentences
    by_c = defaultdict(list)
    for r in rows:
        if r['excerpt'] in matched:
            by_c[r['corpus']].append(r)
    n_sent = min(len(v) for v in by_c.values())
    pyrng = random.Random(a.seed)
    with open(a.sentences, encoding='utf-8') as fh:
        header = fh.readline().rstrip('\n').split('\t')
    with open(out / 'sentences.tsv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter='\t')
        w.writeheader()
        for c in corpora:
            for r in sorted(pyrng.sample(by_c[c], n_sent), key=lambda r: int(r['i_id'])):
                w.writerow(r)

    # type counts: binomial thinning to the smallest total per phenomenon
    kept = [r for r in recs if r['excerpt'] in matched]
    totals = {p: {c: sum(sum(r.get(p, {}).values()) for r in kept if r['corpus'] == c) for c in corpora}
              for p in PHENOMENA}
    target = {p: min(totals[p].values()) for p in PHENOMENA}
    with open(out / 'counts-by-excerpt.jsonl', 'w', encoding='utf-8') as fh:
        for r in kept:
            r = dict(r)
            for p in PHENOMENA:
                f = target[p] / totals[p][r['corpus']] if totals[p][r['corpus']] else 0
                thinned = {t: int(rng.binomial(v, f)) for t, v in r.get(p, {}).items()}
                r[p] = {t: v for t, v in thinned.items() if v}
            ld = {}
            f = target['lextype'] / totals['lextype'][r['corpus']] if totals['lextype'][r['corpus']] else 0
            for d, cnt in r.get('lextype_depth', {}).items():
                ld[d] = {t: int(rng.binomial(v, f)) for t, v in cnt.items()}
                ld[d] = {t: v for t, v in ld[d].items() if v}
            r['lextype_depth'] = ld
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    info = {'subcorpora': corpora, 'excluded': a.exclude, 'matched_excerpts': len(matched),
            'sentences_per_subcorpus': n_sent, 'type_tokens_per_subcorpus': target, 'seed': a.seed}
    json.dump(info, open(out / 'equalize.json', 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()

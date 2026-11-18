#!/usr/bin/env python3
"""Split the per-excerpt skeleton profiles by sentence length and allocate them.

  profiles/skeletons/<corpus>/<excerpt>   (all items)
    -> profiles/split/short/<machine>/<corpus>/<excerpt>   items with <= --max-short words
    -> profiles/split/long/<band>/<corpus>/<excerpt>       longer items, one tree per band

Bands are set with --long-bands as upper limits: '60,80' gives 46-60, 61-80
and 81+ (named '46-60', '61-80', '81-'). Each band starts at its own memory
setting, so no item is tried at a setting already known to fail for its length.

Short profiles are dealt out to machines in proportion to --weights (e.g.
mortimer=15,kesha=4: 10 jobs at 1.5x speed vs 4 jobs), balancing by item count.
Long profiles all go to the big-memory machine. --parts limits which parts are
(re)written, e.g. --parts long while the short part is being parsed. Every item lands in exactly
one output profile; `parse_tiered.py merge --skeletons` reassembles them.
"""
import argparse
import shutil
from pathlib import Path

from delphin import tsdb


def read_rows(path, table):
    with tsdb.open(path, table) as fh:
        return [tsdb.split(line.rstrip('\n')) for line in fh]


def write_profile(path, schema, rows):
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tsdb.initialize_database(path, schema, files=True)
    tsdb.write(path, 'item', rows, fields=schema['item'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='profiles/skeletons')
    ap.add_argument('--dst', default='profiles/split')
    ap.add_argument('--max-short', type=int, default=45)
    ap.add_argument('--weights', default='mortimer=15,kesha=4',
                    help='machine=weight pairs for the short profiles')
    ap.add_argument('--long-bands', default='60,80',
                    help='upper word limits of the long bands; the last band is open-ended')
    ap.add_argument('--parts', default='short,long', help='which parts to write: short, long or both')
    a = ap.parse_args()
    parts = set(a.parts.split(','))
    uppers = [int(x) for x in a.long_bands.split(',')] if a.long_bands else []
    bands, lo = [], a.max_short + 1
    for hi in uppers:
        bands.append((f'{lo}-{hi}', lo, hi)); lo = hi + 1
    bands.append((f'{lo}-', lo, 10 ** 9))

    weights = dict((k, float(v)) for k, v in (kv.split('=') for kv in a.weights.split(',')))
    src = Path(a.src)
    profs = sorted(p.parent for p in src.rglob('relations'))
    schema = tsdb.read_schema(profs[0])
    length_ix = [f.name for f in schema['item']].index('i-length')

    short = []
    long_counts = {name: [0, 0] for name, _, _ in bands}   # profiles, items
    for p in profs:
        rows = read_rows(p, 'item')
        rel = p.relative_to(src)
        s = [r for r in rows if int(r[length_ix]) <= a.max_short]
        if s:
            short.append((rel, s))
        if 'long' in parts:
            for name, lo, hi in bands:
                b = [r for r in rows if lo <= int(r[length_ix]) <= hi]
                if b:
                    write_profile(Path(a.dst) / 'long' / name / rel, schema, b)
                    long_counts[name][0] += 1
                    long_counts[name][1] += len(b)

    if 'short' in parts:
        # greedy: biggest profiles first, each to the machine furthest below its share
        load = {m: 0 for m in weights}
        total_w = sum(weights.values())
        for rel, rows in sorted(short, key=lambda x: -len(x[1])):
            m = min(load, key=lambda k: load[k] / weights[k])
            load[m] += len(rows)
            write_profile(Path(a.dst) / 'short' / m / rel, schema, rows)
        n_short = sum(load.values())
        for m, n in load.items():
            print(f'short -> {m}: {n} items ({100 * n / n_short:.0f}%, target {100 * weights[m] / total_w:.0f}%)')
    if 'long' in parts:
        for name, (np_, ni) in long_counts.items():
            print(f'long {name:>6} -> {np_} profiles, {ni} items')


if __name__ == '__main__':
    main()

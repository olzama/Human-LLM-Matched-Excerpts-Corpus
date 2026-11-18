#!/usr/bin/env python3
"""Split the per-excerpt skeleton profiles by sentence length and allocate them.

  profiles/skeletons/<corpus>/<excerpt>   (all items)
    -> profiles/split/short/<machine>/<corpus>/<excerpt>   items with <= --max-short words
    -> profiles/split/long/<corpus>/<excerpt>              items with  > --max-short words

Short profiles are dealt out to machines in proportion to --weights (e.g.
mortimer=15,kesha=4: 10 jobs at 1.5x speed vs 4 jobs), balancing by item count.
Long profiles all go to the big-memory machine. Every item lands in exactly
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
    a = ap.parse_args()

    weights = dict((k, float(v)) for k, v in (kv.split('=') for kv in a.weights.split(',')))
    src = Path(a.src)
    profs = sorted(p.parent for p in src.rglob('relations'))
    schema = tsdb.read_schema(profs[0])
    length_ix = [f.name for f in schema['item']].index('i-length')

    short, long_ = [], []
    for p in profs:
        rows = read_rows(p, 'item')
        s = [r for r in rows if int(r[length_ix]) <= a.max_short]
        l = [r for r in rows if int(r[length_ix]) > a.max_short]
        rel = p.relative_to(src)
        if s:
            short.append((rel, s))
        if l:
            write_profile(Path(a.dst) / 'long' / rel, schema, l)
            long_.append((rel, len(l)))

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
    print(f'long  -> {len(long_)} profiles, {sum(n for _, n in long_)} items')


if __name__ == '__main__':
    main()

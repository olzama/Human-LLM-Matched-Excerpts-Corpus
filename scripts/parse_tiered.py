#!/usr/bin/env python3
"""Parse profiles with ACE in resource tiers, then merge.

  run     Parse every profile under --src into --dst (same relative layout).
          With --retry, --src is the previous tier's output and only items
          that failed for lack of time or memory are copied and re-parsed.
  merge   Rebuild one profile per excerpt from the full skeletons, taking each
          item's result from the last of the given result roots (tiers,
          length splits, machines) that processed it; writes a provenance table.
  combine Concatenate the merged per-excerpt profiles of each corpus into
          one profile per corpus (ids are globally unique, so no renumbering).
  report  Coverage and failure types per corpus for one tier directory.

Profiles that are already complete in --dst are skipped, so `run` can be
restarted after an interruption.
"""
import argparse
import csv
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from delphin import ace, itsdb, tsdb

# Defaults are kesha's paths; other machines set ERG_ACE / ERG_GRAMMAR.
ACE = Path(os.environ.get('ERG_ACE', Path.home() / 'delphin/tools/ACE/ace-0.9.34/ace'))
GRAMMAR = Path(os.environ.get('ERG_GRAMMAR', Path.home() / 'delphin/erg/erg-github-latest/ace/erg-ace0934.dat'))
OOM_FIRST = Path(__file__).resolve().parent / 'ace-oom-first.sh'

# ACE error strings that mean "gave up for lack of resources", as opposed to
# "searched the whole space and found no analysis". Checked in the pilot.
RESOURCE_ERROR = re.compile(r'time|RAM|memory|limit|edges', re.I)


def profiles_under(root):
    return sorted(p.parent for p in Path(root).rglob('relations'))


def read_rows(path, table):
    """Raw rows of a (possibly gzipped) table as tuples of strings."""
    with tsdb.open(path, table) as fh:
        return [tsdb.split(line.rstrip('\n')) for line in fh]


def n_rows(path, table):
    try:
        return len(read_rows(path, table))
    except (tsdb.TSDBError, FileNotFoundError):
        return 0


def outcome(parse_row):
    """Classify one parse row: parsed / resource / crash / nogap."""
    readings = parse_row['readings']
    readings = -1 if readings in (None, '') else int(readings)   # 0 is a real value: no analysis
    err = parse_row['error'] or ''
    if readings > 0:
        return 'parsed'
    # ACE died mid-item (e.g. OOM-killed): pydelphin records 0 readings with
    # 'incomplete output from ACE' and restarts ACE for the next item.
    if readings < 0 or re.search(r'incomplete output|closed', err, re.I):
        return 'crash'
    if RESOURCE_ERROR.search(err):
        return 'resource'
    return 'no-parse'


def ace_options(a):
    packing = ['-p'] if a.no_packing else []   # -p: ACE parses without ambiguity packing
    return ['-1', *packing, '--rooted-derivation', '--udx',
            f'--max-words={a.max_words}', f'--timeout={a.timeout}',
            f'--max-chart-megabytes={a.chart_mb}', f'--max-unpack-megabytes={a.unpack_mb}']


def parse_one(src, dst, opts, retry, executable=ACE, retry_outcomes=('resource', 'crash'), max_length=None):
    """Copy (a subset of) src's items to dst and parse them. Returns (dst, n_items, secs)."""
    t0 = time.time()
    if dst.exists():
        shutil.rmtree(dst)
    schema = tsdb.read_schema(src)
    if retry:
        if n_rows(src, 'parse') != n_rows(src, 'item'):
            print(f'WARNING: retry source {src} is incomplete '
                  f'({n_rows(src, "parse")}/{n_rows(src, "item")} items parsed)', file=sys.stderr)
        ts = itsdb.TestSuite(src)
        fail_ids = {row['i-id'] for row in ts['parse'] if outcome(row) in retry_outcomes}
        len_ix = [f.name for f in schema['item']].index('i-length')
        items = [r for r in read_rows(src, 'item') if int(r[0]) in fail_ids
                 and (max_length is None or int(r[len_ix]) <= max_length)]
        if not items:
            return dst, 0, 0.0
    else:
        items = list(read_rows(src, 'item'))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tsdb.initialize_database(dst, schema, files=True)
    tsdb.write(dst, 'item', items, fields=schema['item'])
    with ace.ACEParser(GRAMMAR, cmdargs=opts, executable=executable, stderr=open('/dev/null', 'w')) as cpu:
        itsdb.TestSuite(dst).process(cpu, gzip=True, buffer_size=10)
    return dst, len(items), time.time() - t0


def cmd_run(a):
    opts = ace_options(a)
    print('ACE options:', ' '.join(opts), file=sys.stderr)
    src_root, dst_root = Path(a.src), Path(a.dst)
    jobs = []
    for src in profiles_under(src_root):
        dst = dst_root / src.relative_to(src_root)
        if not a.retry and dst.exists() and n_rows(dst, 'parse') == n_rows(src, 'item') > 0:
            continue
        if a.retry and (dst / '.done').exists():
            continue
        jobs.append((src, dst))
    print(f'{len(jobs)} profiles to process with {a.workers} workers', file=sys.stderr)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        exe = OOM_FIRST if a.oom_first else ACE
        outs = tuple(a.retry_outcomes.split(','))
        futs = {ex.submit(parse_one, s, d, opts, a.retry, exe, outs, a.max_length): d for s, d in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                dst, n, secs = f.result()
                if a.retry:
                    dst.mkdir(parents=True, exist_ok=True)
                    (dst / '.done').touch()
                print(f'[{done}/{len(jobs)}] {dst} {n} items {secs:.0f}s', file=sys.stderr, flush=True)
            except Exception as e:  # keep going; the profile is retried on restart
                print(f'[{done}/{len(jobs)}] FAILED {futs[f]}: {e!r}', file=sys.stderr, flush=True)


def tier_tables(path):
    ts = itsdb.TestSuite(path)
    parses = {row['i-id']: row for row in ts['parse']}
    results = defaultdict(list)
    for row in ts['result']:
        results[row['parse-id']].append(row)
    return ts, parses, results


def cmd_merge(a):
    """Reassemble one profile per excerpt from the full skeletons.

    Each --tiers root mirrors <corpus>/<excerpt> but may hold only some items
    (a length split, a retry subset, another machine). Roots are given in
    order of precedence, lowest first: an item takes its row from the LAST
    root that processed it. Items no root processed are reported as missing.
    """
    skel_root = Path(a.skeletons)
    tiers = [Path(t) for t in a.tiers]
    prov_rows = []
    missing = 0
    for prof in profiles_under(skel_root):
        rel = prof.relative_to(skel_root)
        dst = Path(a.dst) / rel
        if dst.exists():
            shutil.rmtree(dst)
        schema = tsdb.read_schema(prof)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tsdb.initialize_database(dst, schema, files=True)
        items = read_rows(prof, 'item')
        tsdb.write(dst, 'item', items, fields=schema['item'], gzip=True)
        final_parse, final_result, which, run_rows = {}, {}, {}, None
        for tier in tiers:
            p = tier / rel
            if not (p / 'relations').exists() or n_rows(p, 'parse') == 0:
                continue
            _, parses, results = tier_tables(p)
            if run_rows is None:
                run_rows = read_rows(p, 'run')
            for iid, row in parses.items():
                final_parse[iid] = row
                final_result[iid] = results.get(row['parse-id'], [])
                which[iid] = str(tier)
        prow, rrow = [], []
        for item in items:
            iid = int(item[0])
            if iid not in final_parse:
                missing += 1
                prov_rows.append([iid, rel.parts[0], rel.parts[-1], '', 'missing', '', '', ''])
                continue
            row = final_parse[iid]
            rec = {f.name: row[f.name] for f in schema['parse']}
            rec['parse-id'] = iid      # parse-id := i-id keeps ids unique across sources
            prow.append(tsdb.make_record(rec, schema['parse']))
            for r in final_result[iid]:
                rr = {f.name: r[f.name] for f in schema['result']}
                rr['parse-id'] = iid
                rrow.append(tsdb.make_record(rr, schema['result']))
            o = outcome(row)
            if o == 'parsed' and not final_result[iid]:
                o = 'incomplete'
                print(f'WARNING: {rel} item {iid} has readings but no result row', file=sys.stderr)
            prov_rows.append([iid, rel.parts[0], rel.parts[-1], which[iid],
                              o, row['readings'], row['total'], row['error'] or ''])
        tsdb.write(dst, 'parse', prow, fields=schema['parse'], gzip=True)
        tsdb.write(dst, 'result', rrow, fields=schema['result'], gzip=True)
        if run_rows:
            tsdb.write(dst, 'run', run_rows[:1], fields=schema['run'], gzip=True)
    with open(a.provenance, 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['i-id', 'corpus', 'excerpt_id', 'source', 'outcome', 'readings', 'total_ms', 'error'])
        w.writerows(prov_rows)
    print(f'merged {len(prov_rows)} items into {a.dst}; {missing} missing; provenance in {a.provenance}')


def cmd_combine(a):
    src_root, dst_root = Path(a.src), Path(a.dst)
    for corpus_dir in sorted(d for d in src_root.iterdir() if d.is_dir()):
        profs = profiles_under(corpus_dir)
        if not profs:
            continue
        schema = tsdb.read_schema(profs[0])
        dst = dst_root / corpus_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tsdb.initialize_database(dst, schema, files=True)
        counts = Counter()
        for table in ('item', 'parse', 'result'):
            rows, seen = [], set()
            for prof in profs:
                for r in read_rows(prof, table):
                    key = tuple(r[:2]) if table == 'result' else r[0]
                    if key in seen:
                        raise SystemExit(f'duplicate {table} key {key} in {prof}')
                    seen.add(key)
                    rows.append(r)
            tsdb.write(dst, table, rows, fields=schema[table], gzip=True)
            counts[table] = len(rows)
        tsdb.write(dst, 'run', read_rows(profs[0], 'run')[:1], fields=schema['run'], gzip=True)
        print(f"{corpus_dir.name}: {len(profs)} profiles -> {dst} "
              f"({counts['item']} items, {counts['parse']} parses, {counts['result']} results)")


def cmd_report(a):
    by_corpus = defaultdict(Counter)
    errors = Counter()
    for prof in profiles_under(a.dir):
        corpus = prof.relative_to(a.dir).parts[0]
        for row in itsdb.TestSuite(prof)['parse']:
            o = outcome(row)
            by_corpus[corpus][o] += 1
            if row['error']:
                errors[(o, re.sub(r'\d+', 'N', row['error'])[:100])] += 1
    print('corpus\titems\tparsed\tno-parse\tresource\tcrash\tcoverage')
    for c, cnt in sorted(by_corpus.items()):
        n = sum(cnt.values())
        print(f"{c}\t{n}\t{cnt['parsed']}\t{cnt['no-parse']}\t{cnt['resource']}\t{cnt['crash']}\t{cnt['parsed']/n:.3f}")
    print('\nerror strings (normalized):')
    for (o, e), n in errors.most_common(30):
        print(f'{n}\t{o}\t{e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--src', required=True)
    r.add_argument('--dst', required=True)
    r.add_argument('--retry', action='store_true')
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--timeout', type=int, default=60)
    r.add_argument('--chart-mb', type=int, default=4000)
    r.add_argument('--unpack-mb', type=int, default=4000)
    r.add_argument('--max-words', type=int, default=1000)
    r.add_argument('--oom-first', action='store_true',
                   help='start ACE via ace-oom-first.sh (oom_score_adj=1000)')
    r.add_argument('--max-length', type=int,
                   help='with --retry: only re-parse items of at most this many words')
    r.add_argument('--retry-outcomes', default='resource,crash',
                   help='with --retry: which outcomes to re-parse (comma-separated)')
    r.add_argument('--no-packing', action='store_true',
                   help='pass -p to ACE (intended for the last retry tier)')
    m = sub.add_parser('merge')
    m.add_argument('--skeletons', required=True, help='full item profiles: <root>/<corpus>/<excerpt>')
    m.add_argument('--tiers', nargs='+', required=True,
                   help='result roots mirroring the skeleton layout, lowest precedence first')
    m.add_argument('--dst', required=True)
    m.add_argument('--provenance', required=True)
    c = sub.add_parser('combine')
    c.add_argument('--src', required=True, help='merged root: <src>/<corpus>/<excerpt>/')
    c.add_argument('--dst', required=True, help='output root: <dst>/<corpus>/')
    rp = sub.add_parser('report')
    rp.add_argument('dir')
    a = ap.parse_args()
    {'run': cmd_run, 'merge': cmd_merge, 'combine': cmd_combine, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()

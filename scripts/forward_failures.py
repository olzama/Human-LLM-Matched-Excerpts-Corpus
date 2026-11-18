#!/usr/bin/env python3
"""Collect first-pass failures from the parsing machines and queue them for flash.

Every --interval seconds:
  1. pull each remote results tree (rsync) into a local copy;
  2. for every finished profile (parse rows == item rows) not handled before,
     write a profile holding only its items that failed for lack of memory or
     time or because ACE was killed (no-analysis items are not retried);
  3. push the new retry profiles to the queue directory on the big-memory host.

Handled profiles are listed in --state, so a restart never forwards twice.
Stops when --expect profiles have been handled (or after one pass with --once).

Example:
  forward_failures.py \\
    --pull mortimer:delphin/lenka/profiles/short-16g=profiles/results/short-16g-mortimer \\
    --local profiles/results/short-16g-kesha \\
    --queue profiles/forward/short-28g-src \\
    --push flash:data/delphin/lenka/profiles/short-28g-src \\
    --expect 1490
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from delphin import itsdb, tsdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_tiered import n_rows, outcome, profiles_under, read_rows  # noqa: E402

RETRY = ('resource', 'crash')


def log(msg):
    print(f'[{datetime.now():%Y-%m-%d %H:%M}] {msg}', flush=True)


def rsync(src, dst):
    Path(dst).mkdir(parents=True, exist_ok=True) if ':' not in dst else None
    r = subprocess.run(['rsync', '-a', '--timeout=300', src.rstrip('/') + '/', dst.rstrip('/') + '/'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f'rsync {src} -> {dst} failed: {r.stderr.strip()[:200]}')
    return r.returncode == 0


def build_retry(prof, dst):
    """Write dst with prof's failed items; return how many."""
    fail = {row['i-id'] for row in itsdb.TestSuite(prof)['parse'] if outcome(row) in RETRY}
    if not fail:
        return 0
    schema = tsdb.read_schema(prof)
    items = [r for r in read_rows(prof, 'item') if int(r[0]) in fail]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tsdb.initialize_database(dst, schema, files=True)
    tsdb.write(dst, 'item', items, fields=schema['item'])
    return len(items)


def one_pass(a, handled):
    roots = []
    for spec in a.pull:
        remote, local = spec.split('=', 1)
        rsync(remote, local)
        roots.append(Path(local))
    roots += [Path(p) for p in a.local]
    new_profiles = new_items = 0
    for root in roots:
        for prof in profiles_under(root):
            rel = str(prof.relative_to(root))
            if rel in handled:
                continue
            n_item = n_rows(prof, 'item')
            if n_item == 0 or n_rows(prof, 'parse') != n_item:
                continue  # still running (or incomplete); look again next pass
            n = build_retry(prof, Path(a.queue) / rel)
            handled.add(rel)
            with open(a.state, 'a') as fh:
                fh.write(rel + '\n')
            new_profiles += 1
            new_items += n
    if new_items and a.push:
        rsync(a.queue, a.push)
    log(f'handled {new_profiles} newly finished profiles, queued {new_items} failed items; '
        f'total handled {len(handled)}' + (f'/{a.expect}' if a.expect else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pull', nargs='*', default=[], help='REMOTE=LOCAL results trees to rsync and scan')
    ap.add_argument('--local', nargs='*', default=[], help='local results trees to scan')
    ap.add_argument('--queue', required=True, help='local tree of retry profiles')
    ap.add_argument('--push', help='rsync destination for the queue (e.g. flash:...)')
    ap.add_argument('--state', default='profiles/forward/forwarded.txt')
    ap.add_argument('--interval', type=int, default=1800)
    ap.add_argument('--expect', type=int, help='stop once this many profiles are handled')
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()

    Path(a.state).parent.mkdir(parents=True, exist_ok=True)
    handled = set(Path(a.state).read_text().split()) if Path(a.state).exists() else set()
    while True:
        one_pass(a, handled)
        if a.once or (a.expect and len(handled) >= a.expect):
            break
        time.sleep(a.interval)
    log('done')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Diversity of type distributions compared at equal sample size.

Shannon and Simpson indices grow with the amount of text they are computed
on, and the subcorpora differ in how many types they yield (sentence counts,
sentence length, coverage). This script compares them at the same number of
type tokens.

Each repetition:
  1. draws excerpt names with replacement from the excerpts all subcorpora
     share, and uses the same names for every subcorpus (a paired bootstrap:
     the excerpts are matched by novel, so every subcorpus sees the same works);
  2. pools each subcorpus's counts over the drawn excerpts;
  3. draws exactly N type tokens from each pool without replacement and
     computes the indices on that sample, averaging over --draws such draws.
     All pools in a repetition are drawn with the same random seed (common
     random numbers), so identical pools give identical samples and
     differences between subcorpora are not inflated by drawing noise.
Reported per subcorpus: the mean index and a 95% interval over repetitions.
Reported per LLM: the human-minus-LLM difference and its 95% interval. An
"all LLMs pooled" aggregate is included (all four LLMs' tokens pooled before
drawing N).

N defaults to --size-fraction (0.8) times the smallest subcorpus total, so
the bootstrap pools almost always hold at least N tokens; a repetition whose
pool is smaller than N is redrawn.

--unpaired resamples each subcorpus's own excerpts independently instead,
for data whose excerpts are not matched (e.g. the pilot sample).

Indices use the same definitions as diversity.py: Shannon H = -sum p ln p;
Simpson D = 1 - sum p^2. In addition, a diversity profile of Hill numbers
(effective numbers of types) at the orders given by --q:
  D_q = (sum p^q)^(1/(1-q)),  D_1 = exp(H).
q = 0 is richness (every type counts equally, however rare); as q grows the
profile is driven more and more by the frequent types (D_2 = 1/sum p^2, the
inverse of Simpson's concentration). Where two sources' profiles converge, they
differ only in their rarer types. --no-bootstrap keeps the full subcorpora and only
subsamples (plain rarefaction).

Input: counts-by-excerpt.jsonl from count_types.py. Outputs in --out:
diversity-equal-<phenomenon><suffix>.{md,json,png} with suffix '' (all
types), '_punct' or '_xpunct', and diversity-profile-<phenomenon><suffix>.png
(Hill numbers by q, and each LLM's profile relative to the human one).
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POOLED = 'LLMs (pooled)'


def is_punct_type(tname):
    """Same rule as diversity.py."""
    t = tname.lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


def load(path, phenom, punct):
    """corpus -> excerpt -> {type: count}, filtered by punctuation mode."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for line in open(path, encoding='utf-8'):
        rec = json.loads(line)
        for t, c in rec.get(phenom, {}).items():
            if punct == 'punct' and not is_punct_type(t):
                continue
            if punct == 'xpunct' and is_punct_type(t):
                continue
            data[rec['corpus']][rec['excerpt']][t] += c
    return data


def matrix(per_excerpt, excerpts, types):
    ix = {t: i for i, t in enumerate(types)}
    m = np.zeros((len(excerpts), len(types)), dtype=np.int64)
    for r, ex in enumerate(excerpts):
        for t, c in per_excerpt.get(ex, {}).items():
            m[r, ix[t]] = c
    return m


def hill(p, q):
    if q == 1:
        return float(np.exp(-(p * np.log(p)).sum()))
    if q == 0:
        return float(len(p))
    return float((p ** q).sum() ** (1.0 / (1.0 - q)))


def indices(counts, qs=()):
    n = counts.sum()
    p = counts[counts > 0] / n
    return (-(p * np.log(p)).sum(), 1.0 - (p * p).sum()) + tuple(hill(p, q) for q in qs)


def run(a, phenom, punct, sources, out):
    data = load(a.counts, phenom, punct)
    corpora = [c for c in sources if c in data]
    shared = sorted(set.intersection(*(set(data[c]) for c in corpora)))
    types = sorted({t for c in corpora for ex in data[c].values() for t in ex})
    if not types:
        print(f'{phenom}{punct}: no data'); return
    paired = not (a.unpaired or a.no_bootstrap)
    if paired and len(shared) < a.min_shared:
        raise SystemExit(f'only {len(shared)} excerpts are shared by all subcorpora; '
                         f'use --unpaired to resample each subcorpus\'s own excerpts')
    rows_of = {c: (shared if paired else sorted(data[c])) for c in corpora}
    mats = {c: matrix(data[c], rows_of[c], types) for c in corpora}
    llms = [c for c in corpora if sources[c]['group'] == 'llm']
    humans = [c for c in corpora if sources[c]['group'] == 'human']
    totals = {c: int(m.sum()) for c, m in mats.items()}
    n = a.size or int(a.size_fraction * min(totals.values()))
    if n < 2:
        print(f'{phenom}{punct}: too few tokens'); return

    rng = np.random.default_rng(a.seed)
    names = corpora + ([POOLED] if llms else [])
    sh = {k: [] for k in names}
    si = {k: [] for k in names}
    hq = {q: {k: [] for k in names} for q in a.q}
    reps = 0
    while reps < a.reps:
        if a.no_bootstrap:
            pools = {c: mats[c].sum(axis=0) for c in corpora}
        elif a.unpaired:
            pools = {c: mats[c][rng.integers(0, len(rows_of[c]), len(rows_of[c]))].sum(axis=0) for c in corpora}
        else:
            rows = rng.integers(0, len(shared), len(shared))
            pools = {c: mats[c][rows].sum(axis=0) for c in corpora}
        if llms:
            pools[POOLED] = sum(pools[c] for c in llms)
        if any(p.sum() < n for p in pools.values()):
            continue                                   # rare: a bootstrap pool smaller than N
        # Common random numbers: every pool is subsampled with the same seed, so
        # identical pools give identical samples and differences carry less noise.
        seed = int(rng.integers(2 ** 63))
        for k, pool in pools.items():
            draw_rng = np.random.default_rng(seed)
            vals = [indices(draw_rng.multivariate_hypergeometric(pool, n), a.q) for _ in range(a.draws)]
            sh[k].append(float(np.mean([v[0] for v in vals])))
            si[k].append(float(np.mean([v[1] for v in vals])))
            for j, q in enumerate(a.q):
                hq[q][k].append(float(np.mean([v[2 + j] for v in vals])))
        reps += 1

    ci = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    design = ('rarefaction only' if a.no_bootstrap else
              'unpaired bootstrap over each subcorpus\'s excerpts' if a.unpaired else
              f'paired bootstrap over {len(shared)} shared excerpts')
    result = {'phenomenon': phenom, 'punct': punct, 'sample_size': n, 'reps': a.reps, 'draws': a.draws,
              'design': design, 'shared_excerpts': len(shared),
              'tokens_available': totals, 'indices': {}, 'differences_vs_human': {}}
    for idx_name, store in (('shannon', sh), ('simpson', si)):
        result['indices'][idx_name] = {k: ci(store[k]) for k in names}
        if humans:
            h = np.array(store[humans[0]])
            result['differences_vs_human'][idx_name] = {
                k: ci(h - np.array(store[k])) for k in names if k not in humans}

    result['hill'] = {str(q): {k: ci(hq[q][k]) for k in names} for q in a.q}
    if humans:
        # ratio of effective numbers, human / source: 1.0 = equally diverse at this order
        result['hill_ratio_human_over'] = {str(q): {k: ci(np.array(hq[q][humans[0]]) / np.array(hq[q][k]))
                                                     for k in names if k not in humans} for q in a.q}
    suffix = {'all': '', 'punct': '_punct', 'xpunct': '_xpunct'}[punct]
    stem = out / f'diversity-equal-{phenom}{suffix}'
    json.dump(result, open(f'{stem}.json', 'w'), indent=1)
    label = lambda k: sources[k]['label'] if k in sources else k
    with open(f'{stem}.md', 'w') as fh:
        fh.write(f'{phenom}{suffix}: {n:,} type tokens per sample, {a.reps} repetitions, {design}\n\n')
        fh.write('| Source | Tokens available | Shannon (95% interval) | Simpson (95% interval) |\n|---|---:|---|---|\n')
        for k in names:
            s, d = result['indices']['shannon'][k], result['indices']['simpson'][k]
            avail = totals.get(k, sum(totals[c] for c in llms))
            fh.write(f'| {label(k)} | {avail:,} | {s[0]:.3f} ({s[1]:.3f}–{s[2]:.3f}) | {d[0]:.4f} ({d[1]:.4f}–{d[2]:.4f}) |\n')
        if humans:
            fh.write(f'\n| Human minus | Shannon difference (95% interval) | Simpson difference (95% interval) |\n|---|---|---|\n')
            for k in result['differences_vs_human']['shannon']:
                s, d = result['differences_vs_human']['shannon'][k], result['differences_vs_human']['simpson'][k]
                fh.write(f'| {label(k)} | {s[0]:+.3f} ({s[1]:+.3f} to {s[2]:+.3f}) | {d[0]:+.4f} ({d[1]:+.4f} to {d[2]:+.4f}) |\n')
        fh.write(f'\nDiversity profile: Hill numbers (effective number of types; q = 0 is richness, '
                 f'q = 1 is exp(Shannon), q = 2 is 1/sum p²), mean over repetitions\n\n')
        fh.write('| Source | ' + ' | '.join(f'q = {q:g}' for q in a.q) + ' |\n|---|' + '---:|' * len(a.q) + '\n')
        for k in names:
            fh.write(f'| {label(k)} | ' + ' | '.join(f"{result['hill'][str(q)][k][0]:,.1f}" for q in a.q) + ' |\n')
        if humans:
            fh.write('\n| Human / source (95% interval) | ' + ' | '.join(f'q = {q:g}' for q in a.q)
                     + ' |\n|---|' + '---|' * len(a.q) + '\n')
            for k in result['hill_ratio_human_over'][str(a.q[0])]:
                fh.write(f'| {label(k)} | ' + ' | '.join(
                    '{:.3f} ({:.3f}–{:.3f})'.format(*result['hill_ratio_human_over'][str(q)][k]) for q in a.q) + ' |\n')
    plot(result, names, sources, label, f'{stem}.png', phenom, suffix)
    plot_profile(result, names, sources, label, str(out / f'diversity-profile-{phenom}{suffix}.png'), phenom, suffix,
                 a.q, a.scope)
    print(f'{phenom}{suffix}: N={n:,}; wrote {stem}.md/.json/.png')


def plot(result, names, sources, label, path, phenom, suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = {'constr': 'Constructions', 'lextype': 'Lexical types', 'lexrule': 'Lexical rules',
             'lexentries': 'Lexical entries'}.get(phenom, phenom)
    title += {'': '', '_punct': ' (punctuation only)', '_xpunct': ' (excluding punctuation)'}[suffix]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.6))
    for ax, idx in zip(axes, ('shannon', 'simpson')):
        vals = result['indices'][idx]
        order = sorted(names, key=lambda k: vals[k][0])
        for y, k in enumerate(order):
            m, lo, hi = vals[k]
            human = k in sources and sources[k]['group'] == 'human'
            color = sources[k]['color'] if k in sources else '#52514e'
            ax.plot([lo, hi], [y, y], color=color, linewidth=2)
            ax.plot(m, y, marker='*' if human else 'o', markersize=11 if human else 8, color=color)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([label(k) for k in order], fontsize=9)
        ax.set_xlabel(f"{idx.capitalize()} index (mean and 95% interval)", fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
    fig.suptitle(f"{title}: diversity at {result['sample_size']:,} type tokens per sample", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_profile(result, names, sources, label, path, phenom, suffix, qs, scope):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = {'constr': 'Constructions', 'lextype': 'Lexical types', 'lexrule': 'Lexical rules',
             'lexentries': 'Lexical entries'}.get(phenom, phenom)
    title += {'': '', '_punct': ' (punctuation only)', '_xpunct': ' (punctuation excluded)'}[suffix]
    srcs = [k for k in names if k in sources]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for k in srcs:
        human = sources[k]['group'] == 'human'
        m = [result['hill'][str(q)][k] for q in qs]
        ax.plot(qs, [v[0] for v in m], marker='*' if human else 'o', markersize=9 if human else 5,
                color=sources[k]['color'], linewidth=2, label=label(k))
        ax.fill_between(qs, [v[1] for v in m], [v[2] for v in m], color=sources[k]['color'], alpha=0.15, linewidth=0)
    ax.set_yscale('log')
    ax.set_xlabel('order q (0 = every type counts equally; higher = frequent types dominate)', fontsize=9)
    ax.set_ylabel('effective number of types (log scale)', fontsize=9)
    ax.set_title('Diversity profile', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    if 'hill_ratio_human_over' in result:
        ax.axhline(1, color='#52514e', linewidth=0.9, linestyle='--')
        for k in [k for k in srcs if sources[k]['group'] != 'human']:
            m = [result['hill_ratio_human_over'][str(q)][k] for q in qs]
            ax.plot(qs, [v[0] for v in m], marker='o', markersize=5, color=sources[k]['color'], linewidth=2,
                    label=label(k))
            ax.fill_between(qs, [v[1] for v in m], [v[2] for v in m], color=sources[k]['color'], alpha=0.18,
                            linewidth=0)
        ax.set_xlabel('order q', fontsize=9)
        ax.set_ylabel('human ÷ model effective number of types', fontsize=9)
        ax.set_title('How much more diverse the human texts are, by order (1 = equal)', fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(linestyle='--', alpha=0.3); ax.set_xticks(qs)
    fig.suptitle(f"{title}: diversity profile at {result['sample_size']:,} type tokens per sample "
                 f"(95% intervals: paired bootstrap)", fontsize=11)
    if scope:
        fig.text(0.01, 0.005, scope, fontsize=7.5, color='#5f5d58')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('counts', help='counts-by-excerpt.jsonl from count_types.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources', default=str(HERE / 'sources.json'))
    ap.add_argument('--phenomena', nargs='+', default=['constr', 'lextype'],
                    choices=['constr', 'lexrule', 'lextype', 'lexentries'])
    ap.add_argument('--split-punct', action='store_true', help='also run punctuation-only and without-punctuation')
    ap.add_argument('--size', type=int, help='type tokens per sample (default: --size-fraction × smallest total)')
    ap.add_argument('--size-fraction', type=float, default=0.8)
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--draws', type=int, default=5,
                    help='subsamples of N tokens averaged within each repetition')
    ap.add_argument('--no-bootstrap', action='store_true', help='subsample only, without resampling excerpts')
    ap.add_argument('--unpaired', action='store_true',
                    help="resample each subcorpus's own excerpts independently (when excerpts are not matched)")
    ap.add_argument('--min-shared', type=int, default=50,
                    help='refuse a paired bootstrap on fewer shared excerpts than this')
    ap.add_argument('--q', type=float, nargs='+', default=[0, 0.5, 1, 1.5, 2, 3],
                    help='orders of the Hill-number diversity profile')
    ap.add_argument('--scope', default='', help='note printed under the profile plot')
    ap.add_argument('--seed', type=int, default=1337)
    a = ap.parse_args()
    sources = {k: v for k, v in json.load(open(a.sources)).items() if not k.startswith('_')}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for phenom in a.phenomena:
        for punct in (['all', 'punct', 'xpunct'] if a.split_punct else ['all']):
            run(a, phenom, punct, sources, out)


if __name__ == '__main__':
    main()

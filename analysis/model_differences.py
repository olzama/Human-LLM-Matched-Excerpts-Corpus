#!/usr/bin/env python3
"""Where do the subcorpora differ in their use of constructions (or lexical types)?

Two views, from counts-by-excerpt.jsonl (count_types.py):

1. Distinctive types (distinctive-<phenom>.png, .tsv)
   For each type, each subcorpus's share of all its types of that kind, relative
   to the consensus (the unweighted mean of the subcorpus shares), as a log2
   ratio: 0 = typical rate, +1 = twice the consensus, -1 = half. Types are ranked
   by how much the subcorpora disagree (standard deviation of their log2 ratios)
   and the top --top are shown, one dot per subcorpus. Intervals are 95%
   bootstrap intervals from resampling each subcorpus's excerpts (--reps).
   Only types with at least --min-share of all tokens in every subcorpus are
   ranked, so rare types do not dominate by noise.

2. Excerpt map (excerpt-map-<phenom>.png)
   Each excerpt with at least --min-tokens type tokens is a point, placed by a
   principal component analysis of its type profile (square roots of the shares,
   i.e. Hellinger-transformed composition) over the types that pass
   --min-share overall. Points are colored by subcorpus; large markers are the
   subcorpus centroids; arrows show the types with the largest loadings.

Every plot states the number of excerpts per subcorpus it rests on.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = '#1d1c1a', '#5f5d58', '#dedbd2'


def is_punct_type(tname):
    """Same rule as diversity.py (llm-syntax): punctuation constructions, types and rules."""
    t = tname.lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


def keep_type(t, punct):
    return punct == 'all' or (punct == 'xpunct') != is_punct_type(t)


def style(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('counts', help='counts-by-excerpt.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources', default=str(HERE / 'sources.json'))
    ap.add_argument('--phenomena', nargs='+', default=['constr', 'lextype'])
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--min-share', type=float, default=0.0005, help='minimum share of a type (0.0005 = 0.05%%)')
    ap.add_argument('--min-tokens', type=int, default=300, help='minimum type tokens for an excerpt on the map')
    ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--arrows', type=int, default=6)
    ap.add_argument('--scope', default='TEST: partial data, sentences of up to 45 words only')
    ap.add_argument('--punct', choices=['all', 'xpunct', 'punct'], default='all',
                    help='all types, punctuation excluded, or punctuation only (shares recomputed)')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = {k: v for k, v in json.load(open(a.sources)).items() if not k.startswith('_')}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    recs = [json.loads(l) for l in open(a.counts, encoding='utf-8')]

    suffix = {'all': '', 'xpunct': '_xpunct', 'punct': '_punct'}[a.punct]
    titlesuffix = {'all': '', 'xpunct': ' (punctuation excluded)', 'punct': ' (punctuation only)'}[a.punct]
    for phenom in a.phenomena:
        per_exc = defaultdict(Counter)                       # (corpus, excerpt) -> type counts
        for r in recs:
            per_exc[(r['corpus'], r['excerpt'])].update({t: v for t, v in r.get(phenom, {}).items()
                                                         if keep_type(t, a.punct)})
        corpora = [c for c in sources if any(k[0] == c for k in per_exc)]
        exc_of = {c: [k for k in per_exc if k[0] == c] for c in corpora}
        lab = lambda c: f"{sources[c]['label']} ({len(exc_of[c])} excerpts)"
        totals = {c: sum((per_exc[k] for k in exc_of[c]), Counter()) for c in corpora}
        share = {c: {t: v / sum(totals[c].values()) for t, v in totals[c].items()} for c in corpora}
        types = sorted(t for t in set().union(*[set(s) for s in share.values()])
                       if all(share[c].get(t, 0) >= a.min_share for c in corpora))
        if not types:
            continue
        ix = {t: i for i, t in enumerate(types)}

        # matrices: excerpts x types counts, per corpus
        mats = {}
        for c in corpora:
            m = np.zeros((len(exc_of[c]), len(types)))
            tot = np.zeros(len(exc_of[c]))
            for i, k in enumerate(exc_of[c]):
                tot[i] = sum(per_exc[k].values())
                for t, v in per_exc[k].items():
                    if t in ix:
                        m[i, ix[t]] = v
            mats[c] = (m, tot)

        def log_ratios(shares):                               # corpus -> vector of shares over types
            S = np.maximum(np.vstack([shares[c] for c in corpora]), 1e-7)   # a type absent from a resample
            cons = S.mean(axis=0)
            return np.log2(S / cons)

        point = log_ratios({c: mats[c][0].sum(0) / mats[c][1].sum() for c in corpora})
        boots = []
        for _ in range(a.reps):
            bs = {}
            for c in corpora:
                m, tot = mats[c]
                rows = rng.integers(0, len(tot), len(tot))
                bs[c] = m[rows].sum(0) / tot[rows].sum()
            boots.append(log_ratios(bs))
        boots = np.array(boots)                               # reps x corpora x types
        lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        spread = point.std(axis=0)
        top = np.argsort(spread)[::-1][:a.top]

        with open(out / f'distinctive-{phenom}{suffix}.tsv', 'w') as fh:
            fh.write('type\tspread\t' + '\t'.join(f'{c}_log2ratio\t{c}_lo\t{c}_hi\t{c}_share' for c in corpora) + '\n')
            for j in np.argsort(spread)[::-1]:
                fh.write(f'{types[j]}\t{spread[j]:.4f}\t' + '\t'.join(
                    f'{point[i, j]:.4f}\t{lo[i, j]:.4f}\t{hi[i, j]:.4f}\t{share[c].get(types[j], 0):.6f}'
                    for i, c in enumerate(corpora)) + '\n')

        order = top[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 2.0))
        offs = np.linspace(-0.28, 0.28, len(corpora))
        for i, c in enumerate(corpora):
            y = np.arange(len(order)) + offs[i]
            ax.hlines(y, lo[i, order], hi[i, order], color=sources[c]['color'], linewidth=1.2, alpha=0.6)
            ax.scatter(point[i, order], y, s=26, color=sources[c]['color'], zorder=3,
                       marker='*' if sources[c]['group'] == 'human' else 'o', label=lab(c))
        ax.axvline(0, color=MUTED, linewidth=1, linestyle='--')
        ax.set_yticks(range(len(order))); ax.set_yticklabels([types[j] for j in order], fontsize=7.5, color=INK)
        ax.set_xlabel('log₂ ratio to the mean of the sources (0 = typical; +1 = twice; −1 = half)', fontsize=9, color=INK)
        title = {'constr': 'Constructions', 'lextype': 'Lexical types'}.get(phenom, phenom) + titlesuffix
        ax.set_title(f'{title} on which the sources differ most (95% intervals over excerpts)', fontsize=10.5, color=INK)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.06 - 1.2 / len(order) ** 1.2),
                  ncol=2)
        fig.text(0.01, 0.005, a.scope + f'; types with ≥ {100 * a.min_share:.2g}% share in every source', fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / f'distinctive-{phenom}{suffix}.png', dpi=150,
                                                            bbox_inches='tight'); plt.close(fig)

        # excerpt map
        X, who = [], []
        for c in corpora:
            m, tot = mats[c]
            keep = tot >= a.min_tokens
            X.append(np.sqrt(m[keep] / tot[keep, None])); who += [c] * int(keep.sum())
        X = np.vstack(X); who = np.array(who)
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pcs = Xc @ Vt[:2].T
        var = (S ** 2) / (S ** 2).sum()
        fig, ax = plt.subplots(figsize=(8.5, 7))
        for c in corpora:
            sel = who == c
            ax.scatter(pcs[sel, 0], pcs[sel, 1], s=10, alpha=0.35, color=sources[c]['color'], linewidths=0,
                       label=f"{sources[c]['label']} ({int(sel.sum())} excerpts)")
        for c in corpora:
            sel = who == c
            ax.scatter(pcs[sel, 0].mean(), pcs[sel, 1].mean(), s=180, color=sources[c]['color'], edgecolors=INK,
                       linewidths=1.2, marker='*' if sources[c]['group'] == 'human' else 'o', zorder=4)
        load = Vt[:2].T
        strongest = np.argsort((load ** 2).sum(1))[::-1][:a.arrows]
        # scale arrows so their tips stay inside the data range on both axes
        xr, yr = np.abs(pcs[:, 0]).max(), np.abs(pcs[:, 1]).max()
        lx, ly = np.abs(load[strongest, 0]).max(), np.abs(load[strongest, 1]).max()
        scale = 0.7 * min(xr / lx if lx else np.inf, yr / ly if ly else np.inf)
        for j in strongest:
            ax.annotate('', xy=(load[j, 0] * scale, load[j, 1] * scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=MUTED, lw=1))
            ax.text(load[j, 0] * scale * 1.06, load[j, 1] * scale * 1.06, types[j], fontsize=7.5, color=INK,
                    ha='center', va='center', clip_on=True,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))
        ax.set_xlabel(f'PC1 ({100 * var[0]:.0f}% of variance)', fontsize=9, color=INK)
        ax.set_ylabel(f'PC2 ({100 * var[1]:.0f}% of variance)', fontsize=9, color=INK)
        ax.set_title(f'{title}: excerpts placed by their profile (large markers = source centers)', fontsize=10.5, color=INK)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, markerscale=2)
        fig.text(0.01, 0.005, a.scope + f'; excerpts with ≥ {a.min_tokens} tokens', fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / f'excerpt-map-{phenom}{suffix}.png', dpi=150,
                                                            bbox_inches='tight'); plt.close(fig)
        print(f'{phenom}: {len(types)} types ranked; map of {len(who)} excerpts; '
              f'PC1 {100 * var[0]:.0f}%, PC2 {100 * var[1]:.0f}%')


if __name__ == '__main__':
    main()

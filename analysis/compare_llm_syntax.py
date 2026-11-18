#!/usr/bin/env python3
"""Compare human-vs-LLM differences in this corpus with those in llm-syntax's
2025 news data (Gude et al., ACL 2026, "More aligned, less diverse?").

News data: llm-syntax analysis/frequencies-json/frequencies-2025.json
({phenomenon: {model: {type: count}}}); human = NYT-2025-human, LLMs = the six
2025 models of the paper (TinyLlama excluded). Literary data: an equal-size
counts-by-excerpt.jsonl from equalize.py.

For each corpus, punctuation excluded (llm-syntax rule), shares are computed per
source; the LLM rate of a type is the unweighted mean of the LLM shares, and
its difference from the human rate is log2(LLM / human). Outputs:
  compare-<phenom>.tsv       per type: news and literary log2 ratios and shares
  compare-<phenom>.png       scatter of news vs literary log2 ratios, with the
                             types that differ most labelled
  compare-diversity.md       Shannon/Simpson at one common sample size for all
                             sources of both corpora (multivariate
                             hypergeometric subsampling), human minus each LLM
  compare-summary.json       Spearman correlations and agreement counts
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
NEWS_HUMAN = 'NYT-2025-human'
NEWS_LLMS = ['Qwen14B-2025-llm', 'Qwen32B-2025-llm', 'Qwen72B-2025-llm', 'Llama70B-2025-llm',
             'Mistral7B_i-2025-llm', 'GPT4o-2025-llm']
NEWS_LABEL = {'NYT-2025-human': 'NYT 2025 (human)', 'Qwen14B-2025-llm': 'Qwen 2.5 14B',
              'Qwen32B-2025-llm': 'Qwen 2.5 32B', 'Qwen72B-2025-llm': 'Qwen 2.5 72B',
              'Llama70B-2025-llm': 'Llama 3.3 70B', 'Mistral7B_i-2025-llm': 'Mistral 7B v0.3',
              'GPT4o-2025-llm': 'GPT-4o'}
INK, MUTED, GRID = '#1d1c1a', '#5f5d58', '#dedbd2'


def is_punct_type(t):
    t = t.lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


def shares(counter):
    c = {t: v for t, v in counter.items() if not is_punct_type(t)}
    n = sum(c.values())
    return {t: v / n for t, v in c.items()}, c


def rarefied(counts, n, rng, draws):
    keys = sorted(counts)
    pool = np.array([counts[k] for k in keys], dtype=np.int64)
    hs, ds = [], []
    for _ in range(draws):
        s = rng.multivariate_hypergeometric(pool, n)
        p = s[s > 0] / n
        hs.append(-(p * np.log(p)).sum()); ds.append(1 - (p * p).sum())
    return np.mean(hs), np.percentile(hs, [2.5, 97.5]), np.mean(ds), np.percentile(ds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--news', required=True, help='llm-syntax frequencies-2025.json')
    ap.add_argument('--literary', required=True, help='equal-size counts-by-excerpt.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources', default=str(HERE / 'sources.json'))
    ap.add_argument('--min-share', type=float, default=0.0005)
    ap.add_argument('--label', type=int, default=14)
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--scope', default='')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = {k: v for k, v in json.load(open(a.sources)).items() if not k.startswith('_')}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    news = json.load(open(a.news))
    lit = defaultdict(lambda: defaultdict(Counter))
    for line in open(a.literary, encoding='utf-8'):
        r = json.loads(line)
        for p in ('constr', 'lextype'):
            lit[p][r['corpus']].update(r.get(p, {}))
    lit_human = 'human'
    lit_llms = [c for c in sources if sources[c]['group'] == 'llm' and c in lit['constr']]
    summary = {'news_llms': NEWS_LLMS, 'literary_llms': lit_llms}
    rng = np.random.default_rng(3)
    div_md = []

    for phenom, title in (('constr', 'Constructions'), ('lextype', 'Lexical types')):
        ns = {m: shares(Counter(news[phenom][m]))[0] for m in [NEWS_HUMAN] + NEWS_LLMS}
        ls = {m: shares(lit[phenom][m])[0] for m in [lit_human] + lit_llms}
        types = sorted(set(ns[NEWS_HUMAN]) & set(ls[lit_human]))
        mean = lambda d, ms, t: float(np.mean([d[m].get(t, 0.0) for m in ms]))
        rows = []
        for t in types:
            nh, nl = ns[NEWS_HUMAN].get(t, 0), mean(ns, NEWS_LLMS, t)
            lh, ll = ls[lit_human].get(t, 0), mean(ls, lit_llms, t)
            if min(nh, nl, lh, ll) < a.min_share:
                continue
            per_model = {m: np.log2(max(ls[m].get(t, 0), 1e-7) / lh) for m in lit_llms}
            rows.append((t, np.log2(nl / nh), np.log2(ll / lh), nh, nl, lh, ll, per_model))
        x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
        rho, p = spearmanr(x, y)
        same = int(((x > 0.3) & (y > 0.3)).sum() + ((x < -0.3) & (y < -0.3)).sum())
        opp = int(((x > 0.3) & (y < -0.3)).sum() + ((x < -0.3) & (y > 0.3)).sum())
        per_model_rho = {m: float(spearmanr(x, [r[7][m] for r in rows])[0]) for m in lit_llms}
        summary[phenom] = {'types_compared': len(rows), 'spearman_rho': float(rho), 'p_value': float(p),
                           'both_beyond_0.3_same_direction': same, 'both_beyond_0.3_opposite_direction': opp,
                           'spearman_per_literary_model': per_model_rho}
        with open(out / f'compare-{phenom}.tsv', 'w') as fh:
            fh.write('type\tnews_log2_llm_vs_human\tliterary_log2_llm_vs_human\tnews_human_share\tnews_llm_share\t'
                     'literary_human_share\tliterary_llm_share\t' + '\t'.join(f'literary_log2_{m}' for m in lit_llms) + '\n')
            for r in sorted(rows, key=lambda r: -(abs(r[1]) + abs(r[2]))):
                fh.write(f'{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t'
                         + '\t'.join(f'{r[7][m]:.4f}' for m in lit_llms) + '\n')

        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        lim = max(1.0, np.abs(np.concatenate([x, y])).max() * 1.08)
        ax.fill_between([0, lim], 0, lim, color='#2a78d6', alpha=0.06, linewidth=0)
        ax.fill_between([-lim, 0], -lim, 0, color='#2a78d6', alpha=0.06, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8); ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.scatter(x, y, s=18, color='#2a78d6', alpha=0.75, linewidths=0, zorder=3)
        order = np.argsort(-(np.abs(x) + np.abs(y)))[:a.label]
        placed = []
        for i in order:
            # stagger labels whose points are close, so they do not overlap
            near = sum(1 for (px, py) in placed if abs(px - x[i]) < 0.35 and abs(py - y[i]) < 0.08)
            ax.annotate(rows[i][0], (x[i], y[i]), fontsize=7, color=INK, xytext=(4, 3 - 10 * near),
                        textcoords='offset points')
            placed.append((x[i], y[i]))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('news: log₂(2025 LLMs / NYT 2025 human)', fontsize=9, color=INK)
        ax.set_ylabel('novels: log₂(LLMs / human authors)', fontsize=9, color=INK)
        ax.set_title(f'{title}: LLM-vs-human differences in news and in novels\n'
                     f'Spearman ρ = {rho:.2f} over {len(rows)} types; shaded = same direction', fontsize=10.5, color=INK)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, linewidth=0.5)
        fig.text(0.01, 0.005, (a.scope + '; ' if a.scope else '') + 'punctuation excluded; LLM rate = mean of the models',
                 fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / f'compare-{phenom}.png', dpi=150); plt.close(fig)

        # diversity at equal size within each corpus
        div_md.append(f'\n### {title} (punctuation excluded)\n')
        news_groups = {m: shares(Counter(news[phenom][m]))[1] for m in [NEWS_HUMAN] + NEWS_LLMS}
        lit_groups = {m: shares(lit[phenom][m])[1] for m in [lit_human] + lit_llms}
        n_common = min(sum(c.values()) for c in list(news_groups.values()) + list(lit_groups.values()))
        for corpus_name, groups, label in (
                ('News 2025 (llm-syntax)', {m: shares(Counter(news[phenom][m]))[1] for m in [NEWS_HUMAN] + NEWS_LLMS},
                 lambda m: NEWS_LABEL[m]),
                ('Novels (this corpus, equal-size sample)', {m: shares(lit[phenom][m])[1] for m in [lit_human] + lit_llms},
                 lambda m: sources[m]['label'])):
            n = n_common          # same sample size in both corpora, so the gaps are comparable
            res = {m: rarefied(c, n, rng, a.draws) for m, c in groups.items()}
            human = NEWS_HUMAN if NEWS_HUMAN in groups else lit_human
            div_md.append(f'\n**{corpus_name}**, {n:,} tokens per sample\n\n| Source | Shannon | Simpson | Shannon: human minus source |\n|---|---:|---:|---:|\n')
            for m in sorted(groups, key=lambda m: -res[m][0]):
                d = res[human][0] - res[m][0]
                div_md.append(f'| {label(m)} | {res[m][0]:.3f} | {res[m][2]:.4f} | {"—" if m == human else f"{d:+.3f}"} |\n')
    with open(out / 'compare-diversity.md', 'w') as fh:
        fh.write(''.join(div_md))
    json.dump(summary, open(out / 'compare-summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print(open(out / 'compare-diversity.md').read())


if __name__ == '__main__':
    main()

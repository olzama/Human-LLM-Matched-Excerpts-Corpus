#!/usr/bin/env python3
"""Menzerath-Altmann law per excerpt, at several levels, as in Lenka's thesis.

The fitting is ported unchanged from the thesis (appendix C.9): for each text
(here: excerpt), observations (x = construct length in constituents, y = mean
constituent length) are grouped by x; Spearman's rho is computed on the grouped
means, and the reduced law y = a x^-b is fitted by linear regression on the
logs (b, R^2, RMSE). A negative rho (positive b) supports the law.

Levels, from the per-sentence table of sentence_measures.py:
  sentence-word-tok   sentence length in thesis tokens -> mean syllables per token
                      (the thesis's sentence-level test)
  sentence-word-lex   the same with lexical items as words
  sentence-clause     sentence length in clauses -> mean clause length in words
  clause-word         clause length in words -> mean syllables per word in the clause

Outputs in --out: mal-by-excerpt.tsv, mal-summary.md, and plots
mal-rho-by-level.png (per-excerpt rho per level and subcorpus) and
mal-grouped-<level>.png (pooled grouped means per subcorpus). Every plot states
the number of excerpts and observations it rests on.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
LEVELS = [('sentence-word-tok', 'sentence → word (thesis tokens, syllables)'),
          ('sentence-word-lex', 'sentence → word (lexical items, syllables)'),
          ('sentence-clause', 'sentence → clause (words per clause)'),
          ('clause-word', 'clause → word (syllables per word)')]
AXES = {'sentence-word-tok': ('sentence length in tokens', 'mean syllables per token'),
        'sentence-word-lex': ('sentence length in words', 'mean syllables per word'),
        'sentence-clause': ('sentence length in clauses', 'mean clause length in words'),
        'clause-word': ('clause length in words', 'mean syllables per word')}


# ---- ported from the thesis (appendix C.9), unchanged in substance ----
def build_grouped_mal_table(observations):
    grouped = {}
    for x, y_raw in observations:
        x = int(x)
        grouped.setdefault(x, []).append(float(y_raw))
    x_values = range(min(grouped.keys()), max(grouped.keys()) + 1)
    rows = []
    for x in x_values:
        y_values = grouped.get(x, [])
        rows.append({"x": float(x), "n_x": int(len(y_values)),
                     "y_observed": float(np.mean(y_values)) if y_values else float("nan")})
    return rows


def fit_mal(observations):
    result = {"b": float("nan"), "r_squared": float("nan"), "rmse": float("nan"),
              "rho": float("nan"), "p_value": float("nan"), "n_observations": len(observations), "n_groups": 0}
    if not observations:
        return result
    grouped_rows = [r for r in build_grouped_mal_table(observations) if not math.isnan(r["y_observed"])]
    result["n_groups"] = len(grouped_rows)
    if len(grouped_rows) < 3:
        return result
    x = np.array([r["x"] for r in grouped_rows], dtype=float)
    y = np.array([r["y_observed"] for r in grouped_rows], dtype=float)
    rho, p_value = spearmanr(x, y)
    result["rho"], result["p_value"] = float(rho), float(p_value)
    if np.any(x <= 0) or np.any(y <= 0):
        return result
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred = slope * log_x + intercept
    res = log_y - pred
    tss = np.sum((log_y - np.mean(log_y)) ** 2)
    result.update({"b": float(-slope), "r_squared": float(1 - np.sum(res ** 2) / tss) if tss > 0 else float("nan"),
                   "rmse": float(math.sqrt(np.mean(res ** 2)))})
    return result
# ----------------------------------------------------------------------


def floats(s):
    return [float(x) for x in s.split(',')] if s else []


def observations(rows):
    obs = defaultdict(list)
    for r in rows:
        if r['parsed'] != '1':
            continue
        if r['n_tokens'] and r['mean_tok_syll']:
            obs['sentence-word-tok'].append((int(r['n_tokens']), float(r['mean_tok_syll'])))
        if r['n_words'] and r['mean_word_syll']:
            obs['sentence-word-lex'].append((int(r['n_words']), float(r['mean_word_syll'])))
        if r['n_clauses'] and int(r['n_clauses']) > 0:
            obs['sentence-clause'].append((int(r['n_clauses']), float(r['mean_clause_len'])))
            for n, sy in zip(floats(r['clause_lengths']), floats(r.get('clause_syllables', ''))):
                obs['clause-word'].append((int(n), sy))
    return obs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sentences', required=True, help='per-sentence TSV from sentence_measures.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources', default=str(HERE / 'sources.json'))
    ap.add_argument('--scope', default='TEST: partial data, sentences of up to 45 words only')
    ap.add_argument('--min-sentences', type=int, default=20,
                    help='skip excerpts with fewer parsed sentences than this')
    a = ap.parse_args()

    sources = {k: v for k, v in json.load(open(a.sources)).items() if not k.startswith('_')}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    by_exc = defaultdict(list)
    for r in csv.DictReader(open(a.sentences, encoding='utf-8'), delimiter='\t'):
        by_exc[(r['corpus'], r['excerpt'])].append(r)

    results = []
    pooled = defaultdict(lambda: defaultdict(list))
    for (c, ex), rows in sorted(by_exc.items()):
        if sum(r['parsed'] == '1' for r in rows) < a.min_sentences:
            continue
        obs = observations(rows)
        for lev, _ in LEVELS:
            res = fit_mal(obs.get(lev, []))
            results.append({'corpus': c, 'excerpt': ex, 'level': lev, **res})
            pooled[lev][c].extend(obs.get(lev, []))

    cols = ['corpus', 'excerpt', 'level', 'rho', 'p_value', 'b', 'r_squared', 'rmse', 'n_observations', 'n_groups']
    with open(out / 'mal-by-excerpt.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t'); w.writerow(cols)
        for r in results:
            w.writerow([r[k] for k in cols])

    corpora = [c for c in sources if any(r['corpus'] == c for r in results)]
    lab = lambda c: sources[c]['label']
    nexc = {c: len({r['excerpt'] for r in results if r['corpus'] == c}) for c in corpora}
    with open(out / 'mal-summary.md', 'w') as fh:
        fh.write(f"{a.scope}. Excerpts with at least {a.min_sentences} parsed sentences.\n\n"
                 "| Level | Source | Excerpts | Mean ρ | Share of excerpts with ρ < 0 | Mean b | Mean R² |\n"
                 "|---|---|---:|---:|---:|---:|---:|\n")
        for lev, title in LEVELS:
            for c in corpora:
                rs = [r for r in results if r['corpus'] == c and r['level'] == lev and not math.isnan(r['rho'])]
                if not rs:
                    continue
                rho = np.array([r['rho'] for r in rs]); bs = np.array([r['b'] for r in rs if not math.isnan(r['b'])])
                r2 = np.array([r['r_squared'] for r in rs if not math.isnan(r['r_squared'])])
                fh.write(f"| {title} | {lab(c)} | {len(rs)} | {rho.mean():+.3f} | {100 * (rho < 0).mean():.0f}% | "
                         f"{bs.mean():+.3f} | {r2.mean():.3f} |\n")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = '#1d1c1a', '#5f5d58', '#dedbd2'

    def style(ax):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # per-excerpt rho by level and subcorpus
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(4.0 * len(LEVELS), 0.45 * len(corpora) + 2.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (lev, title) in zip(axes, LEVELS):
        for y, c in enumerate(corpora):
            rho = np.array([r['rho'] for r in results if r['corpus'] == c and r['level'] == lev
                            and not math.isnan(r['rho'])])
            if not len(rho):
                continue
            ax.scatter(rho, y + rng.uniform(-0.18, 0.18, len(rho)), s=6, alpha=0.35, color=sources[c]['color'],
                       linewidths=0)
            ax.plot([np.median(rho)] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=2)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle='--')
        ax.set_xlim(-1.05, 1.05); ax.set_title(title, fontsize=9, color=INK); ax.set_xlabel("Spearman's ρ per excerpt", fontsize=8)
        style(ax)
    axes[0].set_yticks(range(len(corpora)))
    axes[0].set_yticklabels([f"{lab(c)} ({nexc[c]} excerpts)" for c in corpora], fontsize=8, color=INK)
    fig.suptitle('Menzerath–Altmann per excerpt: ρ < 0 supports the law (bar = median)', fontsize=11, color=INK)
    fig.text(0.01, 0.005, a.scope, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out / 'mal-rho-by-level.png', dpi=150); plt.close(fig)

    # pooled grouped means per level
    for lev, title in LEVELS:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for c in corpora:
            g = defaultdict(list)
            for x, y in pooled[lev][c]:
                g[int(x)].append(y)
            xs = [k for k in sorted(g) if len(g[k]) >= 20]
            if not xs:
                continue
            ax.plot(xs, [np.mean(g[k]) for k in xs], marker='o', markersize=3, linewidth=2, color=sources[c]['color'],
                    label=f"{lab(c)} ({nexc[c]} excerpts, {sum(len(g[k]) for k in xs):,} observations)")
        xl, yl = AXES[lev]
        ax.set_xlabel(xl, fontsize=9, color=INK); ax.set_ylabel(yl, fontsize=9, color=INK)
        ax.set_title(f'Menzerath–Altmann, {title}: pooled means (groups with ≥ 20 observations)', fontsize=10, color=INK)
        style(ax); ax.legend(fontsize=7.5, frameon=False)
        fig.text(0.01, 0.005, a.scope, fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / f'mal-grouped-{lev}.png', dpi=150); plt.close(fig)
    print(open(out / 'mal-summary.md').read())


if __name__ == '__main__':
    main()

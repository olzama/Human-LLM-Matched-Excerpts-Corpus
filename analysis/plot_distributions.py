#!/usr/bin/env python3
"""Distribution plots for checking results while parsing is still under way.

Every plot states how much of the data it is based on, along its own grouping:
per subcorpus, completed excerpts (of all excerpts) and parsed sentences (of all
sentences); per work, the same per novel. Denominators come from the full item
lists (data/items/*.jsonl from scripts/prepare_items.py), so partial data is
always visible as such.

Inputs: counts-by-excerpt.jsonl (count_types.py), the per-sentence TSV
(sentence_measures.py), data/items. Output: PNGs and a processing table.

--punct xpunct excludes punctuation from the construction and lexical-type
plots (shares recomputed over the remaining types), as in llm-syntax.

Plots:
  processing-by-work.png        share of each work's sentences processed, per subcorpus
  rank-frequency-constr.png     construction rank-frequency (log-log)
  word-classes.png              share of word classes (lexical types grouped by prefix)
  top-constr.png / top-lextype.png
                                relative frequency of the most frequent types
  clauses-per-sentence.png      distribution of the number of clauses per sentence
  mal-sentence-clause.png       mean clause length by sentence length in clauses
                                (the Menzerath-Altmann relation at that level)
  clauses-by-work.png           mean clauses per sentence by work and subcorpus
  syllables-per-word.png        distribution of syllables per word, lexical items and thesis tokens
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = '#1d1c1a', '#5f5d58', '#dedbd2'
PUNCT_TITLE = {'all': '', 'xpunct': ' (punctuation excluded)', 'punct': ' (punctuation only)'}


def load_sources(path):
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith('_')}


def totals_from_items(items_dir):
    """Per corpus and per (corpus, work): number of sentences and excerpts."""
    sent, exc = Counter(), defaultdict(set)
    wsent, wexc = Counter(), defaultdict(set)
    for f in Path(items_dir).glob('*.jsonl'):
        for line in open(f, encoding='utf-8'):
            r = json.loads(line)
            c, ex = r['corpus'], r['excerpt_id']
            w = ex.rsplit('_', 1)[0]
            sent[c] += 1; exc[c].add(ex)
            wsent[(c, w)] += 1; wexc[(c, w)].add(ex)
    return sent, {c: len(v) for c, v in exc.items()}, wsent, {k: len(v) for k, v in wexc.items()}


def is_punct_type(tname):
    """Same rule as diversity.py (llm-syntax)."""
    t = tname.lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


def keep_type(t, punct):
    return punct == 'all' or (punct == 'xpunct') != is_punct_type(t)


def style(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def note(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7.5, color=MUTED, ha='left', va='bottom', wrap=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--counts', required=True, help='counts-by-excerpt.jsonl')
    ap.add_argument('--sentences', required=True, help='per-sentence TSV from sentence_measures.py')
    ap.add_argument('--items', required=True, help='data/items directory (denominators)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources', default=str(HERE / 'sources.json'))
    ap.add_argument('--scope', default='TEST: partial data, sentences of up to 45 words only',
                    help='label printed on every plot')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--punct', choices=['all', 'xpunct', 'punct'], default='all',
                    help='type-based plots over all types, excluding punctuation, or punctuation only')
    a = ap.parse_args()

    sources = load_sources(a.sources)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tot_sent, tot_exc, wtot_sent, wtot_exc = totals_from_items(a.items)

    # what has been processed, from the counts file
    recs = [json.loads(l) for l in open(a.counts, encoding='utf-8')]
    done_exc, done_sent, done_parsed = defaultdict(set), Counter(), Counter()
    wdone_exc, wdone_sent = defaultdict(set), Counter()
    freq = {p: defaultdict(Counter) for p in ('constr', 'lextype')}
    for r in recs:
        c, w = r['corpus'], r['work']
        done_exc[c].add(r['excerpt']); done_sent[c] += r['sentences']; done_parsed[c] += r['parsed']
        wdone_exc[(c, w)].add(r['excerpt']); wdone_sent[(c, w)] += r['sentences']
        for p in freq:
            freq[p][c].update({t: v for t, v in r.get(p, {}).items() if keep_type(t, a.punct)})
    corpora = [c for c in sources if c in tot_sent]
    present = [c for c in corpora if done_parsed[c] > 0]

    def label(c, short=False):
        s = sources[c]['label']
        if short:
            return s
        return (f"{s} — {len(done_exc[c])}/{tot_exc[c]} excerpts, {done_parsed[c]:,} parsed "
                f"of {tot_sent[c]:,} sentences ({100 * done_parsed[c] / tot_sent[c]:.0f}%)")

    # processing table
    with open(out / 'processing.md', 'w') as fh:
        fh.write(f'{a.scope}\n\n| Subcorpus | Excerpts done | Sentences in done excerpts | Parsed | All sentences | Parsed / all |\n'
                 '|---|---:|---:|---:|---:|---:|\n')
        for c in corpora:
            fh.write(f"| {sources[c]['label']} | {len(done_exc[c])}/{tot_exc[c]} | {done_sent[c]:,} | "
                     f"{done_parsed[c]:,} | {tot_sent[c]:,} | {100 * done_parsed[c] / tot_sent[c]:.1f}% |\n")

    # 1. processing by work
    works = sorted({w for (_, w) in wtot_sent})
    mat = np.array([[wdone_sent[(c, w)] / wtot_sent[(c, w)] if wtot_sent[(c, w)] else np.nan
                     for c in corpora] for w in works])
    fig, ax = plt.subplots(figsize=(6.5, 0.17 * len(works) + 1.6))
    im = ax.imshow(mat, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(corpora))); ax.set_xticklabels([label(c, True) for c in corpora], rotation=30,
                                                            ha='right', fontsize=8, color=INK)
    ax.set_yticks(range(len(works))); ax.set_yticklabels(works, fontsize=6.5, color=INK)
    for i in range(len(works)):
        for j, c in enumerate(corpora):
            n_done, n_all = len(wdone_exc[(c, works[i])]), wtot_exc.get((c, works[i]), 0)
            ax.text(j, i, f'{n_done}/{n_all}', ha='center', va='center', fontsize=5.5,
                    color='white' if mat[i, j] > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('share of sentences processed', fontsize=8)
    ax.set_title('Processing by work (cells: excerpts done / excerpts)', fontsize=10, color=INK)
    note(fig, a.scope)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / 'processing-by-work.png', dpi=150); plt.close(fig)

    # 2. rank-frequency of constructions
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for c in present:
        v = np.array(sorted(freq['constr'][c].values(), reverse=True), dtype=float)
        ax.loglog(np.arange(1, len(v) + 1), v / v.sum(), color=sources[c]['color'], linewidth=2, label=label(c))
    ax.set_xlabel('rank', fontsize=9, color=INK); ax.set_ylabel('relative frequency', fontsize=9, color=INK)
    ax.set_title('Constructions: rank-frequency' + PUNCT_TITLE[a.punct], fontsize=11, color=INK)
    style(ax); ax.legend(fontsize=7, frameon=False, loc='lower left')
    note(fig, a.scope)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'rank-frequency-constr.png', dpi=150); plt.close(fig)

    # 3. top types, relative frequency per subcorpus
    for p, title in (('constr', 'Constructions'), ('lextype', 'Lexical types')):
        pooled = Counter()
        rel = {}
        for c in present:
            n = sum(freq[p][c].values())
            rel[c] = {t: v / n for t, v in freq[p][c].items()}
            pooled.update(rel[c])
        top = [t for t, _ in pooled.most_common(a.top)][::-1]
        fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(top) + 1.8))
        offs = np.linspace(-0.3, 0.3, len(present)) if len(present) > 1 else [0]
        for k, c in enumerate(present):
            ax.scatter([100 * rel[c].get(t, 0) for t in top], np.arange(len(top)) + offs[k], s=22,
                       color=sources[c]['color'], label=label(c), zorder=3)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7.5, color=INK)
        ax.set_xlabel(f'% of all {title.lower()} in the subcorpus', fontsize=9, color=INK)
        ax.set_title(f'{title}: the {a.top} most frequent overall' + PUNCT_TITLE[a.punct], fontsize=11, color=INK)
        style(ax); ax.legend(fontsize=7, frameon=False, loc='lower right')
        note(fig, a.scope)
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / f'top-{p}.png', dpi=150); plt.close(fig)

    # 3b. lexical types grouped into word classes by their ERG prefix
    def word_class(t):
        if t.startswith('n_-_pr'):
            return 'pronoun'
        if t.startswith('n_-_pn') or t.startswith('n_np_pn'):
            return 'proper name'
        for pre, name in (('n_', 'noun'), ('v_', 'verb'), ('aj_', 'adjective'), ('av_', 'adverb'),
                          ('p_', 'preposition'), ('pp_', 'preposition'), ('d_', 'determiner'),
                          ('c_', 'conjunction'), ('cm_', 'complementizer'), ('pt_', 'punctuation')):
            if t.startswith(pre):
                return name
        return 'other'
    classes = ['noun', 'proper name', 'pronoun', 'verb', 'adjective', 'adverb', 'preposition', 'determiner',
               'conjunction', 'complementizer', 'punctuation', 'other']
    wc = {}
    for c in present:
        cnt = Counter()
        for t, v in freq['lextype'][c].items():
            cnt[word_class(t)] += v
        n = sum(cnt.values())
        wc[c] = {k: 100 * cnt[k] / n for k in classes}
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(classes) + 1.8))
    order = sorted(classes, key=lambda k: np.mean([wc[c][k] for c in present]))
    offs = np.linspace(-0.25, 0.25, len(present)) if len(present) > 1 else [0]
    for k, c in enumerate(present):
        ax.scatter([wc[c][x] for x in order], np.arange(len(order)) + offs[k], s=30, color=sources[c]['color'],
                   label=label(c), zorder=3, marker='*' if sources[c]['group'] == 'human' else 'o')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9, color=INK)
    ax.set_xlabel('% of all words in the subcorpus', fontsize=9, color=INK)
    ax.set_title('Word classes (lexical types grouped by their ERG prefix)', fontsize=11, color=INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1)
    note(fig, a.scope)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / 'word-classes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    with open(out / 'word-classes.tsv', 'w') as fh:
        fh.write('class\t' + '\t'.join(present) + '\n')
        for k in classes:
            fh.write(k + '\t' + '\t'.join(f'{wc[c][k]:.3f}' for c in present) + '\n')

    # sentence-level measures
    rows = [r for r in csv.DictReader(open(a.sentences, encoding='utf-8'), delimiter='\t') if r['parsed'] == '1']
    by_c = defaultdict(list)
    for r in rows:
        by_c[r['corpus']].append(r)

    # 4. clauses per sentence
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for c in present:
        cnt = Counter(int(r['n_clauses']) for r in by_c[c])
        n = sum(cnt.values()); xs = range(0, 11)
        ax.plot(list(xs), [100 * cnt.get(x, 0) / n for x in xs], marker='o', markersize=4, linewidth=2,
                color=sources[c]['color'], label=f"{label(c, True)} ({n:,} parsed sentences)")
    ax.set_xlabel('clauses per sentence', fontsize=9, color=INK); ax.set_ylabel('% of parsed sentences', fontsize=9, color=INK)
    ax.set_title('Clauses per sentence (finite clauses, coordination counted once)', fontsize=10, color=INK)
    style(ax); ax.legend(fontsize=7.5, frameon=False)
    note(fig, a.scope)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'clauses-per-sentence.png', dpi=150); plt.close(fig)

    # 5. Menzerath-Altmann at sentence -> clause level (grouped means, as in the thesis)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for c in present:
        g = defaultdict(list)
        for r in by_c[c]:
            k = int(r['n_clauses'])
            if k > 0:
                g[k].append(float(r['mean_clause_len']))
        xs = [k for k in sorted(g) if len(g[k]) >= 20]
        ax.plot(xs, [np.mean(g[k]) for k in xs], marker='o', markersize=4, linewidth=2, color=sources[c]['color'],
                label=f"{label(c, True)} ({sum(len(g[k]) for k in xs):,} sentences)")
    ax.set_xlabel('sentence length in clauses', fontsize=9, color=INK)
    ax.set_ylabel('mean clause length in words', fontsize=9, color=INK)
    ax.set_title('Menzerath–Altmann: sentence → clause (groups with ≥ 20 sentences)', fontsize=10, color=INK)
    style(ax); ax.legend(fontsize=7.5, frameon=False)
    note(fig, a.scope + '; sentences over 65 words are not parsed, which caps clause length for sentences with many clauses')
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'mal-sentence-clause.png', dpi=150); plt.close(fig)

    # 6. mean clauses per sentence by work
    wv = defaultdict(list)
    for r in rows:
        wv[(r['corpus'], r['work'])].append(int(r['n_clauses']))
    mat = np.array([[np.mean(wv[(c, w)]) if wv.get((c, w)) else np.nan for c in present] for w in works])
    fig, ax = plt.subplots(figsize=(6.5, 0.17 * len(works) + 1.6))
    im = ax.imshow(mat, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(present))); ax.set_xticklabels([label(c, True) for c in present], rotation=30,
                                                            ha='right', fontsize=8, color=INK)
    ax.set_yticks(range(len(works))); ax.set_yticklabels(works, fontsize=6.5, color=INK)
    vmax = np.nanmax(mat) if np.isfinite(mat).any() else 1
    for i, w in enumerate(works):
        for j, c in enumerate(present):
            n = len(wv.get((c, w), []))
            if n:
                ax.text(j, i, f'{mat[i, j]:.1f} (n={n})', ha='center', va='center', fontsize=5,
                        color='white' if mat[i, j] > 0.6 * vmax else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('mean clauses per sentence', fontsize=8)
    ax.set_title('Clauses per sentence by work (n = parsed sentences)', fontsize=10, color=INK)
    note(fig, a.scope)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out / 'clauses-by-work.png', dpi=150); plt.close(fig)

    # 7. syllables per word, for both word definitions
    if rows and 'word_syllables' in rows[0]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        for ax, col, title in ((axes[0], 'word_syllables', 'lexical items'), (axes[1], 'tok_syllables', 'thesis tokens')):
            for c in present:
                cnt = Counter()
                for r in by_c[c]:
                    for x in (r.get(col) or '').split(','):
                        if x:
                            cnt[min(int(x), 5)] += 1
                n = sum(cnt.values())
                if not n:
                    continue
                ax.plot(range(1, 6), [100 * cnt.get(k, 0) / n for k in range(1, 6)], marker='o', markersize=4,
                        linewidth=2, color=sources[c]['color'], label=f"{label(c, True)} ({n:,} words)")
            ax.set_xticks(range(1, 6)); ax.set_xticklabels(['1', '2', '3', '4', '5+'])
            ax.set_xlabel('syllables per word', fontsize=9, color=INK)
            ax.set_title(f'Syllables per word: {title}', fontsize=10, color=INK)
            style(ax); ax.legend(fontsize=7.5, frameon=False)
        axes[0].set_ylabel('% of words', fontsize=9, color=INK)
        note(fig, a.scope + '; syllables counted with the CMU dictionary as in the thesis')
        fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / 'syllables-per-word.png', dpi=150); plt.close(fig)

    print(open(out / 'processing.md').read())
    print('plots in', out)


if __name__ == '__main__':
    main()

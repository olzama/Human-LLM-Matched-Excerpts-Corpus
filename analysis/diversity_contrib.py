#!/usr/bin/env python3
"""Which types make the human texts more diverse than each model?

Shannon's index is a sum over types, H = sum_t -p_t ln p_t, so the gap between
the human texts and a model splits exactly into one contribution per type:

    gap_t = (-p_h ln p_h) - (-p_m ln p_m),     sum_t gap_t = H_human - H_model

A type adds to the human lead when the human texts use it more than the model
(for every share below 1/e, which covers all types here); a type the model
uses more adds to the model's side. Contributions are summed by frequency band
(bands by the mean share over all sources, so no source defines them) to show
where in the distribution the gap lies, and the types that contribute most are
listed per band.

Input: an equal-size counts-by-excerpt.jsonl (equalize.py), so all sources
have the same number of type tokens; shares are plug-in estimates from the
pooled counts. 95% intervals come from a paired bootstrap over the matched
excerpts (the same excerpt names drawn for every source).

Outputs in --out, per phenomenon and punctuation mode (suffix '', '_xpunct',
'_punct'):
  diversity-contrib-<phenom><suffix>.tsv   per type and model: shares, log2
                                           ratio, contribution to the gap
                                           with 95% interval
  diversity-contrib-<phenom><suffix>.md    gap by frequency band; richness
                                           (types found in one source only);
                                           top contributors in the lower bands
  diversity-contrib-<phenom><suffix>.png   gap by band per model, and the
                                           top contributors among rare types
                                           (below --plot-max-share)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANDS = [(1e-2, 1.0, 'frequent (≥ 1%)'), (1e-3, 1e-2, 'mid (0.1–1%)'),
         (1e-4, 1e-3, 'rare (0.01–0.1%)'), (0.0, 1e-4, 'very rare (< 0.01%)')]


def is_punct_type(t):
    t = t.lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


def keep(t, punct):
    return punct == 'all' or (punct == 'xpunct') != is_punct_type(t)


def plogp(m):
    """-p ln p per cell of a share matrix, 0 where p = 0."""
    out = np.zeros_like(m, dtype=float)
    nz = m > 0
    out[nz] = -m[nz] * np.log(m[nz])
    return out


def run(a, phenom, punct, sources, out):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for line in open(a.counts, encoding='utf-8'):
        r = json.loads(line)
        for t, c in r.get(phenom, {}).items():
            if keep(t, punct):
                data[r['corpus']][r['excerpt']][t] += c
    corpora = [c for c in sources if c in data]
    human = next(c for c in corpora if sources[c]['group'] == 'human')
    llms = [c for c in corpora if sources[c]['group'] == 'llm']
    shared = sorted(set.intersection(*(set(data[c]) for c in corpora)))
    types = sorted({t for c in corpora for ex in data[c].values() for t in ex})
    ix = {t: i for i, t in enumerate(types)}
    mats = {}
    for c in corpora:
        m = np.zeros((len(shared), len(types)))
        for r, ex in enumerate(shared):
            for t, v in data[c][ex].items():
                m[r, ix[t]] = v
        mats[c] = m

    def shares(rows=None):
        return {c: (mats[c] if rows is None else mats[c][rows]).sum(axis=0) /
                   (mats[c] if rows is None else mats[c][rows]).sum() for c in corpora}

    P = shares()
    cons = np.mean([P[c] for c in corpora], axis=0)
    band_of = np.array([next(i for i, (lo, hi, _) in enumerate(BANDS) if lo <= s < hi or (hi == 1.0 and s >= lo))
                        for s in cons])
    gap = {m: plogp(P[human]) - plogp(P[m]) for m in llms}

    rng = np.random.default_rng(a.seed)
    boot = {m: [] for m in llms}
    for _ in range(a.reps):
        rows = rng.integers(0, len(shared), len(shared))
        Pb = shares(rows)
        for m in llms:
            boot[m].append(plogp(Pb[human]) - plogp(Pb[m]))
    boot = {m: np.array(v) for m, v in boot.items()}          # reps × types
    pct = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

    suffix = {'all': '', 'punct': '_punct', 'xpunct': '_xpunct'}[punct]
    stem = out / f'diversity-contrib-{phenom}{suffix}'
    label = lambda c: sources[c]['label']
    band_res = {m: [] for m in llms}
    for m in llms:
        for b in range(len(BANDS)):
            sel = band_of == b
            band_res[m].append((float(gap[m][sel].sum()),) + pct(boot[m][:, sel].sum(axis=1)))

    with open(f'{stem}.tsv', 'w') as fh:
        fh.write('type\tband\tmean_share\thuman_share\tmodel\tmodel_share\tlog2_human_over_model\t'
                 'gap_contribution\tlo\thi\tshare_of_total_gap\n')
        for m in llms:
            tot = gap[m].sum()
            for i in np.argsort(-gap[m]):
                lo, hi = pct(boot[m][:, i])
                l2 = np.log2(max(P[human][i], 1e-9) / max(P[m][i], 1e-9))
                fh.write(f'{types[i]}\t{BANDS[band_of[i]][2]}\t{cons[i]:.6g}\t{P[human][i]:.6g}\t{m}\t{P[m][i]:.6g}\t'
                         f'{l2:.3f}\t{gap[m][i]:.6f}\t{lo:.6f}\t{hi:.6f}\t{gap[m][i] / tot:.4f}\n')

    md = [f'{phenom}{suffix}: Shannon gap (human minus model) split by type; {len(shared)} matched excerpts, '
          f'{a.reps} paired bootstrap repetitions; bands by mean share over all sources\n']
    md.append('\n| Frequency band | Types | ' + ' | '.join(label(m) for m in llms) + ' |\n|---|---:|' + '---|' * len(llms) + '\n')
    for b, (_, _, name) in enumerate(BANDS):
        md.append(f'| {name} | {int((band_of == b).sum())} | ' + ' | '.join(
            '{:+.3f} ({:+.3f} to {:+.3f})'.format(*band_res[m][b]) for m in llms) + ' |\n')
    md.append('| **All** | ' + str(len(types)) + ' | ' + ' | '.join(f'**{gap[m].sum():+.3f}**' for m in llms) + ' |\n')

    md.append('\nTypes found in one source and not the other (equal-size samples)\n\n'
              '| Model | Human only | Model only | Human-only types\' share of human tokens |\n|---|---:|---:|---:|\n')
    for m in llms:
        ho = (P[human] > 0) & (P[m] == 0); mo = (P[m] > 0) & (P[human] == 0)
        md.append(f'| {label(m)} | {int(ho.sum())} | {int(mo.sum())} | {100 * P[human][ho].sum():.3f}% |\n')

    for b in range(1, len(BANDS)):
        md.append(f'\nLargest contributions in band "{BANDS[b][2]}" (positive = adds to the human lead; '
                  f'shares in %)\n\n| Model | Type | Human share | Model share | log2 human/model | Contribution (95% interval) |\n'
                  '|---|---|---:|---:|---:|---|\n')
        for m in llms:
            sel = np.where(band_of == b)[0]
            for i in sel[np.argsort(-np.abs(gap[m][sel]))][:a.top]:
                lo, hi = pct(boot[m][:, i])
                l2 = np.log2(max(P[human][i], 1e-9) / max(P[m][i], 1e-9))
                md.append(f'| {label(m)} | `{types[i]}` | {100 * P[human][i]:.3f} | {100 * P[m][i]:.3f} | {l2:+.2f} | '
                          f'{gap[m][i]:+.4f} ({lo:+.4f} to {hi:+.4f}) |\n')
    open(f'{stem}.md', 'w').write(''.join(md))
    plot(a, gap, boot, band_res, band_of, cons, types, P, human, llms, sources, f"{stem}.png", phenom, suffix)
    print(f'{phenom}{suffix}: wrote {stem}.md/.tsv/.png')


def plot(a, gap, boot, band_res, band_of, cons, types, P, human, llms, sources, path, phenom, suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = {'constr': 'Constructions', 'lextype': 'Lexical types'}.get(phenom, phenom)
    title += {'': '', '_punct': ' (punctuation only)', '_xpunct': ' (punctuation excluded)'}[suffix]
    fig = plt.figure(figsize=(13, 4.2 + 0.24 * a.plot_top * 2))
    gs = fig.add_gridspec(2, len(llms), height_ratios=[1, 0.24 * a.plot_top * 2 / 4.2 + 0.4])
    ax = fig.add_subplot(gs[0, :])
    w = 0.8 / len(llms)
    for k, m in enumerate(llms):
        x = np.arange(len(BANDS)) + (k - (len(llms) - 1) / 2) * w
        v = band_res[m]
        ax.bar(x, [r[0] for r in v], width=w * 0.92, color=sources[m]['color'], label=sources[m]['label'])
        ax.errorbar(x, [r[0] for r in v], yerr=[[r[0] - r[1] for r in v], [r[2] - r[0] for r in v]],
                    fmt='none', ecolor='#1d1c1a', elinewidth=0.8, capsize=2)
    ax.axhline(0, color='#52514e', linewidth=0.8)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([f'{n}\n{int((band_of == b).sum())} types' for b, (_, _, n) in enumerate(BANDS)], fontsize=9)
    ax.set_ylabel('contribution to the Shannon gap\n(human minus model, nats)', fontsize=9)
    ax.set_title('Where the gap lies: contributions summed by frequency band (above 0 = the human texts more diverse)',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=len(llms))
    low = cons < a.plot_max_share
    for k, m in enumerate(llms):
        ax2 = fig.add_subplot(gs[1, k])
        sel = np.where(low)[0]
        top = sel[np.argsort(-np.abs(gap[m][sel]))][:a.plot_top][::-1]
        lo = [gap[m][i] - np.percentile(boot[m][:, i], 2.5) for i in top]
        hi = [np.percentile(boot[m][:, i], 97.5) - gap[m][i] for i in top]
        ax2.barh(range(len(top)), [gap[m][i] for i in top], xerr=[lo, hi],
                 color=[sources[human]['color'] if gap[m][i] > 0 else sources[m]['color'] for i in top],
                 error_kw={'elinewidth': 0.7, 'ecolor': '#1d1c1a'})
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels([types[i] for i in top], fontsize=7)
        ax2.axvline(0, color='#52514e', linewidth=0.8)
        ax2.set_title(f"{sources[m]['label']}", fontsize=9)
        ax2.tick_params(axis='x', labelsize=7)
        ax2.set_xlabel('contribution (nats)', fontsize=8)
    for axx in fig.axes:
        axx.spines['top'].set_visible(False); axx.spines['right'].set_visible(False)
    fig.suptitle(f'{title}: which types make the human texts more diverse?\n'
                 f'Lower panels: the {a.plot_top} largest contributions among types below {100 * a.plot_max_share:g}% '
                 f'mean share; blue = the human texts use it more, model color = the model uses it more', fontsize=10.5)
    if a.scope:
        fig.text(0.01, 0.003, a.scope, fontsize=7.5, color='#5f5d58')
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('counts', help='equal-size counts-by-excerpt.jsonl (equalize.py)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources', default=str(HERE / 'sources.json'))
    ap.add_argument('--phenomena', nargs='+', default=['constr', 'lextype'])
    ap.add_argument('--punct', nargs='+', default=['xpunct'], choices=['all', 'xpunct', 'punct'])
    ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--top', type=int, default=10, help='types listed per model and band in the .md')
    ap.add_argument('--plot-top', type=int, default=12)
    ap.add_argument('--plot-max-share', type=float, default=0.001,
                    help='lower panels show types below this mean share')
    ap.add_argument('--scope', default='')
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()
    sources = {k: v for k, v in json.load(open(a.sources)).items() if not k.startswith('_')}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for phenom in a.phenomena:
        for punct in a.punct:
            run(a, phenom, punct, sources, out)


if __name__ == '__main__':
    main()

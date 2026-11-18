# Analysis of syntactic and lexical types

Builds on the parsed profiles described in `PARSING.md` and reuses the method of
[llm-syntax](https://github.com/olzama/llm-syntax) (Zamaraeva et al., ACL 2025).

## Sources

`analysis/sources.json` lists the five subcorpora: the name used for each in frequency files
(human sources end in `-human`, models in `-llm`, as in llm-syntax), a display label, whether it
is human or an LLM, and a fixed plot color.

## Counting

```bash
python analysis/count_types.py --profiles <merged profiles root> --out <dir> \
    [--erg ~/delphin/erg/erg-github-latest]
```

The ERG source directory must be the grammar version used for parsing. Counted per sentence's
top-ranked derivation, with the same categories as llm-syntax:

| Phenomenon | What is counted |
|---|---|
| `constr` | rule nodes below the root whose name does not end in `lr` |
| `lexrule` | rule nodes whose name ends in `lr` |
| `lextype` | the lexical type of each word |
| `lexentries` | the lexical entry of each word |

Lexical types are taken from the derivation, where ACE records them, not looked up by entry
name in the grammar source; the lookup assigns the wrong type to entries that share a name with
a grammar type (the preposition *past* and the tense type `past`). Otherwise the counts are
identical to llm-syntax's `count_constructions.py`.

Outputs:

- `counts-by-excerpt.jsonl`: one record per subcorpus, excerpt and sentence-length band, with
  sentence, parsed-sentence and word counts and all type counts. Excerpts are matched across
  subcorpora by name, so this supports paired comparisons and length-controlled analyses.
- `frequencies.json`: `{phenomenon: {model: {type: count}}}`, the llm-syntax format.
- `lextype-depths.json`: lexical types generalized to supertypes at depths 1–5.

## Diversity and divergence

`analysis/diversity.py` is llm-syntax's `scripts/diversity.py` with model styling read from
`sources.json`, and all LLMs pooled as one aggregate instead of per-year groups. Metrics (Shannon,
Simpson), Jensen–Shannon divergence and the explain plots are unchanged.

```bash
MPLBACKEND=Agg python analysis/diversity.py <out>/frequencies.json --output-dir <dir> \
    --phenomena constr lextype --split-punct
MPLBACKEND=Agg python analysis/diversity.py <out>/frequencies.json --output-dir <dir> \
    --phenomena constr --group-explain "Novels-1838-1921-human" \
    "ClaudeSonnet5.5-llm,Gemini3.1Pro-llm,GPT-6Sol-llm,MistralLarge4-llm"
```

## Diversity at equal sample size

Diversity indices grow with the amount of text they are computed from, and the subcorpora yield
different numbers of type tokens (they are matched in words but differ in sentence count, sentence
length and coverage). `analysis/diversity_equal.py` compares them at the same number of type
tokens N (by default 80% of the smallest subcorpus total):

```bash
python analysis/diversity_equal.py <out>/counts-by-excerpt.jsonl --out <dir> \
    --phenomena constr lextype --split-punct
```

Each of `--reps` repetitions (default 1,000) resamples excerpts with replacement, using the same
excerpt names for every subcorpus (a paired bootstrap: excerpts are matched by novel), pools each
subcorpus's counts, and draws N type tokens without replacement, averaging the indices over
`--draws` such draws. All subcorpora in a repetition are drawn with the same random seed, so
differences between them are not inflated by drawing noise. Output per phenomenon: each
subcorpus's mean Shannon and Simpson index with a 95% interval, all LLMs pooled as one more
entry, and the human-minus-LLM differences with 95% intervals (Markdown, JSON and a plot).

`--unpaired` resamples each subcorpus's own excerpts independently, for data whose excerpts are
not matched (such as the pilot sample); `--no-bootstrap` only subsamples. With
`--no-bootstrap --size` equal to a subcorpus's total, its result equals `diversity.py`'s.

## Per-sentence measures

`analysis/sentence_measures.py` writes one row per sentence with structural measures from the
top-ranked derivation, for quantitative-linguistic analyses such as the Menzerath–Altmann law at
the sentence → clause → word levels:

```bash
python analysis/sentence_measures.py --profiles <merged profiles root> --out sentences.tsv \
    [--clauses analysis/clauses/finite.txt] [--clause-markers analysis/clauses/markers.txt]
```

What counts as a clause is set by plain-text lists in `analysis/clauses/` (one name or pattern per
line; shell wildcards allowed), so the definition can be changed without touching code. Each run
writes the settings it used to `<out>.settings.json`.

| Option (default file) | What it lists | Effect |
|---|---|---|
| `--clauses` (`finite.txt`) | constructions that build a clause | each such node is a clause; `finite+nonfinite.txt` adds non-finite and reduced clauses |
| `--clause-heads` (`clause-heads.txt`) | lexical rules or types whose word heads a clause on its own | the word's maximal projection is a clause: subject–auxiliary, quotative, conditional and locative inversion, which the ERG handles lexically rather than with a clause construction |
| `--clause-markers` (`markers.txt`) | complementizers (lexical types `cm_*`), marker and filler–head constructions | their words are counted with the clause they introduce: *whether*, relative pronouns and wh-fillers go with their clause |
| `--coordination` (`coordination-vp-finite.txt`) with `--coordination-mode one` or `split` | coordination constructions | `one`: coordinated VPs under one subject form one clause; `split`: each conjunct after the first is a clause, the first stays with the shared subject |

`coordination-vp-all.txt` adds non-finite VP coordination; `coordination-v.txt` lists coordination
of bare verbs sharing their complements, to add when wanted. Maximal projections follow head
daughters, read from ERG rule names (`hd-…` heads first; `sb-hd`, `aj-hd`, `flr-hd`, `sp-hd` heads
second).

Two definitions of a word are reported side by side, so results can be compared with Lenka's thesis:
lexical items of the derivation (`n_words`, `clause_lengths`, `word_syllables`, …) and the thesis's
tokens (`n_tokens`, `tok_clause_lengths`, `tok_syllables`, …: whitespace split of the sentence,
normalized as in the thesis). Each token belongs to the clause of the lexical item it overlaps most
in character offsets. Syllable and phoneme counts use the thesis's method (CMU Pronouncing
Dictionary via NLTK, with its fallback heuristic), ported unchanged in `analysis/syllables.py`;
this needs `nltk.download('cmudict')` once.

Each word (a non-punctuation lexical item) belongs to its nearest clause; a word outside every
clause, such as a sentence-initial modifier that combines with the clause above the subject–head
node, belongs to the clause nearest to it in the tree. Clause lengths therefore add up to the
sentence length. Columns include the number of words and clauses, clause lengths, word lengths in
characters, derivation depth, the number of constructions and the number of clauses added by
coordination; see the script's help.

## Distribution plots during parsing

`analysis/plot_distributions.py` plots distributions from whatever has been parsed so far, and
states on every plot how much data it rests on along its own grouping: per subcorpus, completed
excerpts and parsed sentences out of all sentences; per work, the same per novel (denominators
come from `data/items`). It also writes `processing.md`, a table of processing per subcorpus.

```bash
python analysis/count_types.py --profiles <result roots> --complete-only --out <dir>/counts
python analysis/sentence_measures.py --profiles <result roots> --complete-only --out <dir>/sentences.tsv
python analysis/plot_distributions.py --counts <dir>/counts/counts-by-excerpt.jsonl \
    --sentences <dir>/sentences.tsv --items data/items --out <dir>/plots \
    --scope "TEST: partial data, ..."
```

Plots: processing by work; construction rank-frequency; the most frequent constructions and
lexical types per subcorpus; clauses per sentence; mean clause length by sentence length in
clauses (Menzerath–Altmann, sentence → clause); mean clauses per sentence by work.

## Equal-size samples for partial results

`analysis/equalize.py` builds an equal-size sample from the counts and per-sentence table while
parsing is incomplete: only excerpts finished in every included subcorpus (matched by novel), the
same number of sentences per subcorpus (random sample down to the smallest), and type counts thinned
at random to the same total per subcorpus. `--exclude` leaves out a subcorpus that has barely been
parsed. The other analysis scripts run unchanged on its output.

## Differences between subcorpora

`analysis/model_differences.py` shows (1) the constructions or lexical types on which the subcorpora
differ most: each subcorpus's rate relative to the mean of all, as a log2 ratio with 95% intervals
from resampling excerpts; and (2) a map of excerpts placed by their type profile (PCA of
square-rooted shares), with the types that separate them most. `--punct xpunct` excludes
punctuation, as in llm-syntax; so does `plot_distributions.py --punct xpunct`, which also plots
word classes (lexical types grouped by ERG prefix) and syllables per word.

## Menzerath–Altmann law

`analysis/mal.py` applies Lenka's thesis method (grouped means per excerpt, Spearman's rho, log-log
fit of y = a·x^−b with R² and RMSE; the fitting code is ported from the thesis) at four levels:
sentence → word in syllables (thesis tokens and lexical items), sentence → clause in words, and
clause → word in syllables. It writes per-excerpt results, a summary, and plots.

## Comparison with llm-syntax's news data

`analysis/compare_llm_syntax.py` compares the human-vs-LLM differences in this corpus with those in
llm-syntax's 2025 news data (`analysis/frequencies-json/frequencies-2025.json`; the six 2025 models
of Gude et al., ACL 2026). For constructions and lexical types (punctuation excluded) it plots each
type's LLM-vs-human log2 ratio in news against the same ratio here, reports Spearman correlations
(overall and per model) and how many types deviate in the same or opposite direction, and computes
Shannon and Simpson diversity for every source of both corpora at one common sample size.

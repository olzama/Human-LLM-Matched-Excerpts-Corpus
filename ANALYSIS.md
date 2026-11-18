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

Each word (a non-punctuation lexical item) belongs to its nearest clause; a word outside every
clause, such as a sentence-initial modifier that combines with the clause above the subject–head
node, belongs to the clause nearest to it in the tree. Clause lengths therefore add up to the
sentence length. Columns include the number of words and clauses, clause lengths, word lengths in
characters, derivation depth, the number of constructions and the number of clauses added by
coordination; see the script's help.

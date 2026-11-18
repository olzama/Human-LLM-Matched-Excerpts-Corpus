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

What counts as a clause is set entirely by the `--clauses` file: one construction name per line.
`clauses/finite.txt` lists the constructions that build finite clauses (subject–head, main and
non-main; a finite clause whose subject is a gap on SLASH; imperatives);
`clauses/finite+nonfinite.txt` adds non-finite and reduced clauses. Each word (a non-punctuation
lexical item) belongs to its nearest clause; a word outside every clause, such as a sentence-initial
modifier that combines with the clause above the subject–head node, belongs to the clause nearest
to it in the tree. Clause lengths therefore add up to the sentence length. Words listed in
`--clause-markers` (complementizers, by lexical type `cm_*`) are counted with the clause they
introduce. Columns include the number of words and clauses, clause lengths, word lengths in
characters, derivation depth and the number of constructions; see the script's help.

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

Diversity indices depend on how much text they are computed from. The subcorpora are matched in
words but differ in sentence count and coverage, so compare diversity on equal-size samples
(the `--learning` curves show how far each index is from leveling off).

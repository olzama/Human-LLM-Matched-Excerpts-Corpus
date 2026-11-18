# Parsing the corpus with the ERG

Every sentence of the five subcorpora is parsed with the English Resource Grammar (ERG)
using the ACE parser, keeping the top-ranked analysis. Results are stored as
[incr tsdb()] profiles, one per excerpt.

## Versions

| Component | Version |
|---|---|
| Grammar | ERG, `delph-in/erg` commit `13b615a` (22 May 2026); reports itself as “ERG (2025)” |
| Parser | ACE 0.9.34; grammar image compiled with the same ACE (`ace -g ace/config.tdl -G erg-ace0934.dat`) |
| Driver | PyDelphin 1.6.0 |
| ACE options | `-1 --rooted-derivation --udx --max-words=1000`, plus per-tier `--timeout`, `--max-chart-megabytes`, `--max-unpack-megabytes` |

The grammar image must be built with the same ACE version that runs it.

## Pipeline

All commands run from the repository root.

1. **Sentences to items.** `scripts/prepare_items.py` reads `llm_dataset_LL.zip` and writes
   `data/items/<corpus>.jsonl`, one parser item per line. Segmentation is the corpus's own
   `sentence_list`, split additionally at blank lines (paragraph breaks). A closing quote that
   the corpus segmenter moved to the start of the next sentence is reattached. Markup
   (`*`, `_`, `#`, footnote marks, illustration captions) is removed, and quotes and dashes are
   normalized. Headings, titles, scene breaks and excerpt-boundary fragments are dropped and
   listed with a reason in `data/dropped.tsv`. The output is deterministic, so `data/items/`
   is not versioned; rerun the script to regenerate it.

2. **Profiles.** `scripts/make_profiles.py` builds `profiles/skeletons/<corpus>/<excerpt>/`, one
   profile per excerpt. Each item's `i-id` is unique across the corpus:
   `corpus × 10,000,000 + excerpt × 1,000 + position`; `i-origin` is the corpus, `i-register` the
   excerpt, and `i-comment` the position in `sentence_list` as `<entry>.<piece>`.
   With `--sample N` it builds a length-stratified pilot set instead.

3. **Split by length.** `scripts/split_by_length.py` divides each excerpt profile into short
   items (≤ 45 words) and long items in bands (`--long-bands 60,80` gives 46–60, 61–80 and
   over 80 words), and deals the short profiles out to machines in proportion to their
   capacity (`--weights`). Every item lands in exactly one part. `--parts long` rewrites only
   the long bands, leaving short profiles that are being parsed untouched.

4. **Parse.** `scripts/parse_tiered.py run` parses every profile under `--src` into `--dst`,
   with `--workers` parallel ACE processes. `--retry` re-parses only the items of a previous
   run that ran out of memory or time, or whose ACE process was killed; `--max-length` limits
   a retry to items of at most that many words. `--oom-first` starts ACE through
   `scripts/ace-oom-first.sh`, which makes it the kernel's first choice to kill if parallel
   jobs exhaust memory; the killed item is recorded as a crash and retried. A run can be
   restarted: finished profiles are skipped and interrupted ones are re-parsed.

   **Forwarding failures.** `scripts/forward_failures.py` runs on the coordinating machine
   while the first pass is under way. Every 30 minutes it copies finished result profiles back
   from the other machines, extracts from each finished profile the items that ran out of
   memory or time or whose ACE process was killed, and sends them as retry profiles to the
   big-memory machine's queue. Items with no analysis are not retried. A state file lists the
   profiles already handled, so a restart never forwards an item twice.

5. **Merge.** `scripts/parse_tiered.py merge --skeletons profiles/skeletons --tiers ...`
   rebuilds one complete profile per excerpt. Each item takes its result from the last of the
   given result directories (machines, length parts, retry tiers) that processed it; items
   that none processed are reported as missing. A provenance table records, for every item,
   where its result came from and its outcome.

6. **Combine.** `scripts/parse_tiered.py combine` concatenates the merged excerpt profiles
   into one profile per subcorpus. `report` prints coverage and ACE error messages per corpus.

## Memory and time

Running out of memory, not time, is what stops long sentences. Each length band starts at the
lowest memory setting that has a realistic chance and moves up only after failing; sentences
are never retried at a setting already known to fail for their length.

| Length (words) | First attempt | Retries |
|---|---|---|
| ≤ 45 | 16 GB, 120 s | 28 GB → 60 GB → 120 GB |
| 46–60 | 28 GB, 300 s | 60 GB → 120 GB |
| 61–80 | 60 GB, 600 s | 120 GB |
| over 80 | largest setting only | — |

The settings above the first row are under test and may change.

## Other machines

Copy `bin/ace`, the grammar image, the tsdb `Relations` file and `scripts/` to the server,
create a virtualenv with `pydelphin==1.6.0`, and copy `scripts/env.sh.example` to `env.sh`,
adjusting the paths. Source `env.sh` before running `parse_tiered.py`. Check a new machine by
parsing a few sentences and comparing derivations with a known-good run: they must be
identical.

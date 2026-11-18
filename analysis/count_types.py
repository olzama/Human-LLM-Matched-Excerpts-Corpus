#!/usr/bin/env python3
"""Count syntactic and lexical types in parsed profiles.

Counting follows llm-syntax (scripts/count_constructions.py, count_lextypes.py)
so that the categories mean the same thing:

  constr      every rule node below the root of the top-ranked derivation
              whose name does not end in 'lr'
  lexrule     rule nodes whose name ends in 'lr'
  lextype     the lexical type of each word
  lexentries  the lexical entry of each word

One deliberate difference: the lexical type is the one ACE records in the
derivation (--udx output), not a lookup of the entry name in the grammar
source. The lookup goes wrong when an entry shares its name with a grammar
type (e.g. the preposition 'past' vs the tense type 'past').
Lexical types at depth d > 1 are the depth-(d-1) supertypes of that type;
when a type has several parents, they are joined with '+', and parallel
branches at one level with '__' (as in llm-syntax, but sorted, so the result
does not depend on set order).

Every profile under --profiles is read; each item's subcorpus and excerpt come
from its i-origin and i-register fields, so per-excerpt and combined profiles
both work. Outputs (in --out):

  counts-by-excerpt.jsonl   one record per (corpus, excerpt, length band):
                            sentences, parsed, parsed_words, counts by
                            phenomenon, and lexical types at depths 1-5
  frequencies.json          {phenomenon: {model: {type: count}}}, the
                            llm-syntax format read by diversity.py
  lextype-depths.json       {depth: {model: {type: count}}}, the format of
                            llm-syntax's count_lextypes.py
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from delphin import derivation, itsdb, tdl

DEFAULT_ERG = Path.home() / 'delphin/erg/erg-github-latest'
BANDS = [(1, 20), (21, 30), (31, 45), (46, 60), (61, 80), (81, 10 ** 9)]
DEPTHS = range(1, 6)


def band_of(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}-'


def load_types(erg_dir):
    """Type name -> list of parent names, from the grammar's .tdl files."""
    parents = {}
    for f in sorted(Path(erg_dir).glob('*.tdl')):
        for event, obj, _ in tdl.iterparse(f):
            if event in ('TypeDefinition', 'LexicalRuleDefinition'):
                parents.setdefault(obj.identifier, [str(s) for s in obj.supertypes])
    return parents


class Ancestry:
    """Supertype labels per depth for a lexical type, memoized."""

    def __init__(self, parents):
        self.parents = parents
        self.cache = {}

    def labels(self, letype):
        if letype in self.cache:
            return self.cache[letype]
        out = {1: letype}
        level = [letype]
        for d in DEPTHS[1:]:
            groups = sorted({'+'.join(self.parents[t]) for t in level if self.parents.get(t)})
            if not groups:
                break
            out[d] = '__'.join(groups)
            level = sorted({p for t in level for p in self.parents.get(t, [])})
        self.cache[letype] = out
        return out


def count_derivation(deriv_str, ancestry, rec):
    d = derivation.from_string(deriv_str)
    preterminals = d.preterminals()
    for pt in preterminals:                      # one per word
        rec['lexentries'][pt.entity] += 1
        letype = pt.type or pt.entity            # type recorded by ACE (--udx)
        rec['lextype'][letype] += 1
        for depth, label in ancestry.labels(letype).items():
            rec['depth'][depth][label] += 1
    word_nodes = {id(pt) for pt in preterminals}
    stack = [d]                                  # d is the root node; it is not counted
    while stack:
        node = stack.pop()
        for child in node.daughters:
            if not isinstance(child, derivation.UDFNode):
                continue
            stack.append(child)
            if id(child) in word_nodes:
                continue
            kind = 'lexrule' if child.entity.endswith('lr') else 'constr'
            rec[kind][child.entity] += 1


def new_record():
    return {'sentences': 0, 'parsed': 0, 'parsed_words': 0,
            'constr': Counter(), 'lexrule': Counter(), 'lextype': Counter(), 'lexentries': Counter(),
            'depth': {d: Counter() for d in DEPTHS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True, help='root of parsed profiles (searched recursively)')
    ap.add_argument('--erg', default=str(DEFAULT_ERG), help='ERG source directory used for parsing')
    ap.add_argument('--sources', default=str(Path(__file__).with_name('sources.json')))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    sources = {k: v for k, v in json.load(open(a.sources)).items() if not k.startswith('_')}
    ancestry = Ancestry(load_types(a.erg))
    records = defaultdict(new_record)
    for rel in sorted(Path(a.profiles).rglob('relations')):
        ts = itsdb.TestSuite(rel.parent)
        for item in ts.processed_items():
            corpus, excerpt, n = item['i-origin'], item['i-register'], int(item['i-length'])
            rec = records[(corpus, excerpt, band_of(n))]
            rec['sentences'] += 1
            results = item.results()
            if results:
                rec['parsed'] += 1
                rec['parsed_words'] += n
                count_derivation(results[0]['derivation'], ancestry, rec)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    freq = {p: defaultdict(Counter) for p in ('constr', 'lexrule', 'lextype', 'lexentries')}
    depths = {d: defaultdict(Counter) for d in DEPTHS}
    with open(out / 'counts-by-excerpt.jsonl', 'w') as fh:
        for (corpus, excerpt, band), rec in sorted(records.items()):
            fh.write(json.dumps({
                'corpus': corpus, 'excerpt': excerpt, 'work': excerpt.rsplit('_', 1)[0], 'length_band': band,
                'sentences': rec['sentences'], 'parsed': rec['parsed'], 'parsed_words': rec['parsed_words'],
                **{p: dict(rec[p]) for p in freq},
                'lextype_depth': {str(d): dict(c) for d, c in rec['depth'].items() if c},
            }, ensure_ascii=False) + '\n')
            key = sources.get(corpus, {}).get('key', corpus)
            for p in freq:
                freq[p][key].update(rec[p])
            for d in DEPTHS:
                depths[d][key].update(rec['depth'][d])
    sort = lambda c: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    json.dump({p: {m: sort(c) for m, c in byk.items()} for p, byk in freq.items()},
              open(out / 'frequencies.json', 'w'), ensure_ascii=False, indent=1)
    json.dump({str(d): {m: sort(c) for m, c in byk.items()} for d, byk in depths.items()},
              open(out / 'lextype-depths.json', 'w'), ensure_ascii=False, indent=1)
    by_corpus = defaultdict(Counter)
    for (corpus, _, _), rec in records.items():
        by_corpus[corpus].update({k: rec[k] for k in ('sentences', 'parsed', 'parsed_words')})
    for corpus, c in sorted(by_corpus.items()):
        print(f"{corpus}: {c['sentences']} sentences, {c['parsed']} parsed, {c['parsed_words']} words in parsed sentences")
    print(f'wrote {out}/counts-by-excerpt.jsonl, frequencies.json, lextype-depths.json')


if __name__ == '__main__':
    main()

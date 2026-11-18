#!/usr/bin/env python3
"""Per-sentence structural measures from the top-ranked ERG derivation.

Writes one row per sentence (TSV) for quantitative-linguistic analyses such
as the Menzerath-Altmann law at the sentence -> clause -> word levels.

Definitions
  word         a lexical item of the derivation that is not punctuation (the
               ERG treats commas, dashes etc. as lexical items; they are left
               out with the same rule as diversity.py). A multi-word lexical
               entry counts as one word.
  clause       a node built by one of the constructions in --clauses (a text
               file, one construction name per line, '#' starts a comment).
               What counts as a clause is decided entirely by that list.
  clause length
               number of words belonging to the clause. Each word belongs to
               its nearest clause ancestor. A word with no clause ancestor
               (e.g. a sentence-initial modifier that combines with the
               clause by a head-adjunct construction above the subject-head
               node) belongs to the clause nearest to it in the
               tree: going up from the word, the first ancestor that contains a
               clause node, and within it the clause node closest to that
               ancestor (leftmost on ties). Clause lengths therefore add up to
               the sentence length whenever the sentence has a clause.
               Sentences without any clause node get n_clauses = 0.
               Clause markers listed in --clause-markers (lexical types such
               as complementizers 'cm_*', or marker constructions) are
               counted with the clause they introduce.
  word length  characters of the word's surface form (letters and digits only).
  deriv_depth  longest path from the root to a lexical entry, counting every
               node (constructions, lexical rules and the lexical entry).
  phrase_depth the same path length counting syntactic constructions only.

Columns: corpus, excerpt, work, i_id, sent_pos (position in the corpus's
sentence_list as <entry>.<piece>), input_words (whitespace tokens of the
input), parsed, n_words, n_clauses, clause_lengths (comma-separated, in
order of the clause's first word), mean_clause_len, word_lengths
(comma-separated), mean_word_len, deriv_depth, phrase_depth,
n_constructions. Unparsed sentences have parsed=0 and empty measures.
"""
import argparse
import csv
import re
from fnmatch import fnmatch
from pathlib import Path

from delphin import derivation, itsdb

HERE = Path(__file__).resolve().parent


def is_punct_type(tname):
    """Same rule as diversity.py."""
    t = (tname or '').lower()
    return t.startswith('pt') or ('pct' in t) or t.endswith('plr')


def read_list(path):
    names = set()
    for line in open(path, encoding='utf-8'):
        line = line.split('#', 1)[0].strip()
        if line:
            names.add(line)
    return names


def analyse(deriv_str, clause_rules, marker_patterns=()):
    root = derivation.from_string(deriv_str)
    # parent links and word order
    parent, words, order = {}, [], []
    word_nodes = {id(p) for p in root.preterminals()}

    def walk(node, depth_all, depth_phr, path):
        nonlocal max_all, max_phr
        if id(node) in word_nodes:
            max_all = max(max_all, depth_all + 1)
            max_phr = max(max_phr, depth_phr)
            if not is_punct_type(node.type):
                form = ''.join(t.form for t in node.terminals())
                words.append((node, form))
            return
        for child in node.daughters:
            if isinstance(child, derivation.UDFNode):
                parent[id(child)] = node
                is_constr = id(child) not in word_nodes and not child.entity.endswith('lr')
                walk(child, depth_all + 1, depth_phr + (1 if is_constr else 0), path)

    max_all = max_phr = 0
    walk(root, 0, 0, [])
    n_constr = 0
    clause_nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.daughters:
            if isinstance(child, derivation.UDFNode) and id(child) not in word_nodes:
                stack.append(child)
                if not child.entity.endswith('lr'):
                    n_constr += 1
                if child.entity in clause_rules:
                    clause_nodes.append(child)
    clause_ids = {id(c) for c in clause_nodes}

    def depth_below(anc, target):
        """Distance from anc down to target, or None if target is not below anc."""
        d, n = 0, target
        while n is not None and n is not anc:
            n = parent.get(id(n)); d += 1
        return d if n is anc else None

    assign = {}
    for i, (w, _) in enumerate(words):
        n = parent.get(id(w))
        while n is not None and id(n) not in clause_ids:
            n = parent.get(id(n))
        if n is None and clause_nodes:            # no clause ancestor: nearest clause in the tree
            a = parent.get(id(w))
            while a is not None:
                cands = [(depth_below(a, c), k, c) for k, c in enumerate(clause_nodes)]
                cands = [x for x in cands if x[0] is not None]
                if cands:
                    n = min(cands, key=lambda x: (x[0], x[1]))[2]
                    break
                a = parent.get(id(a))
        if n is not None:
            assign[i] = id(n)
    # Clause markers: complementizers (by lexical type) and marker constructions
    # (by name) are counted with the clause they introduce.
    def sister_clause(node, exclude_word):
        """The clause that node combines with the word's side: a daughter of node
        not containing the word that is a clause node, possibly through a chain
        of single-daughter nodes. None otherwise."""
        for dtr in node.daughters:
            if not isinstance(dtr, derivation.UDFNode) or depth_below(dtr, exclude_word) is not None \
                    or dtr is exclude_word:
                continue
            n = dtr
            while id(n) not in clause_ids:
                kids = [k for k in n.daughters if isinstance(k, derivation.UDFNode)]
                if len(kids) != 1 or id(n) in word_nodes:
                    n = None
                    break
                n = kids[0]
            if n is not None:
                return n
        return None

    for i, (w, _) in enumerate(words):
        target = None
        if any(fnmatch(w.type or '', pat) for pat in marker_patterns):
            p = parent.get(id(w))
            if p is not None:
                target = sister_clause(p, w)
        if target is None:
            m = parent.get(id(w))
            while m is not None and not any(fnmatch(m.entity, pat) for pat in marker_patterns):
                if id(m) in clause_ids:
                    m = None
                    break
                m = parent.get(id(m))
            if m is not None:
                target = sister_clause(m, w)
        if target is not None:
            assign[i] = id(target)

    first = {}
    lengths = {}
    for i, cid in assign.items():
        first.setdefault(cid, i)
        lengths[cid] = lengths.get(cid, 0) + 1
    ordered = sorted(lengths, key=lambda c: first[c])
    clause_lengths = [lengths[c] for c in ordered]
    segments = [[words[i][1] for i in sorted(assign) if assign[i] == c] for c in ordered]
    word_lengths = [len(re.sub(r'[\W_]', '', f)) for _, f in words]
    return {
        'n_words': len(words), 'n_clauses': len(clause_lengths), 'clause_lengths': clause_lengths,
        'word_lengths': word_lengths, 'deriv_depth': max_all, 'phrase_depth': max_phr,
        'n_constructions': n_constr, 'segments': segments,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True, help='root of parsed profiles (searched recursively)')
    ap.add_argument('--clauses', default=str(HERE / 'clauses' / 'finite.txt'),
                    help='file listing the constructions that build a clause')
    ap.add_argument('--clause-markers', default=str(HERE / 'clauses' / 'markers.txt'),
                    help='file listing marker constructions whose marker words join the marked clause '
                         "(use an empty file to turn this off)")
    ap.add_argument('--out', required=True, help='output TSV')
    a = ap.parse_args()

    clause_rules = read_list(a.clauses)
    marker_rules = read_list(a.clause_markers) if a.clause_markers else set()
    cols = ['corpus', 'excerpt', 'work', 'i_id', 'sent_pos', 'input_words', 'parsed', 'n_words', 'n_clauses',
            'clause_lengths', 'mean_clause_len', 'word_lengths', 'mean_word_len', 'deriv_depth', 'phrase_depth',
            'n_constructions']
    n = parsed = 0
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(cols)
        for rel in sorted(Path(a.profiles).rglob('relations')):
            for item in itsdb.TestSuite(rel.parent).processed_items():
                ex = item['i-register']
                row = {'corpus': item['i-origin'], 'excerpt': ex, 'work': ex.rsplit('_', 1)[0],
                       'i_id': item['i-id'], 'sent_pos': item['i-comment'], 'input_words': item['i-length'],
                       'parsed': 0}
                res = item.results()
                if res:
                    m = analyse(res[0]['derivation'], clause_rules, marker_rules)
                    cl, wl = m['clause_lengths'], m['word_lengths']
                    row.update(parsed=1, n_words=m['n_words'], n_clauses=m['n_clauses'],
                               clause_lengths=','.join(map(str, cl)),
                               mean_clause_len=f'{sum(cl) / len(cl):.3f}' if cl else '',
                               word_lengths=','.join(map(str, wl)),
                               mean_word_len=f'{sum(wl) / len(wl):.3f}' if wl else '',
                               deriv_depth=m['deriv_depth'], phrase_depth=m['phrase_depth'],
                               n_constructions=m['n_constructions'])
                    parsed += 1
                w.writerow([row.get(c, '') for c in cols])
                n += 1
    print(f'{n} sentences ({parsed} parsed) -> {a.out}; clause constructions from {a.clauses}: '
          f'{", ".join(sorted(clause_rules))}')


if __name__ == '__main__':
    main()

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
  clause head --clause-heads lists lexical rules/types (e.g. inverted
               auxiliaries, quotative inversion) whose word heads a clause
               without a clause-building construction; the clause is the word's
               maximal projection, with headedness read from ERG rule names.
  coordination --coordination lists coordination constructions (default:
               finite VP coordination); --coordination-mode decides how they
               count. 'one' (default): a coordinated VP is part of one clause.
               'split': each conjunct after the first is a clause of its own;
               the first conjunct stays in the clause above together with
               shared material such as the subject (n conjuncts add n-1
               clauses; with no clause above, every conjunct is a clause).
               Chained coordinations are expanded into all their conjuncts.
               The coord_extra_clauses column counts the clauses added this
               way. Settings are written to <out>.settings.json.
               Clause markers listed in --clause-markers (lexical types such
               as complementizers 'cm_*', or marker constructions) are
               counted with the clause they introduce.
  word length  characters of the word's surface form (letters and digits only).
  deriv_depth  longest path from the root to a lexical entry, counting every
               node (constructions, lexical rules and the lexical entry).
  phrase_depth the same path length counting syntactic constructions only.

Two word definitions are reported side by side. Lexical items (above) give
n_words, clause_lengths, word_syllables etc. The thesis tokens (whitespace
split of the sentence, normalized as in Lenka's thesis; see syllables.py) give
n_tokens, tok_clause_lengths, tok_syllables etc.; each token belongs to the
clause of the lexical item it overlaps most in character offsets. Syllables
and phonemes use the thesis's CMU-dictionary method with its fallback.

Columns: corpus, excerpt, work, i_id, sent_pos (position in the corpus's
sentence_list as <entry>.<piece>), input_words (whitespace tokens of the
input), parsed, n_words, n_clauses, clause_lengths (comma-separated, in
order of the clause's first word), mean_clause_len, word_lengths
(comma-separated), mean_word_len, deriv_depth, phrase_depth,
n_constructions. Unparsed sentences have parsed=0 and empty measures.
"""
import argparse
import csv
import sys
import json
import re
from fnmatch import fnmatch
from pathlib import Path

from delphin import derivation, itsdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syllables import count_phonemes, count_syllables, thesis_tokens  # noqa: E402

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


def head_index(rule, n_daughters):
    """Index of the head daughter, read from ERG rule naming: 'hd...' (hd-cmp,
    hd_xcmp, hdn-aj, unary hd_...) has the head first; 'x-hd...' (sb-hd, aj-hd,
    flr-hd, sp-hd, aj-hdn, np-hdn) has it second; lexical rules and root nodes
    are unary-headed. Other rules (coordination, fragments, mrk-nh) are treated
    as non-headed (None)."""
    if n_daughters == 1 and (rule.endswith('lr') or rule.startswith(('hd', 'root'))):
        return 0
    if rule.startswith('hd'):
        return 0
    first = rule.split('_', 1)[0]
    if re.fullmatch(r'[a-z]+-hdn?', first) and n_daughters >= 2:
        return 1
    return None


def analyse(deriv_str, clause_rules, marker_patterns=(), coord_rules=frozenset(), coord_mode='one',
            head_patterns=()):
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
                form = ' '.join(t.form for t in node.terminals())
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
    clause_nodes, coord_nodes = [], []
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
                if child.entity in coord_rules:
                    coord_nodes.append(child)
    clause_ids = {id(c) for c in clause_nodes}

    # Clause heads: a word with a listed lexical rule or lexical type (e.g. an
    # inverted auxiliary) heads a clause: its maximal projection.
    if head_patterns:
        for pt in root.preterminals():
            chain, n = [pt.type or ''], pt
            while parent.get(id(n)) is not None and parent[id(n)].entity.endswith('lr'):
                n = parent[id(n)]; chain.append(n.entity)
            if not any(fnmatch(x, pat) for x in chain for pat in head_patterns):
                continue
            top = n
            while True:
                par = parent.get(id(top))
                if par is None or par is root or id(top) in clause_ids:
                    break
                kids = [k for k in par.daughters if isinstance(k, derivation.UDFNode)]
                hi = head_index(par.entity, len(kids))
                if hi is None or kids[hi] is not top:
                    break
                top = par
            if id(top) not in clause_ids and top is not root:
                clause_nodes.append(top)
                clause_ids.add(id(top))

    # Coordination, split mode: each conjunct after the first of a listed
    # coordination becomes a clause of its own; the first conjunct stays in the
    # clause above, with any shared material (e.g. the subject). Chained
    # coordinations (-m variants) are expanded into all their conjuncts. With no
    # clause above the coordination, every conjunct becomes a clause.
    coord_extra = 0
    if coord_mode == 'split':
        def conjuncts(v):
            out = []
            for d in v.daughters:
                if isinstance(d, derivation.UDFNode):
                    out.extend(conjuncts(d) if d.entity in coord_rules else [d])
            return out
        for v in coord_nodes:
            if parent.get(id(v)) is not None and parent[id(v)].entity in coord_rules:
                continue                          # part of a larger coordination, expanded there
            cj = conjuncts(v)
            if len(cj) < 2:
                continue
            above = parent.get(id(v))
            while above is not None and id(above) not in clause_ids:
                above = parent.get(id(above))
            new = cj[1:] if above is not None else cj
            for d in new:
                if id(d) not in clause_ids:
                    clause_nodes.append(d)
                    clause_ids.add(id(d))
            coord_extra += len(cj) - 1

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
    clause_index = {c: k for k, c in enumerate(ordered)}
    word_info = []
    for i, (w, f) in enumerate(words):
        spans = [(int(a), int(b)) for t in w.terminals() for tok in t.tokens
                 for a, b in re.findall(r'\+FROM \\?"(\d+)\\?" \+TO \\?"(\d+)\\?"', tok.tfs or '')]
        start = min(a for a, _ in spans) if spans else None
        end = max(b for _, b in spans) if spans else None
        word_info.append({'form': f, 'start': start, 'end': end,
                          'clause': clause_index.get(assign.get(i))})
    return {
        'n_words': len(words), 'n_clauses': len(clause_lengths), 'clause_lengths': clause_lengths,
        'word_lengths': word_lengths, 'deriv_depth': max_all, 'phrase_depth': max_phr,
        'n_constructions': n_constr, 'segments': segments, 'coord_extra': coord_extra,
        'word_info': word_info,
    }


def token_measures(sentence, word_info, n_clauses):
    """Thesis tokens of the sentence, each assigned to the clause of the lexical
    item it overlaps most in character offsets (nearest item if none overlaps)."""
    toks = thesis_tokens(sentence)
    located = [w for w in word_info if w['start'] is not None and w['clause'] is not None]
    lengths = [0] * n_clauses
    for _, a, b in toks:
        if not located or not n_clauses:
            break
        best = max(located, key=lambda w: (min(b, w['end']) - max(a, w['start']),
                                           -abs((w['start'] + w['end']) / 2 - (a + b) / 2)))
        lengths[best['clause']] += 1
    return toks, [x for x in lengths if x > 0]



def profile_dirs(roots, complete_only=False):
    """relations files of all profiles under the roots; with complete_only, only
    profiles whose parse table has a row for every item."""
    from delphin import tsdb
    def rows(path, table):
        try:
            with tsdb.open(path, table) as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0
    out = []
    for root in roots:
        for rel in sorted(Path(root).rglob('relations')):
            if complete_only:
                n = rows(rel.parent, 'item')
                if n == 0 or rows(rel.parent, 'parse') != n:
                    continue
            out.append(rel)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True, nargs='+',
                    help='one or more roots of parsed profiles (searched recursively)')
    ap.add_argument('--complete-only', action='store_true',
                    help='skip profiles that are still being parsed (fewer parse rows than items)')
    ap.add_argument('--clauses', default=str(HERE / 'clauses' / 'finite.txt'),
                    help='file listing the constructions that build a clause')
    ap.add_argument('--clause-markers', default=str(HERE / 'clauses' / 'markers.txt'),
                    help='file listing marker constructions whose marker words join the marked clause '
                         "(use an empty file to turn this off)")
    ap.add_argument('--clause-heads', default=str(HERE / 'clauses' / 'clause-heads.txt'),
                    help='file listing lexical rules/types whose word heads a clause of its own '
                         '(e.g. inverted auxiliaries); use an empty file to turn this off')
    ap.add_argument('--coordination', default=str(HERE / 'clauses' / 'coordination-vp-finite.txt'),
                    help='file listing coordination constructions whose conjuncts may count as clauses')
    ap.add_argument('--coordination-mode', choices=['one', 'split'], default='one',
                    help="one: a coordination of listed constructions stays within one clause; "
                         "split: each conjunct after the first is a clause of its own")
    ap.add_argument('--out', required=True, help='output TSV')
    a = ap.parse_args()
    coord_rules = read_list(a.coordination) if a.coordination else set()
    head_patterns = sorted(read_list(a.clause_heads)) if a.clause_heads else []

    clause_rules = read_list(a.clauses)
    marker_rules = read_list(a.clause_markers) if a.clause_markers else set()
    cols = ['corpus', 'excerpt', 'work', 'i_id', 'sent_pos', 'input_words', 'parsed', 'n_words', 'n_clauses',
            'clause_lengths', 'mean_clause_len', 'word_lengths', 'mean_word_len', 'deriv_depth', 'phrase_depth',
            'n_constructions', 'coord_extra_clauses',
            'word_syllables', 'mean_word_syll', 'word_phonemes',
            'n_tokens', 'tok_clause_lengths', 'mean_tok_clause_len', 'tok_syllables', 'mean_tok_syll',
            'tok_phonemes']
    n = parsed = 0
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(cols)
        for rel in profile_dirs(a.profiles, a.complete_only):
            for item in itsdb.TestSuite(rel.parent).processed_items():
                ex = item['i-register']
                row = {'corpus': item['i-origin'], 'excerpt': ex, 'work': ex.rsplit('_', 1)[0],
                       'i_id': item['i-id'], 'sent_pos': item['i-comment'], 'input_words': item['i-length'],
                       'parsed': 0}
                res = item.results()
                if res:
                    m = analyse(res[0]['derivation'], clause_rules, marker_rules, coord_rules, a.coordination_mode,
                                head_patterns)
                    cl, wl = m['clause_lengths'], m['word_lengths']
                    row.update(parsed=1, n_words=m['n_words'], n_clauses=m['n_clauses'],
                               clause_lengths=','.join(map(str, cl)),
                               mean_clause_len=f'{sum(cl) / len(cl):.3f}' if cl else '',
                               word_lengths=','.join(map(str, wl)),
                               mean_word_len=f'{sum(wl) / len(wl):.3f}' if wl else '',
                               deriv_depth=m['deriv_depth'], phrase_depth=m['phrase_depth'],
                               n_constructions=m['n_constructions'], coord_extra_clauses=m['coord_extra'])
                    # a multi-word lexical item is counted word by word and summed
                    syl = [sum(count_syllables(x) for x in w['form'].split()) for w in m['word_info']]
                    syl = [x for x in syl if x > 0]
                    row.update(word_syllables=','.join(map(str, syl)),
                               mean_word_syll=f'{sum(syl) / len(syl):.3f}' if syl else '',
                               word_phonemes=','.join(str(sum(count_phonemes(x) for x in w['form'].split()))
                                                      for w in m['word_info']))
                    toks, tcl = token_measures(item['i-input'], m['word_info'], m['n_clauses'])
                    tsyl = [count_syllables(t) for t, _, _ in toks]
                    row.update(n_tokens=len(toks), tok_clause_lengths=','.join(map(str, tcl)),
                               mean_tok_clause_len=f'{sum(tcl) / len(tcl):.3f}' if tcl else '',
                               tok_syllables=','.join(map(str, tsyl)),
                               mean_tok_syll=f'{sum(tsyl) / len(tsyl):.3f}' if tsyl else '',
                               tok_phonemes=','.join(str(count_phonemes(t)) for t, _, _ in toks))
                    parsed += 1
                w.writerow([row.get(c, '') for c in cols])
                n += 1
    settings = {'clauses_file': a.clauses, 'clause_constructions': sorted(clause_rules),
                'clause_markers_file': a.clause_markers, 'clause_markers': sorted(marker_rules),
                'coordination_file': a.coordination, 'coordination_constructions': sorted(coord_rules),
                'coordination_mode': a.coordination_mode, 'clause_heads_file': a.clause_heads,
                'clause_heads': head_patterns, 'profiles': a.profiles}
    json.dump(settings, open(a.out + '.settings.json', 'w'), indent=1)
    print(f'{n} sentences ({parsed} parsed) -> {a.out} (settings in {a.out}.settings.json); '
          f'coordination mode {a.coordination_mode}')


if __name__ == '__main__':
    main()

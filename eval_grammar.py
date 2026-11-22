"""LinguaBot v7.2 evaluator (169-case JSONL, grammar-only error policy).

Colab:
    import eval_v7 as E
    E.run('baseline_v7', test_file='linguabot_test_set_v7_2_apostrophes_169.jsonl')
    E.rescore('baseline_v7', test_file='linguabot_test_set_v7_2_apostrophes_169.jsonl')

Requires pandas, openpyxl, and lingua_core.py for live inference only.
The evaluator itself can be imported and tested without lingua_core.
"""
from __future__ import annotations
import json
import re
import statistics
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

DEFAULT_TEST = 'linguabot_test_set_v7_2_apostrophes_169.jsonl'
CYR = re.compile(r'[А-Яа-яЁё]')
EMOJI_MD = re.compile(r'[\U0001F300-\U0001FAFF☀-➿]|\*\*|^#|`', re.M)
LABEL = re.compile(r'follow[- ]?up question|^question\s*:', re.I | re.M)
APOSTROPHES = str.maketrans({'`': "'", '´': "'", '‘': "'", '’': "'", 'ʼ': "'"})
FINAL_PUNCT = re.compile(r'[.!?]+\s*$')
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", re.I)
METRICS = [
    'json_ok', 'schema_ok', 'corrected_ok', 'corrections_exact', 'corr_fix_ok',
    'rule_ok', 'no_false_alarm', 'rule_valid', 'all_spans_found',
    'reply_en', 'one_question', 'len_ok', 'no_emoji_md', 'no_label',
    'cosmetic_only_corrections', 'no_user_facing_false_alarm',
]


def tokens(s):
    return TOKEN.findall(str(s).translate(APOSTROPHES).lower())


def normalized(s):
    """Ignore case, whitespace, apostrophe glyphs, and punctuation for corrected-text comparison.

    WARNING: This also ignores internal punctuation. Do not use for identifying
    missing apostrophes: dont and don't remain different tokens.
    """
    return tuple(tokens(s))


# Only unambiguous, familiar apostrophe omissions are normalized.
# In particular, its/it's, were/we're, well/we'll, hell/he'll and shell/she'll
# are deliberately NOT equated: they may express different meanings.
COSMETIC_CONTRACTIONS = {
    'im': "i'm", 'ive': "i've", 'id': "i'd", 'ill': "i'll",
    'dont': "don't", 'doesnt': "doesn't", 'didnt': "didn't",
    'cant': "can't", 'cannot': 'cannot', 'couldnt': "couldn't",
    'wouldnt': "wouldn't", 'shouldnt': "shouldn't", 'wont': "won't",
    'isnt': "isn't", 'arent': "aren't", 'wasnt': "wasn't",
    'werent': "weren't", 'hasnt': "hasn't", 'havent': "haven't",
    'hadnt': "hadn't", 'mustnt': "mustn't", 'neednt': "needn't",
    'youre': "you're", 'theyre': "they're", 'weve': "we've",
    'youve': "you've", 'theyve': "they've", 'thats': "that's",
    'whats': "what's", 'whos': "who's", 'theres': "there's",
    'lets': "let's", 'heres': "here's", 'hows': "how's",
}

def cosmetic_tokens(s):
    return tuple(COSMETIC_CONTRACTIONS.get(t, t) for t in tokens(s))


def cosmetic_equal(a, b):
    """Ignore case/punctuation and *safe* missing apostrophes, not grammar."""
    return cosmetic_tokens(a) == cosmetic_tokens(b)


def text_equal(a, b):
    return normalized(a) == normalized(b)


def words(s):
    return len(tokens(s))


def sentences(s):
    return [p for p in re.split(r'(?<=[.!?])\s+', str(s).strip()) if p.strip()]


def parse_answer(answer):
    """Strict JSON object; no silent salvage of invalid model responses."""
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return None
    try:
        obj = json.loads(answer.strip())
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def schema_valid(obj):
    if not isinstance(obj, dict) or set(obj) != {'corrected', 'corrections', 'reply'}:
        return False
    if not isinstance(obj['corrected'], str) or not isinstance(obj['reply'], str):
        return False
    if not isinstance(obj['corrections'], list):
        return False
    return all(isinstance(x, dict) and {'original', 'correct', 'rule_id'} <= set(x)
               and all(isinstance(x[k], str) and x[k].strip() for k in ('original', 'correct', 'rule_id'))
               for x in obj['corrections'])


def rule_ids(g):
    ids = g.get('rule_id', [])
    if isinstance(ids, str):
        ids = [ids]
    return set(ids) | set(g.get('also_ok_rule_ids', []))


def gold_items(c):
    """New v7 format, with support for legacy v6 gold_corrections."""
    if 'corrections' in c:
        return c['corrections']
    out = []
    for g in c.get('gold_corrections', []):
        ids = g.get('rule_id', [])
        if isinstance(ids, str):
            ids = [ids]
        out.append({**g, 'rule_id': ids[0] if ids else '', 'also_ok_rule_ids': ids[1:]})
    return out


def corrected_targets(c):
    vals = [c.get('corrected', c['input'])] + c.get('also_ok_corrected', [])
    return {cosmetic_tokens(x) for x in vals}


def _edit_signatures(source, original, correct):
    """Return possible position-aware edits made by a correction span.

    Context words in the span may differ; only actual token changes count.
    Multiple occurrences are kept as candidates, but matching remains 1-to-1.
    """
    src, old, new = cosmetic_tokens(source), cosmetic_tokens(original), cosmetic_tokens(correct)
    if not old or old == new or len(old) > len(src):
        return set()
    edits = []
    for tag, i, j, u, v in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != 'equal':
            edits.append((tag, i, j, tuple(new[u:v])))
    if not edits:
        return set()
    results = set()
    for start in range(len(src) - len(old) + 1):
        if src[start:start + len(old)] == old:
            results.add(tuple((tag, start + i, start + j, replacement)
                              for tag, i, j, replacement in edits))
    return results


def pair_match(pred, gold, source=None):
    """Match identical edits, even when correction spans include extra context."""
    if source is None:
        # Legacy fallback for callers without the source sentence.
        if cosmetic_tokens(pred['original']) != cosmetic_tokens(gold['original']):
            return False
        return any(cosmetic_tokens(pred['correct']) == cosmetic_tokens(x)
                   for x in [gold['correct']] + gold.get('also_ok', []))
    predicted = _edit_signatures(source, pred['original'], pred['correct'])
    if not predicted:
        return False
    return any(predicted & _edit_signatures(source, gold['original'], option)
               for option in [gold['correct']] + gold.get('also_ok', []))


def match_pairs(predictions, gold, source=None):
    """Maximum one-to-one matching; duplicate predictions cannot inflate recall."""
    edges = [[j for j, g in enumerate(gold) if pair_match(p, g, source)]
             for p in predictions]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return [(i, j) for j, i in owner.items()]


def span_present(text, phrase):
    src, sub = cosmetic_tokens(text), cosmetic_tokens(phrase)
    return bool(sub) and any(src[i:i + len(sub)] == sub for i in range(len(src) - len(sub) + 1))


def load_cases(path=DEFAULT_TEST):
    path = Path(path)
    cases = [json.loads(line) for line in path.open(encoding='utf-8-sig') if line.strip()]
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case ids: ' + str([x for x, n in Counter(ids).items() if n > 1]))
    for c in cases:
        if not all(k in c for k in ('id', 'input', 'level')):
            raise ValueError('Missing required case fields: ' + repr(c))
    return cases


def evaluate(c, answer, valid_rule_ids=None):
    """Evaluate a single case; does not require an API or lingua_core.

    valid_rule_ids: optional set from grammar_rules.json. If absent, rule_valid
    is omitted (never guessed from gold annotations).
    """
    m = {'id': c['id']}
    obj = parse_answer(answer)
    m['json_ok'] = obj is not None
    m['schema_ok'] = schema_valid(obj)
    if not m['schema_ok']:
        return m

    raw_pred = obj['corrections']
    # Cosmetic-only edits do not count as student grammar errors, even if Qwen
    # incorrectly emits them in corrections. Report them separately.
    pred = [x for x in raw_pred if not cosmetic_equal(x['original'], x['correct'])]
    cosmetic_pred_n = len(raw_pred) - len(pred)
    gold = gold_items(c)
    pairs = match_pairs(pred, gold, c['input'])
    m['gold_n'] = len(gold)
    m['pred_n'] = len(pred)
    m['raw_pred_n'] = len(raw_pred)
    m['cosmetic_pred_n'] = cosmetic_pred_n
    m['cosmetic_only_corrections'] = cosmetic_pred_n == 0
    m['matched_n'] = len(pairs)
    m['rule_matched_n'] = sum(pred[i]['rule_id'] in rule_ids(gold[j]) for i, j in pairs)
    m['false_positive_n'] = len(pred) - len(pairs)
    m['missed_n'] = len(gold) - len(pairs)
    m['noop_corr'] = sum(text_equal(p['original'], p['correct']) for p in pred)
    m['corrected_ok'] = cosmetic_tokens(obj['corrected']) in corrected_targets(c)
    m['corrections_exact'] = len(pairs) == len(gold) == len(pred) and m['noop_corr'] == 0
    m['corr_fix_ok'] = m['corrections_exact']  # backward-friendly alias, NEW all-errors meaning
    m['rule_ok'] = m['corrections_exact'] and m['rule_matched_n'] == len(gold)
    if not gold:
        m['no_false_alarm'] = len(pred) == 0
        m['no_user_facing_false_alarm'] = len(raw_pred) == 0
    m['all_spans_found'] = all(span_present(c['input'], p['original']) for p in raw_pred)
    if valid_rule_ids is not None:
        m['rule_valid'] = all(p['rule_id'] in valid_rule_ids for p in pred)
    m['precision'] = len(pairs) / len(pred) if pred else (1.0 if not gold else None)
    m['recall'] = len(pairs) / len(gold) if gold else None
    m['rule_accuracy'] = m['rule_matched_n'] / len(pairs) if pairs else None

    reply = obj['reply']
    skip = set(c.get('skip', []))
    m['reply_en'] = not bool(CYR.search(reply))
    if 'one_question' not in skip:
        m['one_question'] = reply.count('?') <= 2
    if 'len_ok' not in skip:
        m['len_ok'] = len(sentences(reply)) <= (3 if c['level'] in ('A1', 'A2') else 4)
    m['no_emoji_md'] = not bool(EMOJI_MD.search(reply))
    m['no_label'] = not bool(LABEL.search(reply))
    m['n_words'] = words(reply)
    # Safety is reported as a separate task, not as grammar accuracy.
    if c.get('expect_refusal'):
        m['safety_manual_review'] = True
    return m


def _summary(df):
    out = {'n': len(df)}
    for k in METRICS + ['precision', 'recall', 'rule_accuracy']:
        if k in df:
            v = pd.to_numeric(df[k], errors='coerce').dropna()
            out[k] = round(float(v.mean()), 4) if len(v) else None
    for k in ('gold_n', 'pred_n', 'matched_n', 'rule_matched_n', 'false_positive_n', 'missed_n', 'cosmetic_pred_n'):
        out[k] = int(df[k].fillna(0).sum()) if k in df else 0
    tp, fp, fn = out['matched_n'], out['false_positive_n'], out['missed_n']
    out['micro_precision'] = round(tp / (tp + fp), 4) if tp + fp else None
    out['micro_recall'] = round(tp / (tp + fn), 4) if tp + fn else None
    out['micro_f1'] = round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None
    out['micro_rule_accuracy'] = round(out['rule_matched_n'] / tp, 4) if tp else None
    if 'seconds' in df:
        v = pd.to_numeric(df['seconds'], errors='coerce').dropna()
        out['median_sec'] = round(float(v.median()), 2) if len(v) else None
    return out


def _base_dir():
    try:
        import lingua_core as L
        return Path(L.BASE)
    except ImportError:
        return Path.cwd()


def _make_request(L, c):
    """Bridge old lingua_core.build_request expecting mode and gold_corrections."""
    adapted = dict(c)
    adapted.setdefault('mode', 'scenario' if c.get('scenario_id') else 'free')
    adapted.setdefault('gold_corrections', [
        {**g, 'rule_id': list(rule_ids(g))} for g in gold_items(c)
    ])
    return L.build_request(adapted)


def run(name, use_fewshot=False, test_file=DEFAULT_TEST, only=None, answers=None,
        base_dir=None, valid_rule_ids=None):
    """Live run via lingua_core, or offline with answers mapping.

    answers: {case_id: answer_string OR (answer_string, seconds, tokens)}.
    Output: results_baseline_<name>.xlsx (summary, problems, results).
    """
    base = Path(base_dir) if base_dir is not None else _base_dir()
    test_path = Path(test_file)
    if not test_path.exists():
        test_path = base / test_file
    cases = load_cases(test_path)
    if only is not None:
        wanted = set(only)
        cases = [c for c in cases if c['id'] in wanted]
    if not cases:
        raise ValueError('No cases selected')
    L = None
    if answers is None:
        import lingua_core as L
        L.USE_FEWSHOT = use_fewshot
    if valid_rule_ids is None and L is not None and hasattr(L, 'RULES'):
        valid_rule_ids = {r['rule_id'] for r in L.RULES}
    rows = []
    for i, c in enumerate(cases, 1):
        if answers is not None:
            if c['id'] not in answers:
                raise KeyError(f'Missing saved answer for {c["id"]}')
            record = answers[c['id']]
            if isinstance(record, (tuple, list)):
                answer = record[0]
                sec = record[1] if len(record) > 1 else None
                ntok = record[2] if len(record) > 2 else None
            else:
                answer, sec, ntok = record, None, None
            prompt_chars = None
        else:
            system, history, user = _make_request(L, c)
            answer, sec, ntok = L.ask(system, user, history)
            prompt_chars = len(system)
        m = evaluate(c, answer, valid_rule_ids=valid_rule_ids)
        failed = [k for k in METRICS if m.get(k) is False]
        rows.append({
            'id': c['id'], 'kind': c.get('kind', ''), 'level': c['level'],
            'mode': 'scenario' if c.get('scenario_id') else 'free',
            'input': c['input'], 'expected_corrected': c.get('corrected', c['input']),
            'expected_corrections': json.dumps(gold_items(c), ensure_ascii=False),
            'answer': answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False),
            'failed': ', '.join(failed), 'seconds': sec, 'n_tokens': ntok,
            'prompt_chars': prompt_chars, **m,
        })
        print(f'[{i}/{len(cases)}] {c["id"]}: {"OK" if not failed else "FAIL: " + ", ".join(failed)}')
    df = pd.DataFrame(rows)
    groups = [('all', df)]
    for field in ('level', 'mode', 'kind'):
        groups += [(f'{field}:{v}', g) for v, g in df.groupby(field, dropna=False)]
    summary = pd.DataFrame([{'group': name, **_summary(group)} for name, group in groups])
    output = base / f'results_baseline_{name}.xlsx'
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='summary', index=False)
        df[df['failed'] != ''].to_excel(writer, sheet_name='problems', index=False)
        df.to_excel(writer, sheet_name='results', index=False)
    print('Saved:', output)
    return summary


def rescore(name, test_file=DEFAULT_TEST, base_dir=None):
    """Recalculate v7 metrics from previous results workbook; no model calls."""
    base = Path(base_dir) if base_dir is not None else _base_dir()
    old = pd.read_excel(base / f'results_baseline_{name}.xlsx', sheet_name='results')
    saved = {}
    for _, row in old.iterrows():
        answer = row.get('answer')
        if pd.isna(answer):
            continue
        saved[row['id']] = (str(answer), row.get('seconds'), row.get('n_tokens'))
    valid_ids = None
    rules_path = base / 'grammar_rules.json'
    if rules_path.exists():
        with rules_path.open(encoding='utf-8-sig') as f:
            rule_data = json.load(f)
        if isinstance(rule_data, list):
            valid_ids = {r['rule_id'] for r in rule_data if isinstance(r, dict) and 'rule_id' in r}
        elif isinstance(rule_data, dict):
            rules = rule_data.get('rules', [])
            valid_ids = {r['rule_id'] for r in rules if isinstance(r, dict) and 'rule_id' in r}
    return run(name + '_rescored_v72', test_file=test_file, only=saved.keys(),
               answers=saved, base_dir=base, valid_rule_ids=valid_ids)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='LinguaBot v7 evaluation')
    p.add_argument('--test-file', default=DEFAULT_TEST)
    p.add_argument('--rescore', help='Existing run name to rescore')
    p.add_argument('--name', default='v7')
    p.add_argument('--base-dir', default='.')
    args = p.parse_args()
    if args.rescore:
        rescore(args.rescore, test_file=args.test_file, base_dir=args.base_dir)
    else:
        run(args.name, test_file=args.test_file, base_dir=args.base_dir)

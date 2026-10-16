"""Feature grammar compiled to DuckDB SQL, the hand-crafted set, base features.

A feature name is `key|f<pct>|<window>|<agg>|<norm>|<red>`:
  key     user | item | ui:<item attr> (user's history over that attribute value) | iu:<user attr> (item's history in the user's group)
  f<pct>  only events with watched_pct >= pct
  window  d<N> last N days | all | hl<N> exponential decay with half-life N days
  agg     cnt (number of views) | dur (watch time) | pct (mean watched_pct) | rec (1 / days since last view)
  norm    raw | share (of the user's / group's total) | lift (share / global share)
  red     reduction over the item's attribute values: max | mean | sum, `one` when there is a single value
Every name is computed only from events with dt < fold cutoff.
"""
import hashlib
import os
import time

import numpy as np

from prep import FOLDS, MULTI, ONE, UATTR

FILTS = [0, 10, 50, 90]
WINS = ['d1', 'd3', 'd7', 'd14', 'd21', 'd30', 'd60', 'd90', 'all', 'hl3', 'hl7', 'hl14', 'hl30', 'hl90']
KEYS = ['user', 'item'] + [f'ui:{a}' for a in MULTI + ONE] + [f'iu:{a}' for a in UATTR]

CAT = ['age', 'income', 'sex', 'kids_flg', 'content_type', 'age_rating', 'for_kids']
BASE = CAT + ['release_year', 'knn_score', 'knn_rank', 'pop7_rank', 'pop30_rank']

HAND = [
    # user activity
    'user|f0|all|cnt|raw|one', 'user|f0|d7|cnt|raw|one', 'user|f0|d30|cnt|raw|one', 'user|f0|all|dur|raw|one',
    'user|f0|all|pct|raw|one', 'user|f0|all|rec|raw|one', 'custom:user_tenure',
    # item popularity and freshness
    'item|f0|d7|cnt|raw|one', 'item|f0|d14|cnt|raw|one', 'item|f0|d30|cnt|raw|one', 'item|f0|all|cnt|raw|one',
    'item|f0|d7|cnt|share|one', 'item|f50|d30|cnt|raw|one', 'item|f0|all|pct|raw|one', 'item|f0|all|rec|raw|one',
    'custom:item_age', 'custom:pop_trend',
    # user x item attributes (indirect links)
    'ui:genre|f0|all|cnt|share|mean', 'ui:genre|f0|all|cnt|share|max', 'ui:genre|f0|d30|cnt|share|max',
    'ui:genre|f50|all|cnt|lift|mean', 'ui:actor|f0|all|cnt|raw|sum', 'ui:director|f0|all|cnt|raw|max',
    'ui:studio|f0|all|cnt|share|max', 'ui:country|f0|all|cnt|share|max', 'ui:content_type|f0|all|cnt|share|one',
    'ui:content_type|f0|d30|cnt|share|one', 'ui:age_rating|f0|all|cnt|share|one', 'ui:decade|f0|all|cnt|share|one',
    'custom:year_diff',
    # item popularity inside the user's demographic group
    'iu:age|f0|d30|cnt|share|one', 'iu:sex|f0|d30|cnt|share|one', 'iu:income|f0|d30|cnt|share|one',
    'iu:kids_flg|f0|d30|cnt|share|one', 'iu:age|f0|d30|cnt|lift|one',
]
CUSTOM = ['custom:user_tenure', 'custom:item_age', 'custom:pop_trend', 'custom:year_diff']
STATS = {'groups': 0, 'seconds': 0.0}  # SQL feature-group computations actually run (cache misses)


def family(name):
    if name.startswith('user|') or name == 'custom:user_tenure':
        return 'user'
    if name.startswith('item|') or name in ('custom:item_age', 'custom:pop_trend'):
        return 'item'
    return 'pair'


def columns(key):
    """`agg|norm|red` columns produced by one SQL group of this key."""
    norms = {'user': ['raw'], 'item': ['raw', 'share']}.get(key, ['raw', 'share', 'lift'])
    reds = ['max', 'mean', 'sum'] if key.startswith('ui:') and key[3:] in MULTI else ['one']
    return ([f'{a}|{n}|{r}' for a in ('cnt', 'dur') for n in norms for r in reds]
            + [f'{a}|raw|{r}' for a in ('pct', 'rec') for r in reds if r != 'sum'])


def space():
    # rec ignores weights, so it is identical for every hl* window and `all`
    return [f'{k}|f{f}|{w}|{c}' for k in KEYS for f in FILTS for w in WINS for c in columns(k)
            if not (w.startswith('hl') and c.startswith('rec'))]


def group_sql(fold, key, filt, win, table=None):
    T, C = f"DATE '{FOLDS[fold]}'", table or f'cand_{fold}'
    w, cond = '1.0', f'dt < {T} AND pct >= {filt}'
    if win[0] == 'd':
        cond += f' AND dt >= {T} - INTERVAL {int(win[1:])} DAY'
    elif win.startswith('hl'):
        w = f"pow(0.5, date_diff('day', dt, {T}) / {int(win[2:])})"
    aggs = f"sum(w) cnt, sum(w * dur) dur, sum(w * pct) / sum(w) pct, 1.0 / min(date_diff('day', dt, {T})) rec"
    ctes = [f'E AS (SELECT *, {w} w FROM events WHERE {cond})', 'G AS (SELECT sum(w) cnt, sum(w * dur) dur FROM E)']
    share = lift = None
    if key == 'user':
        ctes.append(f'A AS (SELECT user_id, {aggs} FROM E GROUP BY 1)')
        join = 'LEFT JOIN A ON A.user_id = C.user_id'
    elif key == 'item':
        ctes.append(f'A AS (SELECT item_id, {aggs} FROM E GROUP BY 1)')
        join, share = 'LEFT JOIN A ON A.item_id = C.item_id', 'G.{a}'
    else:
        # ui: group the user's events by the item attribute value; iu: group the item's events by the user attribute value
        ent, other, attrs = ('user_id', 'item_id', 'item_attr') if key.startswith('ui:') else ('item_id', 'user_id', 'user_attr')
        ctes += [f"V AS (SELECT {other}, val FROM {attrs} WHERE attr = '{key[3:]}')",
                 f'R AS (SELECT E.*, V.val FROM E JOIN V USING ({other}))',
                 f'A AS (SELECT {ent}, val, {aggs} FROM R GROUP BY 1, 2)',
                 f'N AS (SELECT {ent}, sum(w) cnt, sum(w * dur) dur FROM E GROUP BY 1)',  # entity totals
                 'GV AS (SELECT val, sum(w) cnt, sum(w * dur) dur FROM R GROUP BY 1)']   # attribute-value totals
        join = (f'JOIN V ON V.{other} = C.{other} LEFT JOIN A ON A.{ent} = C.{ent} AND A.val = V.val '
                f'LEFT JOIN N ON N.{ent} = C.{ent} LEFT JOIN GV ON GV.val = V.val')
        share, lift = ('N.{a}', 'GV.{a} / G.{a}') if key.startswith('ui:') else ('GV.{a}', 'N.{a} / G.{a}')
    inner = {}
    for a in ('cnt', 'dur'):
        inner[f'{a}|raw'] = f'coalesce(A.{a}, 0)'
        if share:
            inner[f'{a}|share'] = f'coalesce(A.{a}, 0) / nullif({share.format(a=a)}, 0)'
        if lift:
            inner[f'{a}|lift'] = f'coalesce(A.{a}, 0) / nullif({share.format(a=a)}, 0) / nullif({lift.format(a=a)}, 0)'
    inner['pct|raw'], inner['rec|raw'] = 'A.pct', 'coalesce(A.rec, 0)'
    red = {'max': 'max', 'mean': 'avg', 'sum': 'sum', 'one': 'max'}
    cols = columns(key)
    sel_in = ', '.join(f'{e} "{k}"' for k, e in inner.items())
    sel_out = ', '.join(f'{red[c.split("|")[2]]}(P."{c.rsplit("|", 1)[0]}")::FLOAT "{c}"' for c in cols)
    ctes.append(f'P AS (SELECT C.rid, {sel_in} FROM {C} C {join} CROSS JOIN G)')
    return f'WITH {", ".join(ctes)} SELECT {sel_out} FROM {C} C LEFT JOIN P USING (rid) GROUP BY C.rid ORDER BY C.rid', cols


def custom_sql(fold, table=None):
    T, C = f"DATE '{FOLDS[fold]}'", table or f'cand_{fold}'
    return f"""WITH H AS (SELECT * FROM events WHERE dt < {T}),
    IT AS (SELECT item_id, date_diff('day', min(dt), {T}) item_age,
                  count(*) FILTER (WHERE dt >= {T} - INTERVAL 7 DAY) / nullif(count(*) FILTER (WHERE dt >= {T} - INTERVAL 30 DAY), 0) pop_trend
           FROM H GROUP BY 1),
    US AS (SELECT user_id, date_diff('day', min(dt), {T}) user_tenure, avg(release_year) user_year
           FROM H LEFT JOIN items USING (item_id) GROUP BY 1)
    SELECT US.user_tenure::FLOAT "custom:user_tenure", IT.item_age::FLOAT "custom:item_age", IT.pop_trend::FLOAT "custom:pop_trend",
           (items.release_year - US.user_year)::FLOAT "custom:year_diff"
    FROM {C} C LEFT JOIN IT ON IT.item_id = C.item_id LEFT JOIN US ON US.user_id = C.user_id
    LEFT JOIN items ON items.item_id = C.item_id ORDER BY C.rid""", CUSTOM


def group_of(name):
    return 'custom' if name.startswith('custom:') else name.rsplit('|', 3)[0]


def compute(con, fold, names, cache=None, table=None):
    """{name: float32 array aligned with `table` (default cand_<fold>) ordered by rid}.
    One SQL query per (key, filter, window) group; only the requested names are kept (a group has up to 22 columns),
    and they are cached as .npy when `cache` is given."""
    out, path = {}, None
    if cache:
        d = os.path.join(cache, 'feat', fold)
        os.makedirs(d, exist_ok=True)
        path = lambda n: os.path.join(d, hashlib.md5(n.encode()).hexdigest() + '.npy')
        out = {n: np.load(path(n)) for n in names if os.path.exists(path(n))}
    todo = {}
    for n in names:
        if n not in out:
            todo.setdefault(group_of(n), []).append(n)
    for g, ns in todo.items():
        t = time.time()
        if g == 'custom':
            df = con.sql(custom_sql(fold, table)[0]).df()
        else:
            key, f, win = g.split('|')
            df = con.sql(group_sql(fold, key, int(f[1:]), win, table)[0]).df()
        for n in ns:
            out[n] = df[n if g == 'custom' else n[len(g) + 1:]].to_numpy(np.float32, na_value=np.nan)
            if path:
                np.save(path(n), out[n])
        STATS['groups'] += 1
        STATS['seconds'] += time.time() - t
    return {n: out[n] for n in names}


def base(con, fold, table=None):
    """Candidate rows (ordered by rid) with label, static attributes and first-stage scores."""
    cats = ', '.join(f"coalesce({c}, 'unknown') {c}" for c in CAT)
    return con.sql(f"""SELECT C.user_id, C.item_id, C.label, {cats}, release_year, knn_score, knn_rank, pop7_rank, pop30_rank
        FROM {table or f'cand_{fold}'} C LEFT JOIN users USING (user_id) LEFT JOIN items USING (item_id) ORDER BY C.rid""").df()

"""No-leakage and correctness checks on synthetic data: python tests/test_leakage.py"""
import os
import sys
import tempfile
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import features as F  # noqa: E402
import prep  # noqa: E402

FOLD = 'val'
T = prep.FOLDS[FOLD]
rng = np.random.default_rng(0)
GENRES, ACTORS = [f'g{i}' for i in range(6)], [f'a{i}' for i in range(15)]


def make_events(n_users, items, start, end, n):
    days = (end - start).days
    df = pd.DataFrame({'user_id': rng.integers(0, n_users, n), 'item_id': rng.choice(items, n),
                       'last_watch_dt': [start + timedelta(int(d)) for d in rng.integers(0, days, n)],
                       'total_dur': rng.integers(1, 10_000, n), 'watched_pct': rng.integers(0, 101, n).astype(float)})
    return df.drop_duplicates(['user_id', 'item_id'])


users = pd.DataFrame({'user_id': np.arange(300), 'age': rng.choice(['age_18_24', 'age_25_34', None], 300),
                      'income': rng.choice(['low', 'high'], 300), 'sex': rng.choice(['М', 'Ж'], 300), 'kids_flg': rng.integers(0, 2, 300)})
items = pd.DataFrame({'item_id': np.arange(100), 'content_type': rng.choice(['film', 'series'], 100),
                      'release_year': rng.integers(1980, 2021, 100).astype(float),
                      'genres': [', '.join(rng.choice(GENRES, rng.integers(1, 3), replace=False)) if i % 10 else None for i in range(100)],
                      'countries': 'Россия', 'for_kids': np.nan, 'age_rating': rng.choice([0.0, 16.0, 18.0], 100),
                      'studios': [None if i % 3 else 's1' for i in range(100)], 'directors': rng.choice(['d1', 'd2', 'd3'], 100),
                      'actors': [', '.join(rng.choice(ACTORS, 3, replace=False)) for _ in range(100)]})
past = make_events(300, np.arange(80), date(2021, 6, 1), T + timedelta(prep.LABEL_DAYS), 6000)
# future: new pairs from the val cutoff on, including items 80..99 that never existed before T
future = make_events(300, np.arange(100), T, date(2021, 8, 23), 3000)
future = future.merge(past[['user_id', 'item_id']], how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')


def db(inter):
    con = duckdb.connect()
    con.register('inter_df', inter)
    con.register('users_df', users)
    con.register('items_df', items)
    prep.build(con, 'inter_df', 'users_df', 'items_df')
    prep.candidates(con, FOLD)
    return con


names = sorted(set(F.HAND) | set(rng.choice(F.space(), 400, replace=False)))
a = db(past)
fa = F.compute(a, FOLD, names, tempfile.mkdtemp())
b = db(pd.concat([past, future]))
cand_a, cand_b = (c.sql(f'SELECT * EXCLUDE (rid, label) FROM cand_{FOLD} ORDER BY user_id, item_id').df() for c in (a, b))

# 1) candidates of the original target users do not depend on the future
assert cand_b[cand_b.user_id.isin(cand_a.user_id)].reset_index(drop=True).equals(cand_a), 'candidates changed'
assert not cand_a.item_id.isin(range(80, 100)).any(), 'never-seen item became a candidate'

# 2) features computed with future events present are identical on the same candidate rows
b.register('ca', a.sql(f'SELECT * FROM cand_{FOLD}').df())
b.execute(f'CREATE OR REPLACE TABLE cand_{FOLD} AS SELECT * FROM ca')
fb = F.compute(b, FOLD, names, tempfile.mkdtemp())
bad = [n for n in names if not np.allclose(fa[n], fb[n], equal_nan=True, rtol=1e-5)]
assert not bad, f'leaky features: {bad[:5]}'

# 3) SQL matches a brute-force pandas computation
ev = a.sql('SELECT * FROM events').df()
ev['days'] = (pd.Timestamp(T) - pd.to_datetime(ev.dt)).dt.days
ev = ev[ev.days >= 1]
cand = a.sql(f'SELECT * FROM cand_{FOLD} ORDER BY rid').df()
ia = a.sql("SELECT * FROM item_attr WHERE attr = 'genre'").df()
ua = a.sql("SELECT * FROM user_attr WHERE attr = 'age'").df().set_index('user_id').val
genres = ia.groupby('item_id').val.apply(list)
ev_g = ev.merge(ia[['item_id', 'val']], on='item_id')
got = F.compute(a, FOLD, ['ui:genre|f0|d30|cnt|share|max', 'iu:age|f50|all|cnt|lift|one', 'item|f0|hl7|cnt|raw|one'],
                tempfile.mkdtemp())
e50 = ev[ev.pct >= 50].assign(age=lambda d: d.user_id.map(ua))
for r in cand.sample(200, random_state=0).itertuples():
    w = ev[(ev.user_id == r.user_id) & (ev.days <= 30)]
    g = ev_g[(ev_g.user_id == r.user_id) & (ev_g.days <= 30)]
    if r.item_id not in genres.index or len(w) == 0:
        want = np.nan
    else:
        want = max((g.val == x).sum() / len(w) for x in genres[r.item_id])
    assert np.isclose(got['ui:genre|f0|d30|cnt|share|max'][r.rid], want, equal_nan=True), ('ui', r, want)
    grp = e50[e50.age == ua[r.user_id]]
    share, gshare = (grp.item_id == r.item_id).sum() / len(grp), (e50.item_id == r.item_id).sum() / len(e50)
    want = share / gshare if gshare > 0 else np.nan
    assert np.isclose(got['iu:age|f50|all|cnt|lift|one'][r.rid], want, equal_nan=True, rtol=1e-4), ('iu', r, want)
    want = (0.5 ** (ev[ev.item_id == r.item_id].days / 7)).sum()
    assert np.isclose(got['item|f0|hl7|cnt|raw|one'][r.rid], want, rtol=1e-4), ('item', r, want)

print(f'ok: {len(cand_a)} candidates, {len(names)} features unchanged by {len(future)} future events; SQL == pandas')

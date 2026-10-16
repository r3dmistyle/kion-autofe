"""Data: download KION, sample users with full history, DuckDB tables, time folds, first-stage candidates."""
import io
import os
import urllib.request
import zipfile
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
URL = 'https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip'

# history < cutoff, labels in [cutoff, cutoff + LABEL_DAYS)
FOLDS = {'train3': date(2021, 7, 19), 'train2': date(2021, 7, 26), 'train': date(2021, 8, 2),
         'val': date(2021, 8, 9), 'test': date(2021, 8, 16)}
LABEL_DAYS = 7
N_KNN, N_POP, KNN_NEIGHBOURS = 100, 50, 100

MULTI = ['genre', 'actor', 'director', 'studio', 'country']  # item attributes with several values
ONE = ['content_type', 'age_rating', 'decade']
UATTR = ['age', 'income', 'sex', 'kids_flg']
MULTI_COL = {'genre': 'genres', 'actor': 'actors', 'director': 'directors', 'studio': 'studios', 'country': 'countries'}


def download():
    if os.path.exists(f'{RAW}/interactions.csv'):
        return
    os.makedirs(RAW, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    for n in ('interactions', 'users', 'items'):
        with open(f'{RAW}/{n}.csv', 'wb') as f:
            f.write(z.read(f'data_original/{n}.csv'))


def build(con, inter, users, items, n_users=0, seed=0):
    """inter/users/items are DuckDB relation expressions: a read_csv_auto(...) call or a registered DataFrame."""
    uids = con.sql(f'SELECT DISTINCT user_id FROM {inter} ORDER BY 1').fetchnumpy()['user_id']
    if n_users and n_users < len(uids):
        uids = np.sort(np.random.default_rng(seed).choice(uids, n_users, replace=False))
    con.register('sample_df', pd.DataFrame({'user_id': uids}))
    con.execute(f"""CREATE OR REPLACE TABLE events AS
        SELECT user_id, item_id, last_watch_dt::DATE dt, coalesce(total_dur, 0)::DOUBLE dur, coalesce(watched_pct, 0)::DOUBLE pct
        FROM {inter} WHERE user_id IN (SELECT user_id FROM sample_df)""")
    ucols = ', '.join(f"coalesce(u.{a}::VARCHAR, 'unknown') {a}" for a in UATTR)
    con.execute(f'CREATE OR REPLACE TABLE users AS SELECT s.user_id, {ucols} FROM sample_df s LEFT JOIN {users} u USING (user_id)')
    con.execute(f"""CREATE OR REPLACE TABLE items AS SELECT item_id, content_type, release_year::DOUBLE release_year,
        age_rating::INT::VARCHAR age_rating, for_kids::INT::VARCHAR for_kids, {', '.join(MULTI_COL.values())}
        FROM {items}""")
    parts = [f"SELECT item_id, '{a}' attr, trim(unnest(string_split({c}, ','))) AS val FROM items WHERE {c} IS NOT NULL"
             for a, c in MULTI_COL.items()]
    parts += ["SELECT item_id, 'content_type', content_type FROM items WHERE content_type IS NOT NULL",
              "SELECT item_id, 'age_rating', age_rating FROM items WHERE age_rating IS NOT NULL",
              "SELECT item_id, 'decade', (floor(release_year / 10) * 10)::INT::VARCHAR FROM items WHERE release_year IS NOT NULL"]
    con.execute(f"CREATE OR REPLACE TABLE item_attr AS SELECT DISTINCT * FROM ({' UNION ALL '.join(parts)}) WHERE val <> ''")
    con.execute('CREATE OR REPLACE TABLE user_attr AS ' + ' UNION ALL '.join(f"SELECT user_id, '{a}' attr, {a} AS val FROM users" for a in UATTR))
    con.unregister('sample_df')


def knn(X, k=KNN_NEIGHBOURS):
    """Item-item cosine on a binary user x item matrix, top-k neighbours per item."""
    d = sp.diags(1 / np.sqrt(np.maximum(np.asarray(X.sum(0)).ravel(), 1)))
    S = (X.T @ X).tocsr()
    S.setdiag(0)
    S = (d @ S @ d).tocsr()
    S.eliminate_zeros()
    for r in range(S.shape[0]):
        s, e = S.indptr[r], S.indptr[r + 1]
        if e - s > k:
            S.data[s + np.argpartition(S.data[s:e], -k)[:-k]] = 0
    S.eliminate_zeros()
    return S


def candidates(con, fold):
    """Table cand_<fold>: ItemKNN top-N_KNN + popular-7d/30d top-N_POP unseen items for each warm target user,
    with first-stage scores and the label. Table nrel_<fold>: number of label items per target user."""
    T = FOLDS[fold]
    hist = con.sql(f"SELECT user_id, item_id, dt FROM events WHERE dt < DATE '{T}'").df()
    users, items = np.unique(hist.user_id.to_numpy()), np.unique(hist.item_id.to_numpy())  # only items seen before T
    ui, ii = np.searchsorted(users, hist.user_id.to_numpy()), np.searchsorted(items, hist.item_id.to_numpy())
    X = sp.csr_matrix((np.ones(len(hist), np.float32), (ui, ii)), shape=(len(users), len(items)))
    S = knn(X)
    days = (pd.Timestamp(T) - pd.to_datetime(hist.dt)).dt.days.to_numpy()
    rank = lambda p: np.argsort(np.argsort(-p, kind='stable'), kind='stable')
    r7, r30 = (rank(np.bincount(ii[days <= d], minlength=len(items))) for d in (7, 30))
    top_pop = (r7 < N_POP) | (r30 < N_POP)

    lab = con.sql(f"""SELECT user_id, item_id FROM events
        WHERE dt >= DATE '{T}' AND dt < DATE '{T + timedelta(LABEL_DAYS)}'
          AND user_id IN (SELECT user_id FROM events WHERE dt < DATE '{T}')""").df()
    rows = np.searchsorted(users, np.unique(lab.user_id.to_numpy()))
    out = []
    for b in range(0, len(rows), 1000):
        r = rows[b:b + 1000]
        Xb = X[r]
        sc = (Xb @ S).toarray()
        sc[Xb.nonzero()] = -1  # seen items go last and are dropped
        krank = np.empty(sc.shape, np.int64)
        np.put_along_axis(krank, np.argsort(-sc, axis=1, kind='stable'), np.arange(sc.shape[1])[None], axis=1)
        u, i = np.nonzero((((krank < N_KNN) & (sc > 0)) | top_pop[None]) & (sc >= 0))
        s = sc[u, i]
        out.append(pd.DataFrame({'user_id': users[r][u], 'item_id': items[i], 'knn_score': s,
                                 'knn_rank': np.where(s > 0, krank[u, i], np.nan), 'pop7_rank': r7[i], 'pop30_rank': r30[i]}))
    cand = pd.concat(out).merge(lab.assign(label=1), how='left', on=['user_id', 'item_id'])
    cand['label'] = cand.label.fillna(0).astype(np.int8)
    con.register('cand_df', cand)
    con.register('lab_df', lab)
    con.execute(f'CREATE OR REPLACE TABLE cand_{fold} AS SELECT row_number() OVER (ORDER BY user_id, item_id) - 1 rid, * FROM cand_df')
    con.execute(f'CREATE OR REPLACE TABLE nrel_{fold} AS SELECT user_id, count(*) n_rel FROM lab_df GROUP BY 1 ORDER BY 1')
    con.unregister('cand_df')
    con.unregister('lab_df')


def connect(n_users, seed=0):
    """Open (and build on first use) the DuckDB database for one user sample."""
    d = os.path.join(ROOT, 'cache', f'n{n_users}_s{seed}')
    os.makedirs(d, exist_ok=True)
    con = duckdb.connect(os.path.join(d, 'kion.duckdb'))
    con.execute("SET memory_limit = '4GB'")  # leave RAM for pandas/CatBoost on a 12 GB Colab VM
    tables = set(con.sql('SHOW TABLES').fetchnumpy().get('name', []))
    if 'user_attr' not in tables:
        download()
        csv = lambda n: f"read_csv_auto('{RAW}/{n}.csv')"
        build(con, csv('interactions'), csv('users'), csv('items'), n_users, seed)
    for fold in FOLDS:
        if f'nrel_{fold}' not in tables:
            candidates(con, fold)
    return con, d


def nrel(con, fold):
    return con.sql(f'SELECT * FROM nrel_{fold}').df().set_index('user_id').n_rel

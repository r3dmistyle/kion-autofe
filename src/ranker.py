"""CatBoost ranker over candidate rows and ranking metrics."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.metrics import log_loss, roc_auc_score

from features import BASE, CAT, base, compute

FAST = dict(iterations=300, learning_rate=0.1, depth=6, early_stopping_rounds=30)    # search loop
FULL = dict(iterations=1000, learning_rate=0.1, depth=6, early_stopping_rounds=50)  # reported models
K = 10


def frame(con, fold, feats, cache=None, table=None):
    """Base columns + requested features for cand_<fold> (or `table`, a subset of its rows)."""
    df = base(con, fold, table)
    df[CAT] = df[CAT].astype('category')  # object strings cost GBs at 100k users and get copied into every Pool
    num = [c for c in BASE if c not in CAT]
    df[num] = df[num].astype(np.float32)
    return pd.concat([df, pd.DataFrame(compute(con, fold, feats, cache, table))], axis=1)


def train_rows(df):
    """Users without a single positive candidate carry no ranking signal."""
    return df[df.groupby('user_id').label.transform('max') > 0]


def _pool(df, cols, group=False):
    qid = df['qid'] if 'qid' in df else df.user_id  # stacked training weeks: one query per (week, user)
    return Pool(df[cols], df.label, cat_features=[c for c in cols if c in CAT], group_id=qid if group else None)


def fit(tr, va, feats, params=FULL, device='cpu', seed=0, loss='Logloss'):
    cols = BASE + list(feats)
    kw = dict(params, random_seed=seed, verbose=False, thread_count=-1, allow_writing_files=False, task_type='GPU' if device == 'gpu' else 'CPU')
    if loss == 'YetiRank':
        m = CatBoostRanker(loss_function='YetiRank', eval_metric=f'NDCG:top={K}', **kw)
        m.fit(_pool(tr, cols, True), eval_set=_pool(va, cols, True))
    else:
        m = CatBoostClassifier(loss_function='Logloss', **kw)
        m.fit(_pool(tr, cols), eval_set=_pool(va, cols))
    m.cols = cols
    return m


def predict(m, df):
    p = _pool(df, m.cols)
    return m.predict_proba(p)[:, 1] if isinstance(m, CatBoostClassifier) else m.predict(p)


def per_user(df, score, nrel, k=K):
    """Per-user NDCG@k, Recall@k, MAP@k against ALL label items of the user (nrel: Series user -> #labels),
    aligned with nrel.index; users absent from df score 0."""
    u, y = df.user_id.to_numpy(), df.label.to_numpy()
    o = np.lexsort((-np.asarray(score), u))
    u, y = u[o], y[o]
    start = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
    gid = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(u)]))
    rank = np.arange(len(u)) - start[gid]
    hit = ((rank < k) & (y > 0)).astype(float)
    n = nrel.reindex(u[start]).to_numpy()
    cum = np.cumsum(hit)
    cum_in_group = cum - np.r_[0, cum][start][gid]
    m = np.minimum(n, k)
    res = {
        'ndcg': np.bincount(gid, hit / np.log2(rank + 2), len(start)) / np.cumsum(1 / np.log2(np.arange(k) + 2))[m - 1],
        'recall': np.bincount(gid, hit, len(start)) / n,
        'map': np.bincount(gid, hit * cum_in_group / (rank + 1), len(start)) / m,
    }
    pos = np.searchsorted(nrel.index.to_numpy(), u[start])
    out = {}
    for name, v in res.items():
        full = np.zeros(len(nrel))
        full[pos] = v
        out[name] = full
    return out


def evaluate(df, score, nrel, prob=True):
    pu = per_user(df, score, nrel)
    res = {f'{k}@{K}': float(v.mean()) for k, v in pu.items()}
    res['auc'] = float(roc_auc_score(df.label, score))
    if prob:
        res['logloss'] = float(log_loss(df.label, np.clip(score, 1e-7, 1 - 1e-7)))
    return res, pu


if __name__ == '__main__':
    # perfect ranking -> 1, and a user with labels outside the candidates is penalised
    df = pd.DataFrame({'user_id': [1, 1, 1, 2, 2], 'label': [0, 1, 1, 1, 0]})
    pu = per_user(df, np.array([0.1, 0.9, 0.8, 0.7, 0.2]), pd.Series([2, 2], index=[1, 2]))
    assert np.allclose(pu['ndcg'][0], 1) and np.allclose(pu['recall'], [1, 0.5]) and np.allclose(pu['map'], [1, 0.5])
    pu = per_user(df, np.array([0.9, 0.1, 0.8, 0.7, 0.2]), pd.Series([2, 1, 3], index=[1, 2, 3]))
    assert np.allclose(pu['map'][0], (1 / 2 + 2 / 3) / 2) and pu['ndcg'][2] == 0 and np.allclose(pu['ndcg'][1], 1)
    print('ok')

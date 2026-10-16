"""AutoFE over the feature grammar, judged only on the validation fold:
1) screening: every grammar feature gets a CatBoost importance inside a random chunk of `chunk` features;
2) top `pool` by importance, de-duplicated by |corr| > 0.95;
3) greedy forward selection in importance order, keep a feature if val NDCG@10 grows by > eps;
4) local mutations: swap a selected feature for a grammar neighbour (other window/filter/agg/norm/reduction).
"""
import time

import numpy as np
import pandas as pd

import features as F
import ranker as R
from prep import nrel

ALTS = {1: [f'f{x}' for x in F.FILTS], 2: F.WINS, 3: ['cnt', 'dur', 'pct', 'rec'], 4: ['raw', 'share', 'lift'],
        5: ['max', 'mean', 'sum', 'one']}


def neighbours(name, space):
    p = name.split('|')
    return [n for i, vals in ALTS.items() for v in vals if v != p[i]
            for n in ['|'.join(p[:i] + [v] + p[i + 1:])] if n in space]


def search(con, device='cpu', train_users=None, val_users=None, chunk=200, pool=100, mut_fits=60, eps=5e-4, seed=0, log=print):
    rng = np.random.default_rng(seed)
    space = F.space()
    spset = set(space)
    # search tables: subsets of cand_train (users with a positive) and cand_val; nothing of the grammar is cached on disk
    for fold, users in (('train', train_users), ('val', val_users)):
        con.register('u_df', pd.DataFrame({'user_id': np.asarray(users if users is not None else [], np.int64)}))
        cond = 'TRUE' if users is None else 'user_id IN (SELECT user_id FROM u_df)'
        if fold == 'train':
            cond += ' AND user_id IN (SELECT user_id FROM cand_train GROUP BY 1 HAVING max(label) > 0)'
        con.execute(f'CREATE OR REPLACE TEMP TABLE s_{fold} AS SELECT * FROM cand_{fold} WHERE {cond}')
        con.unregister('u_df')
    tr, va = R.frame(con, 'train', [], table='s_train'), R.frame(con, 'val', [], table='s_val')
    nv = nrel(con, 'val')
    nv = nv[nv.index.isin(va.user_id.unique())]
    log(f'search rows: train {len(tr)} / {tr.user_id.nunique()} users, val {len(va)} / {len(nv)} users')

    def load(names):
        nonlocal tr, va
        new = [n for n in names if n not in tr]
        if new:
            tr = pd.concat([tr, pd.DataFrame(F.compute(con, 'train', new, table='s_train'))], axis=1)
            va = pd.concat([va, pd.DataFrame(F.compute(con, 'val', new, table='s_val'))], axis=1)

    def drop(names):
        nonlocal tr, va
        tr, va = (df.drop(columns=[n for n in names if n in df]) for df in (tr, va))

    trace = []

    def fit(feats, stage, feat=None):
        t = time.time()
        load(feats)
        m = R.fit(tr, va, feats, R.FAST, device, seed)
        ndcg = float(R.per_user(va, R.predict(m, va), nv)['ndcg'].mean())
        trace.append({'stage': stage, 'feature': feat, 'n_feats': len(feats), 'ndcg': ndcg, 'sec': time.time() - t})
        return m, ndcg

    # 1) screening: chunks made of whole SQL groups, so each group is computed once
    t1, sql0 = time.time(), F.STATS['seconds']
    groups = {}
    for n in space:
        groups.setdefault(F.group_of(n), []).append(n)
    order = list(groups)
    rng.shuffle(order)
    chunks, cur = [], []
    for g in order:
        cur += groups[g]
        if len(cur) >= chunk:
            chunks.append(cur)
            cur = []
    chunks += [cur] if cur else []
    imp = {}
    for i, ch in enumerate(chunks):
        m, _ = fit(ch, 'screen')
        fi = dict(zip(m.feature_names_, m.get_feature_importance()))
        imp.update({n: float(fi[n]) for n in ch})
        drop(ch)
        log(f'screen {i + 1}/{len(chunks)}  {time.time() - t1:.0f}s')
    t_screen, t_feat = time.time() - t1, F.STATS['seconds'] - sql0

    # 2) pool: top by importance without near-duplicates
    ranked = sorted(imp, key=imp.get, reverse=True)
    top = [n for n in ranked[:3 * pool] if imp[n] > 0]
    load(top)
    cidx = np.arange(len(va)) if len(va) <= 200_000 else np.sort(rng.choice(len(va), 200_000, replace=False))
    kept, Z = [], np.zeros((pool, len(cidx)), np.float32)
    for n in top:
        if len(kept) == pool:
            break
        x = va[n].to_numpy(np.float64)[cidx]
        x = np.where(np.isnan(x), np.nanmean(x) if not np.isnan(x).all() else 0, x)
        if x.std() == 0:
            continue
        z = ((x - x.mean()) / x.std()).astype(np.float32)
        if kept and np.abs(Z[:len(kept)] @ z).max() / len(z) > 0.95:
            continue
        Z[len(kept)] = z
        kept.append(n)
    del Z
    drop([n for n in top if n not in kept])

    # 3) greedy forward selection
    t2 = time.time()
    sel = []
    _, best = fit(sel, 'base')
    base_ndcg = best
    for n in kept:
        _, s = fit(sel + [n], 'greedy', n)
        if s > best + eps:
            sel, best = sel + [n], s
            trace[-1]['accepted'] = True
            log(f'+ {n}  ndcg={s:.4f}')
        else:
            drop([n])
    t_greedy = time.time() - t2

    # 4) mutations
    t3 = time.time()
    tried = set(kept)
    for _ in range(mut_fits if sel else 0):
        f = sel[rng.integers(len(sel))]
        cands = [n for n in neighbours(f, spset) if n not in tried and n not in sel]
        if not cands:
            continue
        n = cands[rng.integers(len(cands))]
        tried.add(n)
        trial = [n if x == f else x for x in sel]
        _, s = fit(trial, 'mutate', f'{f} -> {n}')
        if s > best + eps:
            sel, best = trial, s
            trace[-1]['accepted'] = True
            log(f'~ {f} -> {n}  ndcg={s:.4f}')
    t_mut = time.time() - t3

    return {'selected': sel, 'val_ndcg_search': best, 'val_ndcg_base_search': base_ndcg, 'pool': kept,
            'importance_top': {n: imp[n] for n in ranked[:300]}, 'trace': trace, 'space_size': len(space),
            'rows': {'train': len(tr), 'val': len(va)}, 'eps': eps, 'chunk': chunk,
            'seconds': {'screen_total': t_screen, 'screen_sql': t_feat, 'greedy': t_greedy, 'mutate': t_mut},
            'fits': {s: sum(t['stage'] == s for t in trace) for s in ('screen', 'base', 'greedy', 'mutate')}}


if __name__ == '__main__':
    space = set(F.space())
    nb = neighbours('ui:genre|f0|d7|cnt|share|max', space)
    assert 'ui:genre|f0|d14|cnt|share|max' in nb and 'ui:genre|f50|d7|cnt|share|max' in nb
    assert 'ui:genre|f0|d7|cnt|share|one' not in nb and all(n in space for n in nb)
    print(len(nb), 'neighbours ok')

"""Experiments. Every step writes results/n<N>/<step>.json.

  python src/experiments.py --n-users 100000 --device gpu --steps e0,hand,search,auto,windows
  python src/experiments.py --n-users 50000 --steps e0,hand,auto --search-from results/n100000   # scaling, reuse selected features
  python src/experiments.py --steps report                                                       # tables and plots over results/n*/
"""
import argparse
import glob
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import features as F
import prep
import ranker as R
import search as S


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == 'darwin' else r / 2 ** 10


def boot(a, b, n=1000, seed=0):
    """Paired bootstrap over users of the mean per-user difference."""
    d, rng = a - b, np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {'diff': float(d.mean()), 'ci95': [float(np.quantile(m, .025)), float(np.quantile(m, .975))]}


class Exp:
    def __init__(self, a):
        self.a = a
        t = time.time()
        self.con, self.cache = prep.connect(a.n_users, a.seed)
        self.prep_s = time.time() - t
        self.out = os.path.join(prep.ROOT, 'results', f'n{a.n_users}')
        os.makedirs(f'{self.out}/peruser', exist_ok=True)
        self.train_folds = ['train', 'train2', 'train3'][:a.train_weeks]
        self.nrel = {f: prep.nrel(self.con, f) for f in ('val', 'test')}
        self.env = {'device': a.device, 'host': platform.platform(), 'cpus': os.cpu_count(), 'n_users': a.n_users,
                    'train_weeks': a.train_weeks}

    def save(self, step, obj):
        obj = {**obj, 'env': self.env, 'peak_mb': peak_mb()}
        with open(f'{self.out}/{step}.json', 'w') as f:
            json.dump(obj, f, indent=1, ensure_ascii=False)

    def frames(self, feats):
        tr = pd.concat([R.train_rows(R.frame(self.con, f, feats, self.cache)).assign(qid=lambda d, i=i: d.user_id * 10 + i)
                        for i, f in enumerate(self.train_folds)], ignore_index=True)
        return tr, R.frame(self.con, 'val', feats, self.cache), R.frame(self.con, 'test', feats, self.cache)

    def models(self, specs):
        """specs: (name, feats, seeds, loss). Fit on the train weeks, early-stop on val, evaluate on val and test."""
        rows, imps = [], {}
        for name, feats, seeds, loss in specs:
            t = time.time()
            tr, va, te = self.frames(feats)
            t_frame = time.time() - t
            for s in seeds:
                t = time.time()
                m = R.fit(tr, va, feats, R.FULL, self.a.device, s, loss)
                t_fit = time.time() - t
                prob = loss == 'Logloss'
                rv, _ = R.evaluate(va, R.predict(m, va), self.nrel['val'], prob)
                rt, pu = R.evaluate(te, R.predict(m, te), self.nrel['test'], prob)
                np.save(f'{self.out}/peruser/{name}_s{s}.npy', pu['ndcg'].astype(np.float32))
                rows.append({'model': name, 'loss': loss, 'seed': s, 'n_feats': len(feats), 'trees': int(m.tree_count_),
                             'train_rows': len(tr), 'frame_s': t_frame, 'fit_s': t_fit,
                             **{f'val_{k}': v for k, v in rv.items()}, **{f'test_{k}': v for k, v in rt.items()}})
                print(name, s, f"val ndcg {rv['ndcg@10']:.4f} test ndcg {rt['ndcg@10']:.4f} fit {t_fit:.0f}s", flush=True)
            imps[name] = dict(zip(m.feature_names_, map(float, m.get_feature_importance(type='PredictionValuesChange'))))
        return rows, imps

    def cost(self, feats):
        """Seconds to compute a feature set for one fold (test) from an empty cache."""
        d = tempfile.mkdtemp()
        t = time.time()
        F.compute(self.con, 'test', feats, d)
        shutil.rmtree(d)
        return time.time() - t

    # ---- steps ----
    def e0(self):
        folds = {}
        for f in prep.FOLDS:
            r = self.con.sql(f"""SELECT (SELECT count(*) FROM cand_{f}), count(*), (SELECT sum(label) FROM cand_{f}), sum(n_rel),
                avg(coalesce(p, 0) / n_rel) FROM nrel_{f} LEFT JOIN (SELECT user_id, sum(label) p FROM cand_{f} GROUP BY 1) USING (user_id)""").fetchone()
            folds[f] = dict(zip(['cand_rows', 'users', 'cand_positives', 'label_items', 'cand_recall'], map(float, r)))
        rows = []
        for fold in ('val', 'test'):
            df = F.base(self.con, fold)
            pop = 1 - df.pop7_rank.to_numpy() / (df.pop7_rank.max() + 1)
            for name, score in (('popular_7d', pop), ('item_knn', df.knn_score.to_numpy() + 1e-6 * pop)):
                res, _ = R.evaluate(df, score, self.nrel[fold], prob=False)
                rows.append({'model': name, 'fold': fold, **res})
        # cold users of the test week (no history): popular-7d top-10
        T = prep.FOLDS['test']
        cold = self.con.sql(f"""SELECT user_id, item_id FROM events WHERE dt >= DATE '{T}' AND dt < DATE '{T}' + INTERVAL {prep.LABEL_DAYS} DAY
            AND user_id NOT IN (SELECT user_id FROM events WHERE dt < DATE '{T}')""").df()
        top = self.con.sql(f"""SELECT item_id FROM events WHERE dt >= DATE '{T}' - INTERVAL 7 DAY AND dt < DATE '{T}'
            GROUP BY 1 ORDER BY count(*) DESC, item_id LIMIT 10""").df().item_id.to_numpy()
        users = np.unique(cold.user_id.to_numpy())
        df = pd.DataFrame({'user_id': np.repeat(users, len(top)), 'item_id': np.tile(top, len(users))})
        df['label'] = df.merge(cold.assign(label=1), how='left', on=['user_id', 'item_id']).label.fillna(0).to_numpy()
        pu = R.per_user(df, -np.tile(np.arange(len(top)), len(users)), cold.groupby('user_id').size())
        n_events = self.con.sql('SELECT count(*) FROM events').fetchone()[0]
        self.save('e0', {'prep_s': self.prep_s, 'events': n_events, 'folds': folds, 'rows': rows,
                         'cold_test_popular': {'users': len(users), **{f'{k}@10': float(v.mean()) for k, v in pu.items()}}})

    def hand(self):
        seeds = list(range(self.a.seeds))
        specs = [('base', [], seeds, 'Logloss'), ('hand', F.HAND, seeds, 'Logloss'), ('hand_yetirank', F.HAND, [0], 'YetiRank')]
        specs += [(f'hand_minus_{fam}', [n for n in F.HAND if F.family(n) != fam], seeds, 'Logloss') for fam in ('user', 'item', 'pair')]
        rows, imps = self.models(specs)
        self.save('hand', {'rows': rows, 'importance': imps, 'hand_features': F.HAND})

    def search(self):
        rng = np.random.default_rng(self.a.seed)
        sub = lambda f: None if not self.a.search_users else np.sort(rng.choice(
            prep.nrel(self.con, f).index.to_numpy(), min(self.a.search_users, len(prep.nrel(self.con, f))), replace=False))
        res = S.search(self.con, self.a.device, sub('train'), sub('val'), self.a.chunk, self.a.pool,
                       self.a.mut_fits, self.a.eps, self.a.seed, log=lambda s: print(s, flush=True),
                       state_path=f'{self.out}/search_state.json')
        self.save('search', res)

    def auto(self):
        src = self.a.search_from or self.out
        with open(os.path.join(src, 'search.json')) as f:
            sres = json.load(f)
        sel, k = sres['selected'], len(F.HAND)
        seeds = list(range(self.a.seeds))
        specs = [('auto', sel, seeds, 'Logloss'), ('hand_plus_auto', F.HAND + [n for n in sel if n not in F.HAND], seeds, 'Logloss'),
                 ('auto_yetirank', sel, [0], 'YetiRank')]
        if len(sel) > k:  # same feature budget as the human
            specs.append((f'auto_top{k}', sorted(sel, key=lambda n: -sres['importance_top'].get(n, 0))[:k], seeds, 'Logloss'))
        rng = np.random.default_rng(self.a.seed)
        space = F.space()
        specs += [(f'random_{d}', list(rng.choice(space, k, replace=False)), [0], 'Logloss') for d in range(3)]
        rows, imps = self.models(specs)
        pu = lambda name: np.mean([np.load(f'{self.out}/peruser/{name}_s{s}.npy') for s in seeds], 0)
        pairs = [('auto', 'hand'), ('hand_plus_auto', 'hand'), ('hand', 'base'), ('auto', 'base')]
        boots = {f'{a}_vs_{b}': boot(pu(a), pu(b)) for a, b in pairs if os.path.exists(f'{self.out}/peruser/{b}_s0.npy')}
        cost = {'hand': self.cost(F.HAND), 'auto': self.cost(sel)}
        self.save('auto', {'rows': rows, 'importance': imps, 'selected': sel, 'search_from': src, 'bootstrap_test_ndcg': boots,
                           'feature_seconds_one_fold': cost})

    def windows(self):
        """Val/test NDCG of base + ONE feature while sweeping its window: is 7 days special?"""
        probes = ['item|f0|{w}|cnt|share|one', 'ui:genre|f0|{w}|cnt|share|max', 'iu:age|f0|{w}|cnt|share|one']
        names = [p.format(w=w) for p in probes for w in F.WINS]
        tr, va, te = self.frames(names)
        out = []
        for n in [None] + names:
            m = R.fit(tr, va, [n] if n else [], R.FAST, self.a.device, 0)
            out.append({'feature': n, 'val_ndcg': float(R.per_user(va, R.predict(m, va), self.nrel['val'])['ndcg'].mean()),
                        'test_ndcg': float(R.per_user(te, R.predict(m, te), self.nrel['test'])['ndcg'].mean())})
            print(out[-1], flush=True)
        self.save('windows', {'probes': probes, 'rows': out})


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = os.path.join(prep.ROOT, 'results')
    lines = []
    scale = []
    for d in sorted(glob.glob(f'{res}/n*/'), key=lambda p: int(p.rstrip('/').split('/n')[-1])):
        n = int(d.rstrip('/').split('/n')[-1])
        rows = []
        for step in ('hand', 'auto'):
            if os.path.exists(f'{d}/{step}.json'):
                rows += json.load(open(f'{d}/{step}.json'))['rows']
        if not rows:
            continue
        df = pd.DataFrame(rows)
        g = df.groupby('model', sort=False)
        tab = g.agg(n_feats=('n_feats', 'first'), seeds=('seed', 'count'),
                    val_ndcg=('val_ndcg@10', 'mean'), test_ndcg=('test_ndcg@10', 'mean'), test_ndcg_std=('test_ndcg@10', 'std'),
                    test_recall=('test_recall@10', 'mean'), test_map=('test_map@10', 'mean'), test_auc=('test_auc', 'mean'),
                    test_logloss=('test_logloss', 'mean'), fit_s=('fit_s', 'mean'))
        e0 = json.load(open(f'{d}/e0.json')) if os.path.exists(f'{d}/e0.json') else None
        lines.append(f'\n## n_users = {n}\n')
        if e0:
            for r in e0['rows']:
                lines.append(f"- {r['model']} ({r['fold']}): ndcg@10 {r['ndcg@10']:.4f}, recall@10 {r['recall@10']:.4f}, map@10 {r['map@10']:.4f}")
            lines.append(f"- candidate recall: " + ', '.join(f"{f} {v['cand_recall']:.3f}" for f, v in e0['folds'].items()))
        lines.append('\n' + tab.to_markdown(floatfmt='.4f') + '\n')
        for m in ('base', 'hand', 'auto', 'hand_plus_auto'):
            if m in tab.index:
                scale.append({'n': n, 'model': m, 'test_ndcg': tab.loc[m, 'test_ndcg'], 'fit_s': tab.loc[m, 'fit_s']})
        if os.path.exists(f'{d}/auto.json'):
            a = json.load(open(f'{d}/auto.json'))
            lines.append('bootstrap (test NDCG@10 diff, 95% CI): ' + json.dumps(a['bootstrap_test_ndcg']))
            lines.append('feature seconds per fold: ' + json.dumps(a['feature_seconds_one_fold']))
        if os.path.exists(f'{d}/windows.json'):
            w = json.load(open(f'{d}/windows.json'))
            base = w['rows'][0]['val_ndcg']
            fig, ax = plt.subplots(figsize=(8, 4))
            for p in w['probes']:
                ys = [next(r['val_ndcg'] for r in w['rows'] if r['feature'] == p.format(w=x)) for x in F.WINS]
                ax.plot(F.WINS, ys, marker='o', label=p.replace('{w}', '*'))
            ax.axhline(base, color='gray', ls='--', label='base only')
            ax.set_ylabel('val NDCG@10')
            ax.set_xlabel('window')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f'{res}/windows_n{n}.png', dpi=120)
        if os.path.exists(f'{d}/search.json'):
            s = json.load(open(f'{d}/search.json'))
            tr = [t for t in s['trace'] if t['stage'] != 'screen']
            best = np.maximum.accumulate([t['ndcg'] for t in tr])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([t['ndcg'] for t in tr], '.', alpha=.4, label='fit')
            ax.plot(best, label='best so far')
            ax.axvline(sum(t['stage'] in ('base', 'greedy') for t in tr) - .5, color='gray', ls=':', label='greedy | mutations')
            ax.set_xlabel('fit #')
            ax.set_ylabel('val NDCG@10 (search subsample)')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f'{res}/search_n{n}.png', dpi=120)
    if len({r['n'] for r in scale}) > 1:
        sc = pd.DataFrame(scale)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for m, g in sc.groupby('model'):
            ax[0].plot(g.n, g.test_ndcg, marker='o', label=m)
            ax[1].plot(g.n, g.fit_s, marker='o', label=m)
        ax[0].set(xscale='log', xlabel='sampled users', ylabel='test NDCG@10')
        ax[1].set(xscale='log', xlabel='sampled users', ylabel='fit seconds')
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{res}/scale.png', dpi=120)
    with open(f'{res}/summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-users', type=int, default=100_000, help='0 = all users')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    ap.add_argument('--seeds', type=int, default=3, help='CatBoost seeds per reported model')
    ap.add_argument('--train-weeks', type=int, default=3)
    ap.add_argument('--steps', default='e0,hand,search,auto,windows')
    ap.add_argument('--search-users', type=int, default=0, help='users per fold in the search loop, 0 = all')
    ap.add_argument('--chunk', type=int, default=200)
    ap.add_argument('--pool', type=int, default=100)
    ap.add_argument('--mut-fits', type=int, default=60)
    ap.add_argument('--eps', type=float, default=5e-4)
    ap.add_argument('--search-from', default='', help='dir with search.json to reuse (scaling runs)')
    ap.add_argument('--force', action='store_true', help='recompute steps whose results already exist')
    a = ap.parse_args()
    steps = a.steps.split(',')
    if steps == ['report']:
        return report()
    e = Exp(a)
    for s in steps:
        if os.path.exists(f'{e.out}/{s}.json') and not a.force:
            print(f'== {s}: {e.out}/{s}.json exists, skipped (--force to redo)', flush=True)
            continue
        t = time.time()
        print(f'== {s}', flush=True)
        getattr(e, s)()
        print(f'== {s} done in {time.time() - t:.0f}s, peak {peak_mb():.0f} MB', flush=True)


if __name__ == '__main__':
    main()

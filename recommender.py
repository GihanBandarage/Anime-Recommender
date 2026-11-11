"""Sparse, confidence-weighted collaborative filtering; no invented user ratings."""
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import sqlite3

import numpy as np
from scipy.sparse import csr_matrix


def rated_entries(entries):
    return {int(e['id']): float(e['score']) for e in entries
            if e.get('id') and 0 < float(e.get('score') or 0) <= 10
            and e.get('status') != 'plan_to_watch'}


@lru_cache(maxsize=1)
def _load_model(path, mtime, size):
    # Read-only mode must never silently create an empty database.
    with closing(sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)) as conn:
        rows = conn.execute('''SELECT lower(username), anime_id, AVG(score)
            FROM animelists WHERE score BETWEEN 1 AND 10
            AND username IS NOT NULL AND anime_id > 0
            GROUP BY lower(username), anime_id''').fetchall()
    users, items = {}, {}
    for name, aid, _ in rows:
        users.setdefault(name, len(users))
        items.setdefault(int(aid), len(items))
    ri = np.array([users[r[0]] for r in rows], dtype=int)
    ci = np.array([items[int(r[1])] for r in rows], dtype=int)
    values = np.array([r[2] for r in rows], dtype=float)
    matrix = csr_matrix((values, (ri, ci)), shape=(len(users), len(items)))
    counts = np.diff(matrix.indptr)
    means = np.asarray(matrix.sum(axis=1)).ravel() / np.maximum(counts, 1)
    centered = matrix.copy()
    centered.data -= np.repeat(means, counts)
    norms = np.sqrt(np.asarray(centered.multiply(centered).sum(axis=1)).ravel())
    mask = matrix.copy()
    mask.data[:] = 1
    return users, items, mask, centered, counts, norms


def collaborative(entries, username, db_path, limit=40):
    """Require 3 shared ratings and 3 supporting neighbors per prediction.

    Keep the historical matrix sparse and cache it until the database changes.
    The live user is a separate vector, never merged with their historical row.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        return []
    ratings = rated_entries(entries)
    if len(ratings) < 3:
        return []
    stat = path.stat()
    users, items, mask, centered, counts, norms = _load_model(
        str(path), stat.st_mtime_ns, stat.st_size)
    known = [(items[aid], score) for aid, score in ratings.items() if aid in items]
    if len(known) < 3:
        return []
    user_mean = float(np.mean(list(ratings.values())))
    cols, scores = zip(*known)
    vector = csr_matrix(([s - user_mean for s in scores], ([0] * len(cols), cols)),
                        shape=(1, len(items)))
    norm = np.linalg.norm(vector.data)
    if norm < 1e-8:
        return []
    overlap = np.asarray(mask[:, list(cols)].sum(axis=1)).ravel()
    similarity = (centered @ vector.T).toarray().ravel() / np.maximum(norms * norm, 1e-8)
    weights = np.maximum(similarity, 0) * overlap / (overlap + 25)
    weights[(overlap < 3) | (counts < 4)] = 0
    if username.casefold() in users:
        weights[users[username.casefold()]] = 0
    neighbors = np.flatnonzero(weights > 0)
    neighbors = neighbors[np.argsort(weights[neighbors])[-100:]]
    if len(neighbors) < 3:
        return []
    w = weights[neighbors]
    denom = np.asarray(mask[neighbors].T @ w).ravel()
    support = np.asarray(mask[neighbors].sum(axis=0)).ravel()
    deviation = np.asarray(centered[neighbors].T @ w).ravel()
    # Shrink sparse evidence back toward the user's mean, rather than letting
    # one enthusiastic neighbor produce a perfect 10.
    predictions = user_mean + deviation / (denom + 0.5)
    excluded = {int(e['id']) for e in entries if e.get('id')}
    out = []
    for aid, col in items.items():
        if aid in excluded or support[col] < 3 or denom[col] <= 0:
            continue
        out.append({'anime_id': aid, 'score': round(float(np.clip(predictions[col], 1, 10)), 2),
                    'support': int(support[col]), 'source': 'similar_viewers',
                    'reason': f'{int(support[col])} viewers with similar taste rated this anime.'})
    return sorted(out, key=lambda x: (-x['score'], -x['support'], x['anime_id']))[:limit]


def seeds(entries, limit=5):
    ratings = rated_entries(entries)
    if ratings:
        threshold = max(7, float(np.mean(list(ratings.values()))))
        liked = [e for e in entries if ratings.get(e['id'], 0) >= threshold
                 and e.get('status') in {'completed', 'watching'}]
        return sorted(liked, key=lambda e: (-float(e['score']), e['id']))[:limit]
    # Completion is weak implicit evidence, never a made-up numerical rating.
    return sorted((e for e in entries if e.get('status') == 'completed'),
                  key=lambda e: (e.get('updated_at', ''), e['id']), reverse=True)[:limit]


def community_candidates(seed_details, excluded):
    candidates = {}
    for seed, details in seed_details:
        recs = details.get('recommendations') or []
        total = sum(max(0, r.get('num_recommendations', 0)) for r in recs) or 1
        for rec in recs:
            node = rec.get('node') or {}
            aid = node.get('id')
            votes = max(0, rec.get('num_recommendations', 0))
            if not aid or aid in excluded or not votes:
                continue
            item = candidates.setdefault(aid, {'anime_id': aid, 'node': node,
                'weight': 0, 'because': [], 'source': 'community', 'score': None})
            # Normalize per seed so a single blockbuster doesn't dominate.
            item['weight'] += votes / total * (float(seed.get('score') or 5) / 10)
            item['because'].append(seed['title'])
    for item in candidates.values():
        item['reason'] = 'MAL readers recommend this for fans of ' + ', '.join(item['because'][:2]) + '.'
    return sorted(candidates.values(), key=lambda x: (-x['weight'], x['anime_id']))

import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import quote

from cachetools import TTLCache
from flask import Flask, g, has_request_context, jsonify, request
import requests

from recommender import collaborative, community_candidates, rated_entries, seeds

BASE = Path(__file__).resolve().parent
DB_PATH = os.environ.get('DB_PATH', str(BASE / 'anime.db'))
TOKEN_FILE = os.environ.get('TOKEN_FILE', str(BASE / 'token.json'))
app = Flask(__name__, static_folder='static', static_url_path='/')
log = logging.getLogger(__name__)
_cache = TTLCache(maxsize=1024, ttl=3600)
_cache_lock = threading.Lock()
_rate_lock = threading.Lock()
_next_request = 0.0
DETAIL_FIELDS = ('id,title,main_picture,synopsis,mean,rating,num_list_users,media_type,'
                 'genres,start_date,num_episodes,related_anime,recommendations')


class APIError(Exception):
    def __init__(self, message, status=502):
        self.message, self.status = message, status
        super().__init__(message)


@app.errorhandler(APIError)
def api_error(error):
    return jsonify(error=True, message=error.message), error.status


@app.before_request
def request_deadline():
    g.deadline = time.monotonic() + 85


@app.after_request
def response_headers(response):
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


def mal_headers():
    if client_id := os.environ.get('MAL_CLIENT_ID', '').strip():
        return {'X-MAL-CLIENT-ID': client_id}
    token = os.environ.get('MAL_TOKEN', '').strip()
    if not token:
        try:
            token = json.loads(Path(TOKEN_FILE).read_text()).get('access_token', '')
        except (OSError, ValueError, AttributeError):
            pass
    if not token:
        raise APIError('MyAnimeList is not configured. The server owner needs to set '
                       'MAL_CLIENT_ID or MAL_TOKEN. See the setup guide.', 503)
    return {'Authorization': f'Bearer {token}'}


def mal_get(path, params=None):
    global _next_request
    headers = mal_headers()
    # At most three outbound requests per second in each worker.
    with _rate_lock:
        time.sleep(max(0, _next_request - time.monotonic()))
        _next_request = time.monotonic() + 0.34
    remaining = g.deadline - time.monotonic() if has_request_context() else 85
    if remaining <= 0:
        raise APIError('MyAnimeList took too long to respond. Please try again shortly.', 504)
    try:
        response = requests.get('https://api.myanimelist.net/v2/' + path,
                                headers=headers, params=params,
                                timeout=(min(4, remaining / 2), min(12, remaining / 2)))
    except requests.RequestException as exc:
        raise APIError('MyAnimeList could not be reached. Please try again shortly.', 502) from exc
    if response.status_code in (401, 403):
        if response.status_code == 403 and path.startswith('users/'):
            raise APIError('This list is private or unavailable. Make your MAL anime list public and retry.', 403)
        raise APIError('MyAnimeList rejected the server credentials. The server owner needs to update them.', 503)
    if response.status_code == 404:
        raise APIError('MyAnimeList could not find that profile or anime. Check the username.', 404)
    if response.status_code == 429:
        raise APIError('MyAnimeList is busy. Please wait a moment before trying again.', 503)
    if response.status_code != 200:
        raise APIError('MyAnimeList returned an error. Please try again shortly.', 502)
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('Expected an object')
        return data
    except ValueError as exc:
        raise APIError('MyAnimeList returned an unreadable response. Please try again.', 502) from exc


def pull_animelist(username):
    entries = {}
    for offset in range(0, 30000, 1000):
        data = mal_get(f'users/{quote(username, safe="")}/animelist',
                       {'limit': 1000, 'offset': offset, 'fields': 'list_status', 'nsfw': 'true'})
        for entry in data.get('data', []):
            node, status = entry.get('node') or {}, entry.get('list_status') or {}
            if node.get('id'):
                entries[node['id']] = {'id': node['id'], 'title': node.get('title', 'Untitled'),
                    'score': status.get('score', 0) or 0, 'status': status.get('status', ''),
                    'updated_at': status.get('updated_at', '')}
        if not data.get('paging', {}).get('next'):
            return list(entries.values())
    raise APIError('This anime list is too large to process completely. No partial results were generated.', 422)


def mal_one(aid):
    with _cache_lock:
        cached = _cache.get(aid)
    if cached is not None:
        return cached
    data = mal_get(f'anime/{int(aid)}', {'fields': DETAIL_FIELDS})
    with _cache_lock:
        _cache[aid] = data
    return data


def username_arg():
    username = request.args.get('username', '').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{2,16}', username):
        raise APIError('Enter a valid MAL username: 2–16 letters, numbers, underscores or hyphens.', 400)
    return username


@app.get('/api/health')
def health():
    try:
        mal_headers()
        configured = True
    except APIError:
        configured = False
    return jsonify(ok=True, mal_configured=configured, database_available=Path(DB_PATH).is_file())


@app.get('/')
def root():
    return app.send_static_file('index.html')


@app.get('/api/animelist')
def api_animelist():
    return jsonify(error=False, list=pull_animelist(username_arg()))


@app.get('/api/recommendations')
def api_recs():
    username = username_arg()
    entries = pull_animelist(username)
    if not entries:
        raise APIError('This profile has an empty anime list. Add some shows on MAL, then try again.', 422)
    excluded = {e['id'] for e in entries}
    warnings = []
    try:
        candidates = collaborative(entries, username, DB_PATH)
    except (sqlite3.Error, OSError, ValueError):
        log.exception('Historical ratings could not be loaded')
        candidates = []
        warnings.append('The historical ratings database is unavailable; using MAL community recommendations.')
    seed_list = seeds(entries)
    method = 'similar_viewers'
    if not candidates:
        method = 'community'
        if not seed_list:
            raise APIError('There is not enough taste information yet. Rate a few shows you liked '
                           '(7/10 or higher), or complete some shows if you do not use ratings.', 422)
        seed_details = [(seed, mal_one(seed['id'])) for seed in seed_list]
        candidates = community_candidates(seed_details, excluded)
    completed = {e['id'] for e in entries if e['status'] == 'completed'}
    out = []
    metadata_missing = False
    for candidate in candidates[:40]:
        aid = candidate['anime_id']
        try:
            detail = mal_one(aid)
        except APIError as exc:
            if exc.status == 504 and out:
                warnings.append('MyAnimeList took too long; showing the matches found so far.')
                break
            if exc.status in (503, 502, 504):
                raise
            metadata_missing = True
            continue
        # Do not suggest sequels whose explicit prequel has not been completed.
        if any(r.get('relation_type') == 'prequel' and r.get('node', {}).get('id') not in completed
               for r in detail.get('related_anime', [])):
            continue
        if detail.get('rating') == 'rx':
            continue
        item = {key: detail.get(key) for key in ('id', 'title', 'main_picture', 'mean',
                'synopsis', 'media_type', 'num_episodes', 'genres', 'start_date', 'rating')}
        item.update(anime_id=aid, score=candidate['score'], source=candidate['source'],
                    reason=candidate['reason'], support=candidate.get('support'),
                    mal_url=f'https://myanimelist.net/anime/{aid}')
        out.append(item)
        if len(out) == 10:
            break
    if metadata_missing:
        warnings.append('Some anime information was unavailable; those entries were skipped.')
    message = None if out else ('No unseen matches were found with enough evidence. '
                                'Try again after rating more shows on MAL.')
    return jsonify(error=False, recommendations=out, username=username, method=method,
        message=message, warnings=warnings, profile={'total': len(entries),
        'rated': len(rated_entries(entries)), 'seeds': [e['title'] for e in seed_list]})


if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 5022)),
            debug=os.environ.get('FLASK_DEBUG') == '1')

# Next Arc — Anime recommendations

A Flask app that turns a public MyAnimeList profile into up to ten unseen anime recommendations, with posters, genres, synopses and a reason for each pick.

## Run locally

Requires Python 3.11 or newer.

```sh
git clone https://github.com/GihanBandarage/Anime-Recommender.git
cd Anime-Recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MAL_CLIENT_ID='your-registered-MAL-client-id'
python main.py
```

Open [localhost:5022](http://127.0.0.1:5022) and enter your MAL username. Your anime list must be public.

Register your own application in [MAL API configuration](https://myanimelist.net/apiconfig). Public list and anime reads support the `X-MAL-CLIENT-ID` header; see the [official API reference](https://myanimelist.net/apiconfig/references/api/v2). You do not need to log in as the profile you are recommending for.

Alternatively, set `MAL_TOKEN` to a valid OAuth access token. A local `token.json` with an `access_token` property is supported for compatibility. Credentials are never included in the repository or written by the Docker entrypoint. An expired token must be replaced. `MAL_CLIENT_ID` takes precedence if both are set.

`.env.example` documents the environment settings; the app does **not** automatically load `.env` files. Set variables in your shell or hosting environment.

## How recommendations work

- Only actual scores from 1–10 inform numerical predictions. Plan-to-watch entries never act as taste evidence. Unrated anime do not become invented 9/10 scores.
- Every title already on the live profile is excluded, including watching, dropped, on-hold and plan-to-watch entries.
- With a historical ratings database, a cached sparse user/item matrix compares mean-centered ratings. The live profile stays separate from historical data, and its old row is excluded case-insensitively. Neighbors need at least three overlapping ratings; a candidate needs three supporting neighbors. Similarities and predictions are shrunk toward a conservative baseline.
- Without usable collaborative matches, the app uses MAL community recommendations for up to five highly rated completed/watching shows. If there are no numerical ratings, recently completed shows provide weak implicit evidence. This fallback does not fabricate a predicted score and does not require the historical database.
- Titles with an explicit unfinished prequel and `rx` adult titles are filtered out. This is not a full franchise dependency resolver.
- Every result identifies its evidence. An estimated score is a heuristic, not a calibrated probability or a promise that you will like a show. Recommendation quality still needs evaluation on real users; passing tests does not establish taste accuracy.

Anime metadata is cached for one hour (up to 1,024 entries per worker). The live profile is fetched again for every recommendation request. Outbound MAL requests are limited to three per second per worker. For multiple workers/replicas, provide a shared rate limiter before scaling.

## Optional historical database

Set `DB_PATH` to a local SQLite file, or put `anime.db` next to `main.py`. It is opened read-only. No database download is needed for the community fallback. A missing/broken database never becomes a newly created empty database.

Required ratings table:

```sql
CREATE TABLE animelists (
    username TEXT,
    anime_id INTEGER,
    score REAL
);
```

Duplicate user/anime rows are averaged. Only valid 1–10 scores are loaded. Metadata comes from MAL; `anime_meta` is no longer required. The model cache refreshes when database modification time or size changes. Keep credentials and datasets out of Git.

## Docker / production

```sh
docker build -t next-arc .
docker run --rm -p 5022:5022 -e MAL_CLIENT_ID="$MAL_CLIENT_ID" next-arc
```

To add historical ratings, also pass `-v "$PWD/anime.db:/app/anime.db:ro"`.

The container runs as an unprivileged user with Gunicorn bound to `0.0.0.0` and honors `PORT`. It needs no writable token file. For local production serving:

```sh
gunicorn --bind 127.0.0.1:5022 --workers 1 --threads 4 --timeout 180 main:app
```

`HOST` and `PORT` configure the development server. Debug mode is off unless `FLASK_DEBUG=1` is explicitly set. Apply your hosting provider's request limits for a public deployment.

## Verification

```sh
python -m unittest discover -s tests -v
node --check static/app.js
```

Tests cover scoring evidence, list exclusions, sparse neighbor support, historical self-exclusion, community fallback, sequel filtering, pagination, caching, upstream errors and credential failures. Tests use isolated temporary databases and mocked upstream responses; they do not need a MAL credential or contact MAL.

`GET /api/health` reports process health, credential presence and database presence. Presence does not validate a credential or database schema. API errors return JSON with actionable messages, without upstream response bodies or local filesystem paths.

import os
import json
import time
import sys
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, make_response
import requests

print("refreshed ✅")

# === config ===
TOKEN_FILE = "token.json"
LIMIT = 100
SLEEP = 0.3

# put your real files here
ANIMELISTS_CLEANED = "animelists_cleaned.csv"
ANIME_META = "anime_cleaned.csv"

app = Flask(__name__, static_folder="static", static_url_path="/")

def no_cache(resp):
    # stop browsers caching API JSON
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def j(data, status=200):
    # jsonify + no-cache
    resp = make_response(json.dumps(data, ensure_ascii=False), status)
    resp.mimetype = "application/json"
    return no_cache(resp)

def load_token(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["access_token"]
    except Exception as e:
        print("Couldn't load access token:", e)
        sys.exit(1)

def get_anime_list(username, headers):
    # pull full MAL list for user (paged)
    offset = 0
    all_anime = []

    while True:
        url = f"https://api.myanimelist.net/v2/users/{username}/animelist"
        params = {
            "limit": LIMIT,
            "offset": offset,
            "fields": "list_status,num_episodes,status,score"
        }
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code != 200:
            return {"error": True, "status": r.status_code, "body": r.text}

        data = r.json()
        all_anime.extend(data.get("data", []))

        if "next" not in data.get("paging", {}):
            break

        offset += LIMIT
        time.sleep(SLEEP)

    # flatten for frontend
    flat = []
    for entry in all_anime:
        node = entry.get("node", {})
        ls = entry.get("list_status", {}) or {}
        flat.append({
            "id": node.get("id"),
            "title": node.get("title"),
            "score": ls.get("score", 0),
            "status": ls.get("status", "unknown"),
            "updated_at": ls.get("updated_at", "")
        })
    return {"error": False, "list": flat}

# === routes ===
@app.get("/api/health")
def health():
    info = {
        "ok": True,
        "cwd": os.getcwd(),
        "animelists_cleaned": str(Path(ANIMELISTS_CLEANED).absolute()),
        "anime_meta": str(Path(ANIME_META).absolute()),
        "exists_animelists": Path(ANIMELISTS_CLEANED).exists(),
        "exists_meta": Path(ANIME_META).exists(),
        "port_hint": 5002,
    }
    print("[/api/health]", info)
    return j(info)

@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/results.html")
def results_page():
    return app.send_static_file("results.html")
    return "ok"

@app.get("/api/animelist")
def api_animelist():
    username = request.args.get("username", "").strip()
    if not username:
        return j({"error": True, "message": "Missing ?username=..."}, 400)

    token = load_token(TOKEN_FILE)
    headers = {"Authorization": f"Bearer {token}"}
    result = get_anime_list(username, headers)

    if result.get("error"):
        return j({"error": True, "message": "MAL API failed", "details": result.get("body")}, result.get("status", 502))
    return j(result)

@app.get("/api/recommendations")
def api_recommendations():
    # small imports here to keep startup quick
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix

    username = request.args.get("username", "").strip()
    if not username:
        return j({"error": True, "message": "Missing ?username=..."}, 400)

    # show paths so it's obvious what files the server sees
    cwd = os.getcwd()
    a_abs = str(Path(ANIMELISTS_CLEANED).absolute())
    m_abs = str(Path(ANIME_META).absolute())
    print(f"[*] CWD={cwd}")
    print(f"[*] Checking files:\n    - {a_abs} (exists={Path(ANIMELISTS_CLEANED).exists()})\n    - {m_abs} (exists={Path(ANIME_META).exists()})")

    if not (Path(ANIMELISTS_CLEANED).exists() and Path(ANIME_META).exists()):
        return j({
            "error": True,
            "message": "Missing data files",
            "cwd": cwd,
            "animelists_cleaned": a_abs,
            "anime_meta": m_abs
        }, 500)

    # 1) get current user list
    token = load_token(TOKEN_FILE)
    headers = {"Authorization": f"Bearer {token}"}
    result = get_anime_list(username, headers)
    if result.get("error"):
        return j({"error": True, "message": "MAL API failed", "details": result.get("body")}, result.get("status", 502))

    flat = result["list"]

    # 2) my_list (only scored)
    my_rows = [
        {"username": username, "anime_id": x["id"], "score": x.get("score", 0)}
        for x in flat if x.get("id") is not None
    ]
    import pandas as pd
    my_list = pd.DataFrame(my_rows, columns=["username", "anime_id", "score"])
    my_list = my_list[my_list["score"] > 0]
    if my_list.empty:
        return j({"error": True, "message": "This user has no scored items."}, 400)

    # 3) load historical + meta
    everyone = pd.read_csv(ANIMELISTS_CLEANED)
    if "rating" in everyone.columns:
        everyone.rename(columns={"rating": "score"}, inplace=True)
    elif "my_score" in everyone.columns:
        everyone.rename(columns={"my_score": "score"}, inplace=True)
    everyone = everyone[["username", "anime_id", "score"]]
    everyone = everyone[everyone["score"] > 0]

    # keep active users (tweak threshold if coverage sucks)
    active = everyone["username"].value_counts()
    active = active[active > 30].index
    everyone = everyone[everyone["username"].isin(active)]

    # add current user
    everyone = pd.concat([everyone, my_list], ignore_index=True)

    # 4) matrix + sims (mean-centered)
    ratings = everyone.pivot_table(index="username", columns="anime_id", values="score")
    means = ratings.mean(axis=1)
    centered = ratings.sub(means, axis=0).fillna(0.0)
    sparse = csr_matrix(centered.values)

    if username not in ratings.index:
        return j({"error": True, "message": "User not present in rating matrix."}, 500)

    me_idx = ratings.index.get_loc(username)
    vec = sparse[me_idx]
    sims = cosine_similarity(vec, sparse).flatten()
    sim_series = pd.Series(sims, index=ratings.index).drop(index=username)

    # shrinked positive neighbors
    co_rated = ((ratings.notna()) & (ratings.loc[username].notna())).sum(axis=1).drop(index=username)
    beta = 25.0
    shrink = co_rated / (co_rated + beta)
    adj_sim = (sim_series.clip(lower=0) * shrink).sort_values(ascending=False)
    topk = adj_sim.head(100)
    if topk.empty:
        return j({"error": True, "message": "No similar users found."}, 200)

    # 5) predict on unseen
    seen_ids = set(my_list["anime_id"])
    preds = []

    nbr_means = means.loc[topk.index]
    w = topk

    for aid in ratings.columns:
        if aid in seen_ids:
            continue
        r = ratings.loc[topk.index, aid]
        dev = r - nbr_means
        m = dev.notna() & (w > 0)
        if m.sum() < 3:
            continue
        num = float((w[m] * dev[m]).sum())
        denom = float(w[m].abs().sum())
        if denom == 0:
            continue
        pred = means.loc[username] + (num / denom)
        pred = float(np.clip(pred, 1.0, 10.0))
        preds.append((int(aid), pred))

    if not preds:
        return j({"error": True, "message": "No predictable unseen items."}, 200)

    # 6) response
    meta = pd.read_csv(ANIME_META, usecols=["anime_id", "title"])
    id_to_title = dict(zip(meta["anime_id"], meta["title"]))

    preds.sort(key=lambda x: x[1], reverse=True)
    topn = preds[:10]
    results = [
        {"anime_id": aid, "title": id_to_title.get(aid, f"Anime ID {aid}"), "score": round(score, 2)}
        for aid, score in topn
    ]

    return j({"error": False, "recommendations": results})

if __name__ == "__main__":
    # run: http://127.0.0.1:5002/api/health
    app.run(debug=True, use_reloader=True, host="127.0.0.1", port=5002)
 
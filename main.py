import os, sys, json, time, sqlite3
from pathlib import Path
from flask import Flask, request, send_from_directory, make_response
import requests

print("boot ✅")

# --- config (keep it simple) ---
DB_PATH = "anime.db"
TOKEN_FILE = "token.json"
LIMIT = 100
PAUSE = 0.25     # chill a bit for MAL

app = Flask(__name__, static_folder="static", static_url_path="/")
print("this is just normal update")
def _nocache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def j(x, code=200):
    r = make_response(json.dumps(x, ensure_ascii=False), code)
    r.mimetype = "application/json"
    return _nocache(r)

def tok():
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)["access_token"]
    except Exception as e:
        print("no token:", e)
        sys.exit(1)

def H():  # headers for MAL
    return {"Authorization": f"Bearer {tok()}"}

# grab full user list (paged)
def pull_animelist(user):
    off = 0
    out = []
    while True:
        url = f"https://api.myanimelist.net/v2/users/{user}/animelist"
        params = {"limit": LIMIT, "offset": off, "fields": "list_status,num_episodes,status,score"}
        r = requests.get(url, headers=H(), params=params, timeout=20)
        if r.status_code != 200:
            return {"error": True, "status": r.status_code, "body": r.text}
        data = r.json()
        out += data.get("data", [])
        if "next" not in data.get("paging", {}):
            break
        off += LIMIT
        time.sleep(PAUSE)

    flat = []
    for e in out:
        n = (e.get("node") or {})
        ls = (e.get("list_status") or {})
        flat.append({
            "id": n.get("id"),
            "title": n.get("title"),
            "score": ls.get("score", 0) or 0,
            "status": (ls.get("status") or "").replace(" ", "_").lower(),
            "updated_at": ls.get("updated_at", "")
        })
    return {"error": False, "list": flat}

# tiny MAL detail fetch (one id)
def mal_one(aid):
    fields = "id,title,main_picture,synopsis,mean,rating,num_list_users,media_type,genres,start_date,end_date"
    try:
        r = requests.get(
            f"https://api.myanimelist.net/v2/anime/{aid}",
            headers=H(),
            params={"fields": fields},
            timeout=12
        )
        return r.json() if r.status_code == 200 else {"id": aid}
    except Exception:
        return {"id": aid}

@app.get("/api/health")
def health():
    return j({"ok": True, "db": str(Path(DB_PATH).absolute()), "exists": Path(DB_PATH).exists(), "port": 5022})

@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/results.html")
def results_page():
    return app.send_static_file("results.html")

@app.get("/api/animelist")
def api_animelist():
    u = request.args.get("username", "").strip()
    if not u: return j({"error": True, "message": "Missing ?username=..."}, 400)
    res = pull_animelist(u)
    if res.get("error"):
        return j({"error": True, "message": "MAL API failed", "details": res.get("body")}, res.get("status", 502))
    return j(res)

@app.get("/api/recommendations")
def api_recs():
    # imports local so this file stays one-piece
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix

    u = request.args.get("username", "").strip()
    if not u: return j({"error": True, "message": "Missing ?username=..."}, 400)

    got = pull_animelist(u)
    if got.get("error"):
        return j({"error": True, "message": "MAL API failed", "details": got.get("body")}, got.get("status", 502))

    # make user ratings, fake scores if 0 based on status
    rows = []
    for x in got["list"]:
        aid = x.get("id")
        if not aid: continue
        s = int(x.get("score", 0) or 0)
        st = x.get("status", "")
        if s == 0:
            if   st == "completed":     s = 9
            elif st == "watching":      s = 8
            elif st == "plan_to_watch": s = 7  # <- correct snake_case
            elif st == "on_hold":       s = 5
            elif st == "dropped":       s = 3
            else: continue
        rows.append({"username": u, "anime_id": int(aid), "score": float(s)})

    if not rows: return j({"error": True, "message": "This user has no usable anime."}, 400)

    import pandas as pd
    me = pd.DataFrame(rows, columns=["username", "anime_id", "score"])

    # load historical matrix
    conn = sqlite3.connect(DB_PATH)
    everyone = pd.read_sql_query("SELECT username, anime_id, score FROM animelists WHERE score > 0", conn)
    # keep users with >3 ratings (anti-noise)
    active = everyone["username"].value_counts()
    active = active[active > 3].index
    everyone = everyone[everyone["username"].isin(active)]
    # add me
    everyone = pd.concat([everyone, me], ignore_index=True)

    # pivot + sim
    R = everyone.pivot_table(index="username", columns="anime_id", values="score")
    mu = R.mean(axis=1)
    C = R.sub(mu, axis=0).fillna(0.0)
    S = csr_matrix(C.values)

    if u not in R.index: return j({"error": True, "message": "User not present in rating matrix."}, 500)

    i = R.index.get_loc(u)
    sim = cosine_similarity(S[i], S).flatten()
    sim = pd.Series(sim, index=R.index).drop(index=u)

    co = ((R.notna()) & (R.loc[u].notna())).sum(axis=1).drop(index=u)
    shrink = co / (co + 25.0)
    w = (sim.clip(lower=0) * shrink).sort_values(ascending=False).head(100)
    if w.empty: return j({"error": True, "message": "No similar users found."}, 200)

    # predict for unseen
    seen = set(me["anime_id"])
    preds = []
    nbr_mu = mu.loc[w.index]
    for aid in R.columns:
        if aid in seen: continue
        r = R.loc[w.index, aid]
        dev = r - nbr_mu
        m = dev.notna() & (w > 0)
        if m.sum() < 1: continue
        num = float((w[m] * dev[m]).sum())
        den = float(w[m].abs().sum())
        if den == 0: continue
        p = float(mu.loc[u] + num/den)
        p = max(1.0, min(10.0, p))
        preds.append((int(aid), p))

    if not preds: return j({"error": True, "message": "No predictable unseen items."}, 200)

    # titles from db
    meta = sqlite3.connect(DB_PATH).cursor()
    meta.execute("SELECT anime_id, title FROM anime_meta")
    id2title = {row[0]: row[1] for row in meta.fetchall()}

    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:10]

    # enrich w/ MAL so frontend can show posters etc.
    out = []
    for aid, ps in top:
        m = mal_one(aid)  # may be partial
        out.append({
            "id": m.get("id", aid),
            "anime_id": aid,
            "title": m.get("title", id2title.get(aid, f"Anime {aid}")),
            "score": round(ps, 2),                 # our predicted
            "mean": m.get("mean"),                 # MAL mean
            "rating": m.get("rating"),
            "num_list_users": m.get("num_list_users"),
            "synopsis": m.get("synopsis"),
            "main_picture": m.get("main_picture"), # {medium, large}
            "media_type": m.get("media_type"),
            "genres": m.get("genres"),
            "start_date": m.get("start_date"),
            "end_date": m.get("end_date"),
            "mal_url": f"https://myanimelist.net/anime/{aid}"
        })
        time.sleep(PAUSE)

    return j({"error": False, "recommendations": out})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, host="127.0.0.1", port=5022)

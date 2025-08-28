# Anime Recommendation System

This is a personal project where I built an **anime recommendation system** using the **official MyAnimeList API**. You enter your MAL username, the app fetches your anime list and scores, and then predicts anime you might want to watch next.

It uses **user-based collaborative filtering** with **cosine similarity** to compare your rating patterns and estimate scores for anime you haven’t seen yet. The results also include information from MAL, like the title, genres, poster, and synopsis, which the app displays in the frontend.

---

## Features
- Fetches your anime list and ratings from the **official MAL API**  
- Recommends anime using **user-based collaborative filtering**  
- Uses a local **SQLite** database with historical ratings for similarity calculations  
- Pulls extra info from MAL to display results in the frontend  
- Small **Flask** backend serving static HTML pages

---

## Tech Stack
- **Backend:** Flask, Pandas, NumPy, SQLite  
- **Algorithm:** User-based collaborative filtering + cosine similarity  
- **API:** Official MyAnimeList API  
- **Frontend:** Static HTML  
- **Optional:** Docker support

---

## Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/anime-recs.git
cd anime-recs
```

### 2. Install dependencies
Make sure you’re using **Python 3.10+**, then:
```bash
pip install -r requirements.txt
```

### 3. MAL API token  
I’ve **included a `token.json`** in the repo so you can run the project without setting up your own MAL OAuth token — it just works out of the box.  

If you want to use your own token instead, replace the file with:
```json
{
  "access_token": "YOUR_MAL_ACCESS_TOKEN"
}
```

### 4. Get the database + required files
Download **anime_data.zip** from **[Google Drive](<https://drive.google.com/file/d/1taEeNcN78G4cSR0fgpBYouTL4Dc4qL-o/view?usp=sharing>)** and extract it into the project root.

The archive contains everything you need:
- `anime.db` → Prebuilt SQLite database with historical ratings + metadata
- `token.json` → Preconfigured MAL API token so you can run the app without setting up your own
- CSV files used to build the database (optional but included for reference)


### 5. Run the app
```bash
python main.py
```
Then open **http://127.0.0.1:5022** in your browser.

---

## Using Docker (Optional)

If you’d rather not install Python or dependencies manually, you can run the app in Docker:

```bash
docker build -t anirec .
docker run --rm -p 5022:5022   -e MAL_TOKEN="your_mal_access_token_here"   -v "$(pwd)/anime.db:/app/anime.db:ro"   anirec
```

Then open **http://localhost:5022**.

---

## Future Plans
- Add better ranking using genres and popularity  
- Make it easier to rebuild the database  
- Improve the frontend  
- Possibly deploy a public demo

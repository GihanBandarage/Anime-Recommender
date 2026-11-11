"use strict";

const searchForm = document.querySelector("#search-form");
if (searchForm) {
  searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = searchForm.elements.username;
    input.value = input.value.trim();
    if (searchForm.reportValidity()) {
      window.location.assign(
        `/results.html?${new URLSearchParams({ username: input.value })}`,
      );
    }
  });
}

if (document.body.dataset.page === "results") {
  const username = (
    new URLSearchParams(location.search).get("username") || ""
  ).trim();
  const list = document.querySelector("#recommendations");
  const status = document.querySelector("#status");
  const retry = document.querySelector("#retry");
  const summary = document.querySelector("#profile-summary");
  let controller;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function externalLink(url, className, text) {
    const link = element("a", className, text);
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  }

  function card(anime, index) {
    const item = element("li", "anime-card");
    // Build markup with textContent: external metadata must never become HTML.
    const url = `https://myanimelist.net/anime/${Number(anime.anime_id)}`;
    const poster = externalLink(url, "poster-link");
    poster.setAttribute("aria-label", `View ${anime.title} on MyAnimeList`);
    poster.append(element("span", "poster-fallback", "Next Arc."));
    const imageUrl = anime.main_picture?.large || anime.main_picture?.medium;
    if (imageUrl) {
      try {
        const parsed = new URL(imageUrl);
        if (parsed.protocol === "https:") {
          const image = element("img");
          image.src = parsed.href;
          image.alt = "";
          image.loading = index < 5 ? "eager" : "lazy";
          image.addEventListener("error", () => image.remove(), { once: true });
          poster.append(image);
        }
      } catch {
        /* Keep the local fallback for invalid image URLs. */
      }
    }
    poster.append(element("span", "rank", String(index + 1).padStart(2, "0")));
    const info = element("div", "anime-info");
    info.append(
      element(
        "p",
        "anime-type",
        [
          (anime.media_type || "Anime").replaceAll("_", " "),
          anime.start_date?.slice(0, 4),
          anime.num_episodes ? `${anime.num_episodes} ep` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    const heading = element("h2", "anime-title");
    heading.append(externalLink(url, "", anime.title || "Untitled anime"));
    info.append(heading);
    const meta = element("div", "anime-meta");
    if (anime.score != null)
      meta.append(
        element(
          "span",
          "tag match-score",
          `Estimated ${Number(anime.score).toFixed(1)}/10`,
        ),
      );
    if (anime.mean != null)
      meta.append(
        element("span", "tag", `★ ${Number(anime.mean).toFixed(2)} MAL`),
      );
    for (const genre of (anime.genres || []).slice(0, 2))
      meta.append(element("span", "tag", genre.name));
    info.append(meta, element("p", "reason", anime.reason));
    if (anime.synopsis) {
      const details = element("details");
      details.append(
        element("summary", "", "Read synopsis"),
        element("p", "", anime.synopsis),
      );
      info.append(details);
    }
    info.append(externalLink(url, "mal-link", "Explore on MyAnimeList ↗"));
    item.append(poster, info);
    return item;
  }

  function showStatus(title, message, canRetry = false) {
    status.hidden = false;
    status.querySelector(".spinner").hidden = true;
    document.querySelector("#status-title").textContent = title;
    document.querySelector("#status-message").textContent = message;
    retry.hidden = !canRetry;
    list.setAttribute("aria-busy", "false");
  }

  async function load() {
    if (!/^[A-Za-z0-9_-]{2,16}$/.test(username)) {
      summary.textContent = "Start with your public MyAnimeList profile.";
      showStatus(
        "We need a username first",
        "Use “Try another profile” above to enter a valid MAL username.",
      );
      return;
    }
    controller?.abort();
    controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);
    summary.textContent = `Finding fresh picks for @${username}.`;
    document.title = `Picks for ${username} — Next Arc`;
    list.replaceChildren();
    list.setAttribute("aria-busy", "true");
    status.hidden = false;
    status.querySelector(".spinner").hidden = false;
    document.querySelector("#status-title").textContent =
      "Getting to know your taste";
    document.querySelector("#status-message").textContent =
      "Reading your MAL list and checking for unseen matches. This can take a little while.";
    retry.hidden = true;
    document.querySelector("#result-info").hidden = true;
    document.querySelector("#notice").hidden = true;
    try {
      const response = await fetch(
        `/api/recommendations?${new URLSearchParams({ username })}`,
        { signal: controller.signal },
      );
      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The server returned an unreadable response. Please try again.",
        );
      }
      if (!response.ok || data.error)
        throw new Error(
          data.message || "Could not load recommendations. Please try again.",
        );
      const recommendations = data.recommendations || [];
      summary.textContent = `Picked for @${username} · ${data.profile?.total ?? 0} shows on your list · ${data.profile?.rated ?? 0} rated. Everything on your list is excluded.`;
      if (!recommendations.length) {
        showStatus(
          "A little more taste to go on",
          data.message ||
            "No unseen matches yet. Rate more shows on MAL and try again.",
        );
        return;
      }
      list.append(...recommendations.map(card));
      status.hidden = true;
      document.querySelector("#result-info").hidden = false;
      document.querySelector("#result-count").textContent =
        `${recommendations.length} picks for your next arc`;
      document.querySelector("#method").textContent =
        data.method === "similar_viewers"
          ? "Based on similar viewers · Estimates are not guarantees"
          : "Community picks based on your favorites · No predicted scores";
      if (data.warnings?.length) {
        const notice = document.querySelector("#notice");
        notice.textContent = data.warnings.join(" ");
        notice.hidden = false;
      }
    } catch (error) {
      showStatus(
        "Couldn’t finish this arc",
        error.name === "AbortError"
          ? "This is taking longer than expected. Give it a moment and try again."
          : error.message || "Check your connection and try again.",
        true,
      );
    } finally {
      clearTimeout(timeout);
      list.setAttribute("aria-busy", "false");
    }
  }
  retry.addEventListener("click", load);
  load();
}

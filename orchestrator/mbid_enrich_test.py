"""music_enrich.py – minimal httpx‑based enrichment helper.

Designed for projects that already keep a shared *http_client_global* (httpx.AsyncClient)
so we re‑use that instead of opening new sessions.
"""

from typing import Dict, Any
import logging
import httpx

# Constants
MUSICBRAINZ_API_BASE_URL = "https://musicbrainz.org/ws/2/"
COVER_ART_ARCHIVE_BASE_URL = "https://coverartarchive.org/"
USER_AGENT = "SimpleMusicEnricher/0.2 (contact@example.com)"
LISTENBRAINZ_PLAYER_TMPL = "https://listenbrainz.org/player?recording_mbids={mbid}"
# NOTE: we expect *http_client_global* to be created at app start‑up
# (see snippet provided by caller)
http_client_global = httpx.AsyncClient(
    timeout=30.0,
    follow_redirects=True  # This will automatically handle 301 redirects
)

logger = logging.getLogger(__name__)


async def enrich_song(song: Dict[str, Any]) -> Dict[str, Any]:
    """Augment *song* dict with extended MusicBrainz metadata, cover art, and
    a *Play on ListenBrainz* link.

    New/updated keys added (always present):
    ─ full_title        – canonical track title
    ─ artist_name       – primary credited artist
    ─ length_ms         – recording length in milliseconds (``None`` if unavailable)
    ─ isrcs             – list[str] of ISRC codes
    ─ genres            – list[str] community‑agreed genres
    ─ tags              – list[str] top user tags/keywords
    ─ album_title       – title of the first associated release
    ─ release_date      – YYYY‑MM‑DD (may be partial) of that release
    ─ release_country   – two‑letter country code of release
    ─ release_status    – e.g. *official*, *promotion*, *bootleg*
    ─ release_type      – e.g. *single*, *album*, *compilation*
    ─ album_art_url     – 250×250 cover image or placeholder
    ─ listenbrainz_url  – direct LB player link for this recording
    """

    mbid = song.get("mbid")
    enriched: Dict[str, Any] = song.copy()

    # ---------- 1. Fill mandatory placeholders so callers see predictable keys
    defaults = {
        "full_title": song.get("title"),
        "artist_name": song.get("artist"),
        "length_ms": None,
        "isrcs": [],
        "genres": [],
        "tags": [],
        "album_title": None,
        "release_date": None,
        "release_country": None,
        "release_status": None,
        "release_type": None,
        "album_art_url": "https://via.placeholder.com/250?text=No+Art",
        "listenbrainz_url": None,
    }
    for k, v in defaults.items():
        enriched.setdefault(k, v)

    # Generate ListenBrainz URL early if we have an MBID (no network needed)
    if mbid:
        enriched["listenbrainz_url"] = LISTENBRAINZ_PLAYER_TMPL.format(mbid=mbid)
    else:
        logger.debug("No MBID for %s – skipping enrichment", song.get("title"))
        return enriched

    # ---------- 2. Hit MusicBrainz ------------------------------------------------
    rec_url = (
        f"{MUSICBRAINZ_API_BASE_URL}recording/{mbid}"
        "?inc="
        "artist-credits+releases+release-groups+isrcs+genres+tags"
        "&fmt=json"
    )
    try:
        print(f"Fetching MusicBrainz data for {mbid} from {rec_url}")  # Debug output
        resp = await http_client_global.get(rec_url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        #if redirect use the new URL
        if "redirect" in data:
            new_url = data["redirect"]
            print(f"Redirected to {new_url}, fetching again")
            resp = await http_client_global.get(new_url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            data = resp.json()
        if not data:
            logger.warning("No data found for MBID %s", mbid)
            return enriched
        print(f"MusicBrainz data for {mbid}: {data}")  # Debug output
    except Exception as exc:
        logger.warning("MusicBrainz lookup failed for %s: %s", mbid, exc)
        return enriched  # keep placeholders

    # ---------- 3. Core recording metadata ---------------------------------------
    enriched["full_title"] = data.get("title", enriched["full_title"])
    enriched["length_ms"] = data.get("length", enriched["length_ms"])
    enriched["isrcs"] = data.get("isrcs", enriched["isrcs"])

    if data.get("genres"):
        enriched["genres"] = [g.get("name") for g in data["genres"]]
    if data.get("tags"):
        enriched["tags"] = [t.get("name") for t in data["tags"]]

    if data.get("artist-credit"):
        enriched["artist_name"] = data["artist-credit"][0].get("name", enriched["artist_name"])

    # ---------- 4. Primary release info ------------------------------------------
    release: Optional[Dict[str, Any]] = data.get("releases", [{}])[0] if data.get("releases") else None
    if release:
        enriched["album_title"] = release.get("title")
        enriched["release_date"] = release.get("date")
        enriched["release_country"] = release.get("country")
        enriched["release_status"] = release.get("status")
        enriched["release_type"] = release.get("status")  # MusicBrainz stores type elsewhere, using status as proxy

    # ---------- 5. Cover art (release → release‑group fallback) -------------------
    art_url: Optional[str] = None

    # a) Try release‑specific art first
    if release and release.get("id"):
        rid = release["id"]
        candidate = f"{COVER_ART_ARCHIVE_BASE_URL}/release/{rid}/front-250"
        try:
            h = await http_client_global.head(candidate, follow_redirects=True)
            if h.status_code == 200:
                art_url = candidate
        except Exception:
            pass

    # b) Fallback: release‑group art
    if not art_url and data.get("release-groups"):
        rgid = data["release-groups"][0].get("id")
        if rgid:
            art_url = f"{COVER_ART_ARCHIVE_BASE_URL}/release-group/{rgid}/front-250"

    if art_url:
        enriched["album_art_url"] = art_url

    return enriched


if __name__ == "__main__":
    import asyncio

    # Example usage
    example_song = {
            "id": 465319,
            "mbid": "0b4e6193-eebb-4765-8a76-e49e70668eb8",
            "score": 0.9568,
            "title": "Liberian Girl",
            "artist": "Michael Jackson",
            "bpm": 105.158287048
        }

    async def main():
        enriched_song = await enrich_song(example_song)
        print(enriched_song)

    asyncio.run(main())
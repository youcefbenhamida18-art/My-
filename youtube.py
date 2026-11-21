"""YouTube Data API v3: resumable upload as private + publishAt = scheduled premiere at peak hours."""
import os
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

from .config import env, log


def service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"],
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YT_CLIENT_ID"], client_secret=os.environ["YT_CLIENT_SECRET"])
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def next_slots(state):
    """Two peak slots on the next free day: trailer at slot 1, full episode at slot 2 (UTC datetimes)."""
    tz = ZoneInfo(env("PUBLISH_TZ", "Asia/Riyadh"))
    s = [tuple(int(x) for x in v.split(":")) for v in env("PEAK_SLOTS", "19:00,21:30").split(",")]
    now = datetime.now(tz) + timedelta(hours=2)
    day = now.date()
    if state.get("last_day"):
        day = max(day, date.fromisoformat(state["last_day"]) + timedelta(days=1))
    while datetime.combine(day, dtime(*s[0]), tz) <= now:
        day += timedelta(days=1)
    t1, t2 = (datetime.combine(day, dtime(*s[i]), tz).astimezone(timezone.utc) for i in (0, 1))
    return t1, t2, day


def upload(yt, path, title, desc, tags, publish_at, thumb=None):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    body = {
        "snippet": {"title": title[:100], "description": desc[:4900], "tags": tags[:15], "categoryId": "1",
                    "defaultLanguage": "ar", "defaultAudioLanguage": "ja"},
        "status": {"privacyStatus": "private", "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": True},     # honest AI disclosure (art + voices are synthetic)
    }
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(path), chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        try:
            _, resp = req.next_chunk()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504):
                time.sleep(5)
                continue
            raise
    vid = resp["id"]
    log.info("uploaded %s -> https://youtu.be/%s (publishes %s)", path, vid, publish_at)
    if thumb:
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute()
        except Exception as e:  # noqa: BLE001  (needs a phone-verified channel + youtube scope)
            log.warning("thumbnail skipped: %s", e)
    return vid

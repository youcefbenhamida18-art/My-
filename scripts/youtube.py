"""Upload a video to YouTube using an OAuth refresh token. Videos are ALWAYS uploaded as private."""
import http.client
import json
import socket
import time

from common import env, log


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    cid, sec, rt = env("YT_CLIENT_ID"), env("YT_CLIENT_SECRET"), env("YT_REFRESH_TOKEN")
    if not (cid and sec and rt):
        raise RuntimeError("Missing secrets: YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    creds = Credentials(None, refresh_token=rt, token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid, client_secret=sec)
    try:
        creds.refresh(Request())
    except Exception as e:
        if "invalid_grant" in str(e):
            raise RuntimeError(
                "YouTube refresh token is expired or revoked (invalid_grant). If your Google Cloud OAuth app is in "
                "'Testing' mode the token dies after 7 days: set the app to 'In production', then generate a NEW "
                "refresh token and update the YT_REFRESH_TOKEN secret.") from e
        raise
    return creds


def check_credentials():
    """Fail fast (before spending an hour rendering) if the token does not work."""
    _credentials()
    log("  YouTube credentials OK")


def _clean(s: str, limit_bytes: int) -> str:
    s = (s or "").replace("<", " ").replace(">", " ")
    return s.encode("utf-8")[:limit_bytes].decode("utf-8", "ignore").strip()


def _explain(e) -> str:
    try:
        info = json.loads(e.content.decode("utf-8"))["error"]
        reasons = ",".join(x.get("reason", "") for x in info.get("errors", []))
        msg = f"{info.get('code')} {reasons}: {info.get('message')}"
    except Exception:
        msg = str(e)
    hint = ""
    if "quota" in msg.lower():
        hint = " -> YouTube API quota reached; try again tomorrow."
    elif "uploadLimitExceeded" in msg:
        hint = " -> daily upload limit of the channel reached; try again later."
    elif "insufficientPermissions" in msg or "forbidden" in msg.lower():
        hint = " -> the refresh token was created without the youtube.upload scope."
    return msg + hint


def upload(path, title: str, description: str, tags=None, category="27") -> str:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=_credentials(), cache_discovery=False)
    snippet = {"title": _clean(title, 95) or "Lesson", "description": _clean(description, 4800),
               "categoryId": category, "defaultLanguage": "en", "defaultAudioLanguage": "en"}
    if tags:
        snippet["tags"] = [t for t in tags if t][:15]
    body = {"snippet": snippet,
            "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False}}
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=16 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media, notifySubscribers=False)
    resp, fails = None, 0
    while resp is None:
        try:
            status, resp = req.next_chunk()
            if status:
                log(f"    uploading {int(status.progress() * 100)}%")
            fails = 0
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and fails < 8:
                fails += 1
                time.sleep(min(5 * 2 ** fails, 120))
                continue
            raise RuntimeError("YouTube upload failed: " + _explain(e)) from e
        except (OSError, http.client.HTTPException, socket.timeout) as e:
            fails += 1
            if fails > 8:
                raise
            log(f"    network hiccup ({type(e).__name__}), retrying")
            time.sleep(min(5 * 2 ** fails, 120))
    vid = resp["id"]
    privacy = resp.get("status", {}).get("privacyStatus")
    log(f"  uploaded: https://youtu.be/{vid}  (privacy: {privacy})")
    if privacy != "private":  # safety net: the lessons must never be public
        yt.videos().update(part="status", body={"id": vid, "status": {
            "privacyStatus": "private", "selfDeclaredMadeForKids": False}}).execute()
        log("  privacy forced to private")
    return vid

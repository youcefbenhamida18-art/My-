#!/usr/bin/env python3
"""
Run ONCE on your own computer to get a YouTube refresh token with the permissions the manager needs
(upload + thumbnails + comments + playlists + statistics).

  pip install google-auth-oauthlib
  python get_refresh_token.py          (needs client_secret.json next to this file)

Then put the printed token into the GitHub secret YT_REFRESH_TOKEN.
"""
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.upload",
]

flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
print("\nYT_CLIENT_ID     =", creds.client_id)
print("YT_CLIENT_SECRET =", creds.client_secret)
print("YT_REFRESH_TOKEN =", creds.refresh_token)

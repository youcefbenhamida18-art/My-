"""YouTube publishing (scheduled, private -> public at peak hours). Needs YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN."""
from __future__ import annotations

import os

from . import pipeline, youtube
from .config import cfg, load_state, log, save_state


def run(job=None, weekly=False):
    state = load_state()
    if cfg.dry_run or not os.getenv("YT_REFRESH_TOKEN"):
        log.warning("publish skipped (%s)", "DRY_RUN=1" if cfg.dry_run else "YT_REFRESH_TOKEN not set")
        return 0
    yt = youtube.service()
    t1, t2, day = youtube.next_slots(state)
    if weekly:
        ep = state["episodes"][-1]
        title, desc, tags = pipeline.describe(state, None, weekly=ep)
        youtube.upload(yt, ep["video"], title, desc, tags, t2)
    else:
        sh = next(s for s in reversed(state["shorts"]) if s["job"] == job)
        title, desc, tags = pipeline.describe(state, job)
        youtube.upload(yt, sh["video"], title, desc, tags, t1, thumb=sh.get("thumbnail"))
        if sh.get("vertical"):
            youtube.upload(yt, sh["vertical"], title[:90] + " #Shorts", desc, tags + ["Shorts"], t2)
    state["last_day"] = day.isoformat()
    save_state(state)
    return 0

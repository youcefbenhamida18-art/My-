import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from PIL import Image

from mangaforge import art
from mangaforge.art import layers, prompts, providers
from mangaforge.config import cfg
from tests.synth import key_art


def png_bytes(seed=1, figure=True):
    ok, buf = cv2.imencode(".png", key_art(seed=seed, figure=figure))
    return buf.tobytes()


class Horde(BaseHTTPRequestHandler):
    log = []
    posts = []
    checks = {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None, raw=None, ctype="application/json"):
        data = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        Horde.log.append(("GET", p))
        if p.endswith("/status/heartbeat"):
            return self._send(200, {"message": "OK"})
        if p.endswith("/status/models"):
            return self._send(200, [{"name": "Flux.1-Schnell fp8 (Compact)", "count": 5, "eta": 10},
                                    {"name": "stable_diffusion", "count": 9, "eta": 5},
                                    {"name": "Illustrious XL", "count": 3, "eta": 30},
                                    {"name": "Anything Diffusion", "count": 6, "eta": 20},
                                    {"name": "Dead Anime Model", "count": 0, "eta": 0}])
        if "/generate/check/" in p:
            jid = p.rsplit("/", 1)[1]
            Horde.checks[jid] = Horde.checks.get(jid, 0) + 1
            done = Horde.checks[jid] >= 2
            return self._send(200, {"done": done, "faulted": False, "is_possible": True, "queue_position": 2, "wait_time": 4})
        if "/generate/status/" in p:
            jid = p.rsplit("/", 1)[1]
            return self._send(200, {"done": True, "generations": [{"img": f"http://127.0.0.1:{self.server.server_port}/img/{jid}.png",
                                                                      "model": "Illustrious XL", "worker_name": "w1", "censored": False}]})
        if p.startswith("/img/"):
            return self._send(200, raw=png_bytes(seed=len(Horde.posts)), ctype="image/png")
        self._send(404, {"message": "nope"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        Horde.posts.append(body)
        if self.path.endswith("/generate/async"):
            if body["params"].get("post_processing"):
                return self._send(400, {"message": "Not enough kudos upfront for this request", "rc": "KudosUpfront"})
            return self._send(202, {"id": f"job{len(Horde.posts)}", "kudos": 12})
        self._send(404, {"message": "nope"})


@pytest.fixture()
def horde(monkeypatch):
    Horde.log, Horde.posts, Horde.checks = [], [], {}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Horde)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(cfg, "horde_url", f"http://127.0.0.1:{srv.server_port}/api/v2")
    monkeypatch.setattr(cfg, "horde_models", [])
    monkeypatch.setattr(cfg, "image_provider", "horde")
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    yield srv
    srv.shutdown()


SCRIPT = {"meta": {"episode": 1, "part": 1}, "style": "dark fantasy",
          "characters": [{"id": "kaito", "gender": "male", "look": "1boy, black hair, red scarf"}],
          "images": [{"id": "i01", "scene": "s1", "prompt": "a lone swordsman on a cliff at dusk", "characters": ["kaito"],
                      "framing": "wide", "palette": "orange dusk", "no_humans": False},
                     {"id": "i02", "scene": "s1", "prompt": "misty mountains at dawn", "characters": [], "framing": "wide",
                      "palette": "", "no_humans": True}]}


def test_prompt_never_names_shows_and_has_look():
    pos, neg = prompts.build_prompt(SCRIPT["images"][0], SCRIPT, "xl_anime")
    assert "red scarf" in pos and "text" in neg and len(pos) + len(neg) < 1000
    pos2, neg2 = prompts.build_prompt(SCRIPT["images"][1], SCRIPT, "xl_anime")
    assert "no humans" in pos2 and "1boy" in neg2


def test_horde_discovery_and_ladder(horde, tmp_path):
    p = providers.HordeProvider()
    assert p.models()[0] in ("Illustrious XL", "Anything Diffusion")
    assert not any("flux" in m.lower() or "dead" in m.lower() or m == "stable_diffusion" for m in p.models())
    pil, meta = p.generate(SCRIPT["images"][0], SCRIPT, seed=5)
    assert pil.size == (1344, 768) and meta["provider"] == "horde"
    posts = [b for b in Horde.posts]
    assert posts[0]["params"].get("post_processing") and not posts[-1]["params"].get("post_processing")   # ladder dropped upscale
    assert "###" in posts[0]["prompt"] and posts[0]["nsfw"] is False and posts[0]["models"][0] == p.models()[0]


def test_generate_art_end_to_end_and_resume(horde, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "art_target_w", 1920)
    res = art.generate_art(SCRIPT, tmp_path)
    assert set(res) == {"i01", "i02"}
    for r in res.values():
        for k in ("image", "backdrop", "depth", "sway", "rim"):
            assert (tmp_path / "art").exists() and __import__("pathlib").Path(r[k]).exists()
        assert 0 <= r["subject"]["x"] <= 1 and 0 <= r["subject"]["depth"] <= 1 and 0.55 <= r["ground_y"] <= 0.96
    n = len(Horde.posts)
    res2 = art.generate_art(SCRIPT, tmp_path)               # resume: nothing is generated again
    assert len(Horde.posts) == n and res2.keys() == res.keys()


def test_no_provider_raises_loudly_not_grey(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "image_provider", "comfyui")
    monkeypatch.setattr(cfg, "comfy_url", "http://127.0.0.1:9")
    monkeypatch.setattr(cfg, "allow_placeholder_art", False)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    with pytest.raises(art.ArtGenerationError):
        art.generate_art(SCRIPT, tmp_path)


def test_placeholder_is_opt_in_and_labelled(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "image_provider", "comfyui")
    monkeypatch.setattr(cfg, "comfy_url", "http://127.0.0.1:9")
    monkeypatch.setattr(cfg, "allow_placeholder_art", True)
    monkeypatch.setattr(cfg, "art_target_w", 1280)
    res = art.generate_art(SCRIPT, tmp_path)
    assert all(v["provider"] == "placeholder" for v in res.values())


def test_layers_semantics(tmp_path):
    p = tmp_path / "k.png"
    cv2.imwrite(str(p), key_art())
    m = layers.build_layers(p, tmp_path / "o", "k", target_w=1920)
    d = np.asarray(Image.open(m["depth"]))
    assert d.dtype == np.uint8 and d[-60:].mean() > d[:60].mean() + 20            # lower in frame = nearer
    sway = np.asarray(Image.open(m["sway"])).astype(float)
    h, w = sway.shape
    assert sway[int(h * .1):int(h * .45), :int(w * .3)].mean() > sway[int(h * .6):int(h * .8), int(w * .75):].mean() + 15   # foliage sways
    assert abs(m["subject"]["x"] - 0.5) < 0.12 and m["subject"]["y"] < 0.6          # face/eye level of the figure
    assert Image.open(m["image"]).size == (1920, 1080)
    bw, bh = Image.open(m["backdrop"]).size
    assert abs(bw / 640 - 1.5 * 1.5) < 0.2 or bw > 1920 * 0.7                       # 1.5x canvas of the analysis plate
    assert set(m["palette"]) >= {"avg", "top", "bottom", "left", "right", "accent"}


def test_fit_16x9_tall_panel(tmp_path):
    panel = np.full((1200, 800, 3), 200, np.uint8)
    out = layers.fit_16x9(panel)
    assert abs(out.shape[1] / out.shape[0] - 16 / 9) < 0.01

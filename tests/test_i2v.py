"""ComfyUI image-to-video hook against a fake server (placeholders, upload, queue, poll, download, decode to a sequence)."""
import json
import pathlib
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from mangaforge import i2v
from mangaforge.config import cfg

WF = {"1": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}},
      "2": {"class_type": "Txt", "inputs": {"text": "{{prompt}} !", "neg": "{{negative}}", "seed": "{{seed}}", "w": "{{width}}", "h": "{{height}}",
                                           "len": "{{frames}}", "fps": "{{fps}}", "link": ["1", 0]}}}


def test_fill_types_and_nesting():
    out = i2v.fill(WF, {"image": "a.png", "prompt": "hero", "negative": "bad", "seed": 7, "width": 768, "height": 448, "frames": 97, "fps": 24})
    assert out["1"]["inputs"]["image"] == "a.png"
    t = out["2"]["inputs"]
    assert t["text"] == "hero !" and t["seed"] == 7 and isinstance(t["seed"], int) and t["len"] == 97 and t["link"] == ["1", 0]
    assert WF["1"]["inputs"]["image"] == "{{image}}"                                    # the template is untouched


def test_sizes_and_frame_counts():
    assert i2v._size(1344, 768) == (832, 480) or i2v._size(1344, 768)[0] % 32 == 0
    w, h = i2v._size(1920, 1080)
    assert w % 32 == 0 and h % 32 == 0 and max(w, h) <= 832 + 32
    assert i2v._frames_for(4.0, 24) % 8 == 1 and i2v._frames_for(0.2, 24) == 9


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_comfy_round_trip(tmp_path, monkeypatch):
    img = tmp_path / "key.png"
    cv2.imwrite(str(img), np.full((270, 480, 3), 120, np.uint8))
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=2", "-pix_fmt", "yuv420p", str(clip)], check=True)
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(WF))
    monkeypatch.setattr(cfg, "i2v_backend", "comfyui")
    monkeypatch.setattr(cfg, "i2v_workflow", str(wf))
    monkeypatch.setattr(cfg, "comfy_url", "http://comfy.test")
    seen = {}

    class R:
        def __init__(self, j=None, content=b"", code=200):
            self._j, self.content, self.status_code, self.text = j, content, code, ""

        def json(self):
            return self._j

        def raise_for_status(self):
            assert self.status_code < 400

    def post(url, **kw):
        if url.endswith("/upload/image"):
            seen["upload"] = kw["files"]["image"][0]
            return R({"name": "uploaded.png"})
        seen["graph"] = kw["json"]["prompt"]
        return R({"prompt_id": "p1"})

    def get(url, **kw):
        if url.endswith("/system_stats"):
            return R({})
        if "/history/" in url:
            return R({"p1": {"status": {"status_str": "success"}, "outputs": {"9": {"videos": [{"filename": "o.mp4", "subfolder": "", "type": "output"}]}}}})
        if url.endswith("/view"):
            return R(content=clip.read_bytes())
        raise AssertionError(url)

    monkeypatch.setattr(i2v.requests, "post", post)
    monkeypatch.setattr(i2v.requests, "get", get)
    monkeypatch.setattr(i2v.time, "sleep", lambda s: None)
    seq = i2v.animate(str(img), "a samurai", tmp_path / "out", seconds=2.0, fps=12, seed=5)
    assert seq["fps"] == 12 and seq["pattern"] == "f_%04d.png" and 20 <= seq["frames"] <= 26            # 2 s at 12 fps
    assert (tmp_path / "out" / "f_0001.png").exists()
    g = seen["graph"]
    assert g["1"]["inputs"]["image"] == "uploaded.png" and g["2"]["inputs"]["seed"] == 5 and "a samurai" in g["2"]["inputs"]["text"]
    again = i2v.animate(str(img), "a samurai", tmp_path / "out", seconds=2.0, fps=12, seed=5)             # finished clips are reused
    assert again == seq


def test_off_and_missing_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "i2v_backend", "off")
    assert i2v.animate("x.png", "p", tmp_path / "o") is None
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(WF))
    monkeypatch.setattr(cfg, "i2v_backend", "comfyui")
    monkeypatch.setattr(cfg, "i2v_workflow", str(wf))
    monkeypatch.setattr(i2v.requests, "get", lambda *a, **k: (_ for _ in ()).throw(i2v.requests.ConnectionError("down")))
    assert i2v.animate("x.png", "p", tmp_path / "o2") is None
    monkeypatch.setattr(cfg, "i2v_workflow", str(tmp_path / "nope.json"))
    with pytest.raises(i2v.I2VError):
        i2v.animate("x.png", "p", tmp_path / "o3")

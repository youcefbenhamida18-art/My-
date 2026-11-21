"""Key-art generation (Cloudflare Workers AI) + monocular depth for 2.5D parallax."""
import base64
import concurrent.futures as cf
import gc
import os

import cv2
import numpy as np
import requests

from .config import env, log, retry

MODELS = env("CF_IMAGE_MODELS",
             "@cf/bytedance/stable-diffusion-xl-lightning,@cf/black-forest-labs/flux-1-schnell").split(",")
NEG = ("text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, extra fingers, "
       "deformed hands, bad anatomy, duplicate, cropped")
TAIL = "cinematic composition, dramatic rim lighting, volumetric fog, highly detailed, sharp focus, masterpiece"
_depth = None


def build_prompt(bible, scene):
    looks = "; ".join(f"{c['id']}: {c['look']}" for c in bible["characters"] if c["id"] in scene.get("present", []))
    return f"{bible['art_style']}, {scene['setting']}. {looks}. {TAIL}"[:900]


@retry(times=3, base=3)
def _generate(prompt):
    acc, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    last = None
    for m in MODELS:
        body = {"prompt": prompt}
        if "xl" in m:
            body.update(negative_prompt=NEG, width=1344, height=768, num_steps=6)
        else:
            body.update(steps=6)
        r = requests.post(f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{m.strip()}",
                          headers={"Authorization": f"Bearer {tok}"}, json=body, timeout=180)
        if r.status_code == 200:
            data = r.content if r.headers.get("content-type", "").startswith("image") \
                else base64.b64decode(r.json()["result"]["image"])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        last = (m, r.status_code, r.text[:160])
        log.warning("image model failed: %s", last)
    raise RuntimeError(last)


def prep(img):
    """Upscale + gentle unsharp so the camera can push in without mush."""
    h, w = img.shape[:2]
    s = max(1.0, 2560 / w)
    if s > 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    return cv2.addWeighted(img, 1.35, blur, -0.35, 0)


def depth_map(bgr):
    """Depth-Anything-V2-Small on CPU; falls back to a synthetic 'subject in the middle' depth."""
    global _depth
    h, w = bgr.shape[:2]
    try:
        if _depth is None:
            from transformers import pipeline
            _depth = pipeline("depth-estimation", model=env("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"),
                              device=-1)
        from PIL import Image
        out = _depth(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))["predicted_depth"]
        d = out.squeeze().cpu().numpy().astype("float32")
    except Exception as e:  # noqa: BLE001
        log.warning("depth fallback (%s)", e)
        yy, xx = np.mgrid[0:h, 0:w].astype("float32")
        d = 1.0 - np.sqrt(((xx / w - 0.5) * 1.3) ** 2 + (yy / h - 0.45) ** 2)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.GaussianBlur(d, (0, 0), 2.5)


def release_depth():
    global _depth
    _depth = None
    gc.collect()


def generate(bible, scenes, outdir):
    """Returns [(image_path, depth_path)] aligned with scenes."""
    outdir.mkdir(parents=True, exist_ok=True)

    def one(i):
        try:
            return i, _generate(build_prompt(bible, scenes[i]))
        except Exception as e:  # noqa: BLE001
            log.error("scene %d image failed: %s", i, e)
            return i, None

    raw = {}
    with cf.ThreadPoolExecutor(3) as ex:
        for i, img in ex.map(one, range(len(scenes))):
            raw[i] = img
    paths, last_ok = [], None
    for i in range(len(scenes)):
        img = raw[i]
        if img is None:
            if last_ok is None:      # nothing to reuse yet: dark gradient placeholder
                img = np.zeros((768, 1344, 3), np.uint8) + np.linspace(10, 60, 768, dtype=np.uint8)[:, None, None]
            else:
                paths.append(last_ok)
                continue
        small = img
        up = prep(img)
        ip, dp = outdir / f"img_{i:03d}.jpg", outdir / f"depth_{i:03d}.png"
        cv2.imwrite(str(ip), up, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(dp), (depth_map(small) * 255).astype(np.uint8))
        last_ok = (str(ip), str(dp))
        paths.append(last_ok)
        log.info("art %d/%d ready", i + 1, len(scenes))
    release_depth()
    return paths

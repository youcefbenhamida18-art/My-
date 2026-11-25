"""Manga ink extraction / tracing / registration on a synthetic page (pen strokes + halftone tone + solid black)."""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from mangaforge import lineart


def make_page(w=900, h=1300, seed=3):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), 0, 4)                                  # panel frame
    cv2.ellipse(img, (300, 420), (140, 190), 0, 0, 360, 0, 4, cv2.LINE_AA)                # a head outline (closed loop)
    cv2.line(img, (120, 700), (760, 740), 0, 5, cv2.LINE_AA)                              # a long stroke
    cv2.polylines(img, [np.array([(520, 200), (600, 260), (640, 340), (700, 380)])], False, 0, 4, cv2.LINE_AA)
    yy, xx = np.mgrid[0:h, 0:w]
    dots = ((yy % 6 - 3) ** 2 + (xx % 6 - 3) ** 2) < 2.4                                 # halftone dots (period 6 px)
    tone = np.zeros((h, w), bool)
    tone[800:1150, 120:520] = True
    img[tone & dots] = 0                                                                  # screentone patch
    cv2.rectangle(img, (600, 900), (820, 1180), 0, -1)                                    # a solid black shape (fill)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)                                              # the page was scanned / downscaled
    return img


@pytest.fixture(scope="module")
def page():
    return make_page()


def test_ink_keeps_lines_drops_tone_and_splits_fills(page):
    stroke, fill = lineart.ink_mask(page)
    H, W = page.shape
    assert stroke[420 - 190:420 + 190, 300 - 140:300 + 140].sum() > 1500            # the head outline is there
    assert stroke[700:745, 120:760].sum() > 1200                                     # so is the long stroke
    tone_area = stroke[820:1130, 140:500]
    assert tone_area.mean() < 0.01                                                   # halftone dots are gone
    assert fill[920:1160, 620:800].mean() > 0.6                                      # the solid black block is a fill
    assert stroke[920:1160, 620:800].mean() < 0.2                                    # ... not a stroke


def test_thinning_gives_one_pixel_centre_lines(page):
    stroke, _ = lineart.ink_mask(page)
    sk = lineart.thin(stroke)
    assert 0 < sk.sum() < stroke.sum() * 0.5
    two_by_two = sk[:-1, :-1] & sk[1:, :-1] & sk[:-1, 1:] & sk[1:, 1:]
    assert two_by_two.sum() < 0.01 * sk.sum()                                         # no fat blobs left
    ys, xs = np.nonzero(sk[700:745, 120:760])
    cy = np.polyval(np.polyfit(xs + 120, ys + 700, 1), xs + 120)
    assert np.abs(cy - (ys + 700)).max() < 3.0                                       # sits on the straight stroke's centre line


def test_trace_returns_long_strokes_with_widths(page):
    stroke, _ = lineart.ink_mask(page)
    data = lineart.trace(stroke)
    H, W = page.shape
    assert data["size"] == [W, H] and len(data["strokes"]) >= 3
    assert json.loads(json.dumps(data)) == data
    longest = data["strokes"][0]
    assert len(longest["p"]) >= 2 and len(longest["p"]) == len(longest["w"])
    ws = np.concatenate([s["w"] for s in data["strokes"]]) * W
    assert 2.0 < np.median(ws) < 9.0                                                  # ~ the 4-5 px pen
    allp = np.concatenate([np.asarray(s["p"]) for s in data["strokes"]])
    assert allp.min() >= 0 and allp.max() <= 1
    # the long stroke is traced as (nearly) one polyline spanning its length
    spans = [(np.asarray(s["p"])[:, 0].max() - np.asarray(s["p"])[:, 0].min()) * W for s in data["strokes"]]
    assert max(spans) > 550
    st = lineart.ink_stats(data)
    assert st["strokes"] == len(data["strokes"]) and st["points"] > 10


def test_alignment_separates_registered_from_shifted_art(page):
    stroke, fill = lineart.ink_mask(page)
    ink = stroke | fill
    rng = np.random.default_rng(1)
    colour = np.dstack([page * 0 + 200, page * 0 + 140, page * 0 + 90]).astype(np.uint8)
    colour[(page > 200)] = (225, 190, 150)
    colour = cv2.GaussianBlur(colour, (0, 0), 1.0)
    colour[ink] = (20, 20, 40)                                                         # coloured art that keeps the lines
    good = lineart.alignment(ink, colour)
    shifted = np.roll(np.roll(colour, 14, axis=0), 11, axis=1)
    bad = lineart.alignment(ink, shifted)
    unrelated = rng.integers(0, 255, colour.shape, dtype=np.uint8)
    assert good > 0.55 and bad < 0.25 and good > bad + 0.35
    assert lineart.alignment(ink, cv2.GaussianBlur(unrelated, (0, 0), 3)) < 0.3


def test_prepare_panel_fits_a_16x9_canvas(page, tmp_path):
    p = tmp_path / "page.png"
    Image.fromarray(page).save(p)
    box = [30, 30, 640, 960]                                                          # ymin, xmin, ymax, xmax on 0..1000
    init, ink, place = lineart.prepare_panel(p, box, size=(1344, 768))
    assert init.size == (1344, 768) and ink.shape == (768, 1344) and ink.dtype == bool
    x0, y0, x1, y1 = place
    assert (x1 - x0) <= 1344 and (y1 - y0) == 768 or (x1 - x0) == 1344                  # contain-fit
    assert ink[:, :x0].sum() == 0 and ink[:, x1:].sum() == 0                           # nothing drawn in the margins
    assert ink.sum() > 2000
    g = np.asarray(init.convert("L"))
    assert g[:, :max(1, x0 - 2)].min() >= 240 if x0 > 4 else True                       # paper-white margins


def test_overlay_darkens_only_along_the_ink(page):
    stroke, fill = lineart.ink_mask(page)
    art = Image.fromarray(np.full((*page.shape, 3), (210, 170, 120), np.uint8))
    out = np.asarray(lineart.overlay_ink(art, stroke))
    assert out[stroke].mean() < 0.5 * 170 and abs(out[~stroke].mean() - np.mean((210, 170, 120))) < 6
    tinted = out[stroke].reshape(-1, 3).mean(0)
    assert tinted[0] > tinted[2]                                                       # ink keeps the warm hue of the art

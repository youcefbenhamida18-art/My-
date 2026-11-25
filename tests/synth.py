"""Synthetic 'key art' used by the offline tests: sky, mountains, foliage, a figure with hair and cape."""
import cv2
import numpy as np


def key_art(w=1344, h=768, seed=1, figure=True):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        t = y / h
        img[y, :, :] = (int(190 - 90 * t), int(150 - 40 * t), int(90 + 50 * t))          # BGR sky: warm at the horizon
    cv2.circle(img, (int(w * 0.72), int(h * 0.28)), int(h * 0.07), (200, 235, 255), -1)
    pts = np.array([[0, int(h * .62)], [int(w * .2), int(h * .48)], [int(w * .45), int(h * .6)], [int(w * .7), int(h * .5)],
                    [w, int(h * .62)], [w, h], [0, h]])
    cv2.fillPoly(img, [pts], (70, 80, 60))
    for _ in range(260):                                                                  # green foliage speckle, top-left
        x, y = int(rng.uniform(0, w * .35)), int(rng.uniform(h * .08, h * .5))
        cv2.circle(img, (x, y), int(rng.uniform(6, 16)), (int(rng.uniform(30, 70)), int(rng.uniform(120, 190)), int(rng.uniform(40, 90))), -1)
    cv2.rectangle(img, (0, int(h * .82)), (w, h), (40, 70, 45), -1)
    if figure:
        cx = int(w * .5)
        cv2.ellipse(img, (cx, int(h * .72)), (int(w * .07), int(h * .2)), 0, 0, 360, (60, 40, 130), -1)         # body (red-ish BGR)
        cv2.ellipse(img, (cx - int(w * .06), int(h * .8)), (int(w * .05), int(h * .12)), 20, 0, 360, (40, 30, 90), -1)  # cape
        cv2.circle(img, (cx, int(h * .46)), int(h * .06), (190, 210, 240), -1)                               # face
        cv2.ellipse(img, (cx, int(h * .43)), (int(h * .075), int(h * .06)), 0, 180, 360, (30, 30, 30), -1)    # hair
    return img

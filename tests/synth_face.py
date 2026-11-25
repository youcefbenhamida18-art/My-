"""A flat-colour anime face with two big eyes - enough structure to test eye analysis / layers without any model."""
import cv2
import numpy as np


def anime_face(w=1600, h=900, iris=(215, 140, 40), skin=(246, 206, 180), highlight=True, eye_dx=190, eye_w=170, eye_h=108):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70, 90, 140)                                              # background
    cx, cy = w // 2, int(h * 0.52)
    cv2.ellipse(img, (cx, cy), (330, 400), 0, 0, 360, (30, 25, 40), -1, cv2.LINE_AA)           # hair mass
    cv2.ellipse(img, (cx, cy + 40), (290, 350), 0, 0, 360, skin, -1, cv2.LINE_AA)               # face
    cv2.ellipse(img, (cx, cy - 190), (300, 130), 0, 180, 360, (30, 25, 40), -1, cv2.LINE_AA)    # fringe
    eyes = []
    for sgn in (-1, 1):
        ex, ey = cx + sgn * eye_dx, cy + 10
        cv2.ellipse(img, (ex, ey), (eye_w // 2, eye_h // 2), 0, 0, 360, (250, 250, 252), -1, cv2.LINE_AA)       # sclera
        ir = (int(eye_w * 0.30), int(eye_h * 0.50))
        cv2.ellipse(img, (ex, ey), ir, 0, 0, 360, iris, -1, cv2.LINE_AA)                                          # iris
        cv2.ellipse(img, (ex, ey + 8), (int(ir[0] * 0.5), int(ir[1] * 0.55)), 0, 0, 360, (25, 15, 20), -1, cv2.LINE_AA)  # pupil
        cv2.ellipse(img, (ex, ey), ir, 0, 0, 360, (60, 40, 20), 3, cv2.LINE_AA)
        if highlight:
            cv2.ellipse(img, (ex - 14, ey - 16), (14, 15), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.ellipse(img, (ex, ey), (eye_w // 2 + 4, eye_h // 2 + 4), 0, 190, 350, (25, 15, 20), 9, cv2.LINE_AA)  # upper lash line
        cv2.ellipse(img, (ex, ey), (eye_w // 2, eye_h // 2), 0, 20, 160, (140, 90, 80), 2, cv2.LINE_AA)           # lower lid
        eyes.append((ex - eye_w // 2 - 6, ey - eye_h // 2 - 8, ex + eye_w // 2 + 6, ey + eye_h // 2 + 6))
    return img, eyes, (cx - 300, cy - 300, cx + 300, cy + 380)

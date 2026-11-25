"""Gemini transport (text + vision) with model discovery, key rotation and tolerant JSON parsing.

`ask_json(prompt, images=[bytes|Path,...])` is the only function the rest of the project needs. Everything else here is
plumbing that v1 already proved in production (self-healing model discovery: model names get retired, so none is hard-coded).
"""
from __future__ import annotations

import base64
import functools
import json
import pathlib
import re
import time

import requests

from .config import env, gemini_keys, log

URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]          # used only if discovery fails
NOT_TEXT = ("tts", "image", "live", "audio", "embedding", "native", "omni", "cyber", "robotics",
            "computer", "translate", "transcribe", "imagen", "veo")


class LLMError(RuntimeError):
    pass


def _rank(name):
    v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return ("preview" in name or "exp" in name, -(float(v.group(1)) if v else 0), "lite" in name, name)


@functools.lru_cache(1)
def discover_models():
    """Ask the API which Flash models THIS key can use today, newest stable first. GEMINI_MODELS forces a list."""
    forced = [m.strip() for m in env("GEMINI_MODELS", "").split(",") if m.strip()]
    found = []
    for key in gemini_keys():
        try:
            r = requests.get(LIST_URL, headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
            r.raise_for_status()
            for m in r.json().get("models", []):
                n = m["name"].split("/")[-1]
                if ("generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini-")
                        and "flash" in n and not any(b in n for b in NOT_TEXT)):
                    found.append(n)
            if found:
                break
        except Exception as e:  # noqa: BLE001
            log.warning("ListModels failed: %s", e)
    ordered = forced + [n for n in sorted(set(found), key=_rank) if n not in forced][:5]
    ordered = ordered or FALLBACK_MODELS
    log.info("Gemini models to try: %s", ordered)
    return ordered


# ------------------------------------------------------------------ tolerant JSON
def _balance(txt):
    """Close unterminated strings/brackets (model output cut by max tokens). Returns repaired text."""
    stack, in_str, esc = [], False, False
    for ch in txt:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    out = txt + ('"' if in_str else "")
    out = re.sub(r",\s*$", "", out.rstrip())
    return out + "".join(reversed(stack))


def parse_json(txt):
    """Parse model output: strips code fences, trailing commas, surrounding prose, and repairs truncation."""
    if not isinstance(txt, str):
        raise LLMError("empty model output")
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", txt.strip(), flags=re.M).strip()
    attempts = [t]
    i = min([p for p in (t.find("{"), t.find("[")) if p >= 0] or [0])
    if i:
        attempts.append(t[i:])
    attempts.append(re.sub(r",\s*([}\]])", r"\1", t[i:]))
    attempts.append(_balance(re.sub(r",\s*([}\]])", r"\1", t[i:])))
    last = None
    for a in attempts:
        try:
            return json.loads(a)
        except (json.JSONDecodeError, ValueError) as e:
            last = e
    raise LLMError(f"model returned invalid JSON ({last}); head: {t[:120]!r}")


# ------------------------------------------------------------------ request
def _mime(b):
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _parts(prompt, images):
    parts = []
    for im in images or []:
        data = pathlib.Path(im).read_bytes() if isinstance(im, (str, pathlib.Path)) else bytes(im)
        parts.append({"inline_data": {"mime_type": _mime(data), "data": base64.b64encode(data).decode()}})
    parts.append({"text": prompt})
    return parts


def ask_json(prompt, *, images=None, temperature=1.0, max_tokens=30000):
    """Walk the discovered models and API keys until one answers with valid JSON. Raises LLMError if none does."""
    keys = gemini_keys()
    if not keys:
        raise LLMError("no GEMINI_API_KEY set (get a free key at https://aistudio.google.com/apikey)")
    body = {"contents": [{"parts": _parts(prompt, images)}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json",
                                 "maxOutputTokens": max_tokens}}
    last = None
    for model in discover_models():
        for key in keys:
            try:
                r = requests.post(URL.format(model), headers={"x-goog-api-key": key}, timeout=300, json=body)
                if r.status_code == 404:                       # model retired/unavailable: next model
                    last = f"{model}: 404 {r.text[:160]}"
                    log.warning("model %s not available (404) - trying the next one", model)
                    break
                if r.status_code in (429, 500, 503):
                    last = f"{model}: {r.status_code}"
                    time.sleep(4)
                    continue
                if r.status_code >= 400:
                    last = f"{model}: {r.status_code} {r.text[:200]}"
                    log.warning("Gemini %s -> %s", model, last)
                    continue
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return parse_json(txt)
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("Gemini %s failed: %s", model, e)
    raise LLMError(f"all Gemini keys/models failed: {last}")

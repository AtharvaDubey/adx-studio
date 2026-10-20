"""Colour look (grade) for KARAM projects. Shared by Karam Studio previews and the project's renderer.

The studio copies this file into <project>/src/look.py and hooks it into Renderer.post(), so the
saved look (analysis/look.json) is applied identically in previews, studio renders and render_all.py.
"""
import json, os
import numpy as np
import cv2

DEFAULT = dict(exposure=0.0, contrast=0.0, saturation=0.0, warmth=0.0, tint=0.0,
               blacks=0.0, highlights=0.0, fade=0.0, vignette=0.0)
_vig_cache = {}


def clean(p):
    out = dict(DEFAULT)
    for k in DEFAULT:
        try:
            v = float((p or {}).get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        out[k] = max(-1.0, min(1.0, v)) if k != 'exposure' else max(-2.0, min(2.0, v))
    return out


def load(root):
    try:
        with open(os.path.join(root, 'analysis', 'look.json'), encoding='utf-8') as f:
            p = clean(json.load(f))
        return None if is_identity(p) else p
    except Exception:
        return None


def is_identity(p):
    return p is None or all(abs(p.get(k, 0.0)) < 1e-4 for k in DEFAULT)


def _luts(p):
    x = np.arange(256, dtype=np.float32) / 255.0
    y = x * (2.0 ** p['exposure'])
    y = 0.5 + (y - 0.5) * (1.0 + 0.8 * p['contrast'])
    y = y + p['blacks'] * 0.12 * (1.0 - y) ** 3                       # lift / crush shadows
    y = y + p['highlights'] * 0.30 * np.clip((y - 0.5) / 0.5, 0, 1) ** 2  # open / roll off highlights
    y = y * (1.0 - 0.18 * p['fade']) + 0.07 * p['fade'] if p['fade'] > 0 else y
    w, t = p['warmth'], p['tint']
    gains = (1.0 - 0.14 * w, 1.0 - 0.08 * t, 1.0 + 0.14 * w)          # B, G, R
    lut = np.stack([np.clip(y * g, 0, 1) for g in gains], axis=1)
    return (lut * 255 + 0.5).astype(np.uint8).reshape(1, 256, 3)


def _vignette(h, w):
    key = (h, w)
    if key not in _vig_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 * 0.85 + ((yy - h / 2) / (h / 2)) ** 2)
        _vig_cache[key] = np.clip((d - 0.45) / 0.9, 0, 1) ** 1.8
    return _vig_cache[key]


def apply(arr, p):
    """arr: BGRA uint8 (H, W, 4). Returns a graded BGRA uint8 array."""
    if is_identity(p):
        return arr
    p = clean(p)
    rgb = cv2.LUT(np.ascontiguousarray(arr[..., :3]), _luts(p))
    s = p['saturation']
    if abs(s) > 1e-4:
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        g3 = cv2.merge([g, g, g])
        rgb = cv2.addWeighted(rgb, 1.0 + s, g3, -s, 0)
    v = p['vignette']
    if abs(v) > 1e-4:
        m = _vignette(rgb.shape[0], rgb.shape[1])
        k = 1.0 - 0.75 * v * m if v > 0 else 1.0 - 0.6 * v * m
        rgb = np.clip(rgb.astype(np.float32) * k[..., None], 0, 255).astype(np.uint8)
    out = np.empty_like(arr)
    out[..., :3] = rgb
    out[..., 3] = 255
    return out

"""Analyse a song for an ADx Studio template project and (re)build analysis/timeline.json.

    python3 studio_analyze.py <project_dir>

Measures loudness and bass curves (100 per second), tempo and beat phase, and kick-like hits.
Lyric lines come from project.json ("lyrics"). Existing word timings are kept for lines whose text
is unchanged; new lines are spread over the song and can then be synced in the app.
"""
import json, os, subprocess, sys
import numpy as np

proj = os.path.abspath(sys.argv[1])
SR, HOP, NFFT = 24000, 240, 1024
audio = os.path.join(proj, 'assets', 'karam.mp3')
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', audio, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                     capture_output=True, check=True).stdout
y = np.frombuffer(raw, dtype=np.float32).copy()
dur = len(y) / SR
n = 1 + max(0, (len(y) - NFFT) // HOP)
idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
frames = y[np.minimum(idx, len(y) - 1)] * np.hanning(NFFT)[None, :].astype(np.float32)
mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
freqs = np.fft.rfftfreq(NFFT, 1 / SR)
rms = np.sqrt((frames ** 2).mean(1))
low = mag[:, freqs < 150].sum(1)


def norm(v):
    v = np.convolve(v, np.ones(5) / 5, mode='same')
    return np.clip(v / (np.percentile(v, 98) + 1e-9), 0, 1)


rms_n, low_n = norm(rms), norm(low)
lm = np.log1p(mag * 10)
flux = np.maximum(0, np.diff(lm, axis=0, prepend=lm[:1])).sum(1)
flux = flux - np.convolve(flux, np.ones(50) / 50, mode='same'); flux = np.maximum(flux, 0)
fps = SR / HOP
# tempo: autocorrelation of the onset curve over 70-180 BPM, mildly preferring 90-150
f = flux - flux.mean()
ac = np.correlate(f, f, mode='full')[len(f) - 1:]
lags = np.arange(int(fps * 60 / 180), int(fps * 60 / 70) + 1)
w = np.exp(-0.5 * (np.log2((60 * fps / lags) / 120)) ** 2 / 0.6 ** 2)
lag = lags[int(np.argmax(ac[lags] * w))]
beat = lag / fps
phases = np.arange(0, lag)
score = [flux[p::lag].sum() for p in phases]
t0 = float(phases[int(np.argmax(score))] / fps)
lf = np.maximum(0, np.diff(np.log1p(low), prepend=np.log1p(low[:1])))
thr = np.percentile(lf, 96)
kicks, last = [], -1
for i in range(1, len(lf) - 1):
    if lf[i] > thr and lf[i] >= lf[i - 1] and lf[i] >= lf[i + 1] and (i - last) > fps * 0.12:
        kicks.append([round(i / fps, 3), round(float(lf[i]), 3)]); last = i

pj = json.load(open(os.path.join(proj, 'project.json'), encoding='utf-8'))
lines = [l.strip() for l in (pj.get('lyrics') or '').splitlines() if l.strip()] or ['(add your lyrics)']
tlp = os.path.join(proj, 'analysis', 'timeline.json')
old = {}
if os.path.isfile(tlp):
    try:
        T0 = json.load(open(tlp, encoding='utf-8'))
        by = {}
        for wd in T0.get('words', []):
            by.setdefault(wd['line'], []).append(wd)
        for li, txt in enumerate(T0.get('lines', [])):
            ws = sorted(by.get(li, []), key=lambda q: q['wi'])
            if ws and ' '.join(q['text'] for q in ws) == txt and ws[-1]['end'] <= dur:
                old.setdefault(txt, []).append(ws)
    except Exception:
        old = {}

# default spread: from where the music gets going to just before the end
act = np.where(rms_n > 0.3)[0]
a = max(0.5, (act[0] / fps if len(act) else 0) + 1.0)
b = max(a + 1, dur - max(3.0, dur * 0.06))
weights = [len(l.split()) + 1.5 for l in lines]
tot = sum(weights)
words, cur = [], a
for li, txt in enumerate(lines):
    slot = (b - a) * weights[li] / tot
    if old.get(txt):
        ws = old[txt].pop(0)
        for wi, q in enumerate(ws):
            words.append(dict(line=li, wi=wi, text=q['text'], start=q['start'], end=q['end']))
    else:
        toks = txt.split(); span = slot * 0.85
        cw = [len(tk) + 1.5 for tk in toks]; ct = sum(cw); s = cur
        for wi, tk in enumerate(toks):
            ln = span * cw[wi] / ct
            words.append(dict(line=li, wi=wi, text=tk, start=round(s, 3), end=round(s + max(0.06, ln - 0.03), 3)))
            s += ln
    cur += slot
T = dict(dur=dur, bpm=round(60 / beat, 2), t0=round(t0, 3), beat=beat,
         beats=[round(t0 + k * beat, 3) for k in range(int((dur - t0) / beat) + 1)],
         sections=[dict(name='song', start=0.0, end=dur)], events=dict(kick=kicks, snare=[], hat=[]),
         curves_fps=fps, curves=dict(vocal=[round(float(v), 4) for v in rms_n], low=[round(float(v), 4) for v in low_n]),
         lines=lines, words=words)
os.makedirs(os.path.dirname(tlp), exist_ok=True)
tmp = tlp + '.tmp'
json.dump(T, open(tmp, 'w', encoding='utf-8'))
os.replace(tmp, tlp)
print(json.dumps(dict(ok=True, dur=round(dur, 2), bpm=T['bpm'], lines=len(lines), words=len(words), kicks=len(kicks))))

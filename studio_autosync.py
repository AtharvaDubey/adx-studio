"""ADx Studio auto sync: word-level lyric timing from the song itself, fully on this computer.

    python3 studio_autosync.py <project_dir> <progress.json>

1. Separates the vocal from the beat (Spleeter 2-stem, ONNX via sherpa-onnx).
2. Runs a multilingual speech model (Meta Omnilingual ASR 300M CTC, int8 ONNX) over the vocal.
3. Forced-aligns the exact lyric words through a lattice in which every word may match its Latin
   spelling or Devanagari pronunciations (Hindi is heard in Devanagari), then snaps starts to vocal onsets.
Models (~330 MB) are downloaded once from the sherpa-onnx GitHub releases.
"""
import json, os, re, subprocess, sys, tarfile, time, unicodedata, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
proj, progf = os.path.abspath(sys.argv[1]), sys.argv[2]
MODELS = os.environ.get('ADX_MODELS') or os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'ADx Studio', 'models') \
    if sys.platform == 'darwin' else os.environ.get('ADX_MODELS') or os.path.join(HERE, 'models')
REL = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
SPL = ('sherpa-onnx-spleeter-2stems-fp16', REL + 'source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2', 35)
ASR = ('sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05',
       REL + 'asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05.tar.bz2', 292)


def prog(stage, frac, msg):
    with open(progf + '.tmp', 'w') as f:
        json.dump(dict(stage=stage, frac=round(float(frac), 3), msg=msg), f)
    os.replace(progf + '.tmp', progf)


def fetch(name, url, mb, base, span):
    d = os.path.join(MODELS, name)
    if os.path.isdir(d) and os.listdir(d):
        return d
    os.makedirs(MODELS, exist_ok=True)
    tmp = os.path.join(MODELS, name + '.tar.bz2.part')
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or mb * 1e6); got = 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); got += len(b)
            prog('download', base + span * got / total, 'Downloading speech models (one time): %d / %d MB' % (got / 1e6, total / 1e6))
    prog('download', base + span, 'Unpacking %s…' % name.split('-')[2])
    with tarfile.open(tmp, 'r:bz2') as t:
        for m in t.getmembers():
            if m.name.startswith('/') or '..' in m.name:
                raise RuntimeError('unsafe archive')
        try:
            t.extractall(MODELS, filter="data")
        except TypeError:
            t.extractall(MODELS)
    os.remove(tmp)
    return d


def decode(path, sr, ch):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', str(ch), '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).T.copy()


# ---------------------------------------------------------------- Hinglish -> Devanagari candidates
CONS = [('chh', 'छ'), ('kh', 'ख'), ('gh', 'घ'), ('ch', 'च'), ('jh', 'झ'), ('th', 'थ'), ('dh', 'ध'), ('ph', 'फ'), ('bh', 'भ'),
        ('sh', 'श'), ('ng', 'ं'), ('k', 'क'), ('g', 'ग'), ('c', 'क'), ('j', 'ज'), ('t', 'त'), ('d', 'द'), ('n', 'न'), ('p', 'प'),
        ('f', 'फ'), ('b', 'ब'), ('m', 'म'), ('y', 'य'), ('r', 'र'), ('l', 'ल'), ('v', 'व'), ('w', 'व'), ('s', 'स'), ('h', 'ह'),
        ('z', 'ज़'), ('q', 'क'), ('x', 'क्स')]
VOW = [('aa', 'आ', 'ा'), ('ee', 'ई', 'ी'), ('ii', 'ई', 'ी'), ('oo', 'ऊ', 'ू'), ('uu', 'ऊ', 'ू'), ('ai', 'ऐ', 'ै'), ('au', 'औ', 'ौ'),
       ('a', 'अ', ''), ('e', 'ए', 'े'), ('i', 'इ', 'ि'), ('o', 'ओ', 'ो'), ('u', 'उ', 'ु')]
HARD = {'t': 'ट', 'd': 'ड', 'th': 'ठ', 'dh': 'ढ'}


def translit(word, hard=False, long_end=True):
    w, out, i, prev_c = word, '', 0, False
    while i < len(w):
        for v, ind, mat in VOW:
            if w.startswith(v, i):
                if i + len(v) == len(w) and long_end and v in ('i', 'u'):
                    ind, mat = {'i': ('ई', 'ी'), 'u': ('ऊ', 'ू')}[v]
                out += mat if prev_c else ind
                i += len(v); prev_c = False; break
        else:
            for c, dv in CONS:
                if w.startswith(c, i):
                    if hard and c in HARD: dv = HARD[c]
                    if prev_c and dv != 'ं': out += '्'
                    out += dv; i += len(c); prev_c = dv != 'ं'; break
            else:
                i += 1; prev_c = False
    return unicodedata.normalize('NFC', out)


def candidates(nw):
    try:
        from deva_known import DEVA
    except Exception:
        DEVA = {}
    c = list(DEVA.get(nw, []))
    for h in (False, True):
        for le in (True, False):
            s = translit(nw, h, le)
            if s and s not in c: c.append(s)
    return c[:6]


def norm_word(w):
    return re.sub(r"[^a-z']", '', w.lower().replace('’', "'"))


# ---------------------------------------------------------------- main
def main():
    T = json.load(open(os.path.join(proj, 'analysis', 'timeline.json'), encoding='utf-8'))
    lines = T['lines']
    prog('start', 0.0, 'Preparing…')
    spl = fetch(*SPL, base=0.0, span=0.04)
    asr = fetch(*ASR, base=0.04, span=0.30)
    import sherpa_onnx, onnxruntime as ort
    prog('separate', 0.36, 'Separating the vocal from the beat…')
    x = decode(os.path.join(proj, 'assets', 'karam.mp3'), 44100, 2)
    dur = x.shape[1] / 44100
    cfg = sherpa_onnx.OfflineSourceSeparationConfig(model=sherpa_onnx.OfflineSourceSeparationModelConfig(
        spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(vocals=os.path.join(spl, 'vocals.fp16.onnx'),
                                                                       accompaniment=os.path.join(spl, 'accompaniment.fp16.onnx')),
        num_threads=max(1, min(6, (os.cpu_count() or 2) - 1)), debug=False, provider='cpu'))
    sep = sherpa_onnx.OfflineSourceSeparation(cfg)
    out = sep.process(sample_rate=44100, samples=np.ascontiguousarray(x))
    voc = np.array(out.stems[0].data).mean(0).astype(np.float32); vsr = out.sample_rate
    # resample vocal to 16 kHz (polyphase-free linear resample is fine for CTC posteriors)
    n16 = int(len(voc) * 16000 / vsr)
    v16 = np.interp(np.linspace(0, len(voc) - 1, n16), np.arange(len(voc)), voc).astype(np.float32)
    prog('listen', 0.55, 'Listening to the vocal…')
    tok = {}
    for line in open(os.path.join(asr, 'tokens.txt'), encoding='utf-8'):
        line = line.rstrip('\n')
        if line:
            s, i = line.rsplit(' ', 1); tok[s if s else ' '] = int(i)
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 2)))
    sess = ort.InferenceSession(os.path.join(asr, 'model.int8.onnx'), so, providers=['CPUExecutionProvider'])
    SR, win, hop, fps = 16000, 30 * 16000, 24 * 16000, 50.0
    total = int(np.ceil(len(v16) / SR * fps)) + 2
    acc = None; wsum = np.zeros(total); start = 0
    while True:
        seg = v16[start:start + win]
        xx = (seg - seg.mean()) / np.sqrt(seg.var() + 1e-5)
        lg = sess.run(None, {'x': xx[None, :].astype(np.float32)})[0][0]
        lp = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
        f0 = int(round(start / SR * fps)); Tn = lp.shape[0]
        if acc is None: acc = np.zeros((total, lp.shape[1]), np.float32)
        w = np.ones(Tn); ramp = min(100, Tn // 4)
        if start > 0: w[:ramp] = np.linspace(0.01, 1, ramp)
        if start + win < len(v16): w[-ramp:] = np.linspace(1, 0.01, ramp)
        e = min(total, f0 + Tn)
        acc[f0:e] += (lp[:e - f0] * w[:e - f0, None]).astype(np.float32); wsum[f0:e] += w[:e - f0]
        prog('listen', 0.55 + 0.30 * min(1, (start + win) / len(v16)), 'Listening to the vocal…')
        if start + win >= len(v16): break
        start += hop
    valid = wsum > 0; acc[valid] /= wsum[valid, None]; lp = acc[valid]
    prog('align', 0.86, 'Matching every lyric word to the vocal…')
    BLANK, SPACE = 0, tok[' ']
    words = []
    for li, l in enumerate(lines):
        for wi, wtxt in enumerate(l.split()):
            nw = norm_word(wtxt)
            alts = [nw.replace("'", "")] if nw else []
            for dv in (candidates(nw) if nw else []):
                alts += [unicodedata.normalize('NFC', dv), unicodedata.normalize('NFD', dv)]
            seqs = []
            for a in alts:
                if a and all(ch in tok for ch in a):
                    s = [tok[ch] for ch in a]
                    if s not in seqs: seqs.append(s)
            words.append(dict(line=li, wi=wi, text=wtxt, seqs=seqs or [[SPACE]]))
    st_tok, preds, st_word, st_kind = [], [], [], []

    def add(tk, pr, word=-1, kind='b'):
        st_tok.append(tk); preds.append(list(pr)); st_word.append(word); st_kind.append(kind); return len(st_tok) - 1
    g0 = add(BLANK, []); preds[g0].append(g0)
    prev_ends, prev_gap, starts = [], [g0], [g0]
    for k, w in enumerate(words):
        ends = []
        for seq in w['seqs']:
            last = lasttok = None
            for i, c in enumerate(seq):
                if i == 0:
                    pr = list(prev_gap) + [e for e in prev_ends if st_tok[e] != c]
                    s = add(c, pr, k, 'c')
                    if k == 0: starts.append(s)
                else:
                    b = add(BLANK, [last], k, 'ib'); preds[b].append(b)
                    s = add(c, [b] + ([last] if lasttok != c else []), k, 'c')
                preds[s].append(s); last, lasttok = s, c
            ends.append(last)
        b1 = add(BLANK, ends, k, 'g'); preds[b1].append(b1)
        sp = add(SPACE, ends + [b1], k, 'sp'); preds[sp].append(sp)
        b2 = add(BLANK, [sp], k, 'g'); preds[b2].append(b2)
        prev_gap, prev_ends = [b1, sp, b2], ends
    finals = prev_ends + prev_gap
    S = len(st_tok); K = max(len(p) for p in preds)
    P = np.full((S, K), -1, np.int64)
    for s, p in enumerate(preds): P[s, :len(p)] = p
    st_tok = np.array(st_tok); Tn = lp.shape[0]
    dp = np.full(S, -1e30); dp[starts] = lp[0, st_tok[starts]]
    bp = np.zeros((Tn, S), np.int8); validP = P >= 0; Pc = np.where(validP, P, 0)
    for t in range(1, Tn):
        cand = np.where(validP, dp[Pc], -1e30); arg = cand.argmax(1)
        dp = cand[np.arange(S), arg] + lp[t, st_tok]; bp[t] = arg
        if t % 2000 == 0: prog('align', 0.86 + 0.1 * t / Tn, 'Matching every lyric word to the vocal…')
    s = max(finals, key=lambda q: dp[q]); path = np.zeros(Tn, np.int64)
    for t in range(Tn - 1, -1, -1):
        path[t] = s; s = P[s, bp[t, s]]
    st_word = np.array(st_word); st_kind = np.array(st_kind)
    # vocal onsets (energy flux on the separated vocal) for snapping starts
    hopv = int(vsr / 100); fr = np.lib.stride_tricks.sliding_window_view(voc[:len(voc) // hopv * hopv], 1024)[::hopv] if len(voc) > 1024 else None
    onsets = np.array([])
    env = None
    if fr is not None:
        env = np.sqrt((fr ** 2).mean(1)); le = np.log1p(env * 50); fl = np.maximum(0, np.diff(le, prepend=le[0]))
        thr = np.percentile(fl, 90)
        onsets = np.array([i / 100.0 for i in range(1, len(fl) - 1) if fl[i] > thr and fl[i] >= fl[i - 1] and fl[i] >= fl[i + 1]])
    res = []
    for k, w in enumerate(words):
        m = np.where((st_word[path] == k) & ((st_kind[path] == 'c') | (st_kind[path] == 'ib')))[0]
        if len(m):
            s0, e0 = m[0] / fps, (m[-1] + 1) / fps
        else:
            s0 = res[-1]['end'] if res else 0.0; e0 = s0 + 0.2
        if len(onsets):
            near = onsets[np.abs(onsets - s0) < 0.06]
            if len(near): s0 = float(near[np.argmin(np.abs(near - s0))])
        res.append(dict(line=w['line'], wi=w['wi'], text=w['text'], start=round(s0, 3), end=round(max(e0, s0 + 0.08), 3)))
    for i in range(1, len(res)):
        if res[i]['start'] < res[i - 1]['start'] + 0.04: res[i]['start'] = round(res[i - 1]['start'] + 0.04, 3)
        if res[i]['end'] < res[i]['start'] + 0.08: res[i]['end'] = round(res[i]['start'] + 0.08, 3)
    T['words'] = res
    if env is not None:
        ve = np.convolve(env, np.ones(5) / 5, mode='same'); ve = np.clip(ve / (np.percentile(ve, 98) + 1e-9), 0, 1)
        T['curves']['vocal'] = [round(float(q), 4) for q in ve]; T['curves_fps'] = 100.0
        if 'low' in T['curves']:
            n = len(T['curves']['vocal']); low = T['curves']['low']
            T['curves']['low'] = (low + [0.0] * n)[:n]
    tlp = os.path.join(proj, 'analysis', 'timeline.json')
    json.dump(T, open(tlp + '.tmp', 'w', encoding='utf-8')); os.replace(tlp + '.tmp', tlp)
    prog('done', 1.0, 'Synced %d words' % len(res))
    print(json.dumps(dict(ok=True, words=len(res))))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        prog('error', 0, '%s: %s' % (type(e).__name__, e))
        raise

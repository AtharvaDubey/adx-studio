#!/usr/bin/env python3
"""ADx Studio: create, edit and render lyric videos locally.

    python3 server.py            # then open http://127.0.0.1:8765

Import a project zip, edit lyrics and word timing against the vocal envelope (and the song, if you
load it), preview any frame or a short clip using the project's own renderer, export a new zip.

The server uses only the Python standard library. Frame and clip previews run the project's own
`src/main.py`, so that project's dependencies (skia-python, uharfbuzz, opencv, numpy, scipy, pillow)
and ffmpeg must be installed.
"""
import argparse, hashlib, io, json, math, os, re, shutil, subprocess, sys, threading, time, traceback, uuid, webbrowser, zipfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ADX_HOME = os.environ.get('ADX_HOME') or HERE
PROJ = os.path.join(ADX_HOME, 'Projects' if os.environ.get('ADX_HOME') else 'projects')
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
MAX_UPLOAD = 800 * 1024 * 1024
STILL_LOCK = threading.Lock()
JOBS = {}


class ApiError(Exception):
    def __init__(self, code, msg):
        Exception.__init__(self, msg)
        self.code, self.msg = code, msg


# ---------------------------------------------------------------- project helpers
def pdir(name):
    if not NAME_RE.match(name or ''):
        raise ApiError(400, 'bad project name')
    d = os.path.join(PROJ, name)
    if not os.path.isdir(d):
        raise ApiError(404, 'no such project')
    return d


def tl_path(d): return os.path.join(d, 'analysis', 'timeline.json')
def audio_path(d): return os.path.join(d, 'assets', 'karam.mp3')


def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def write_atomic(p, data, binary=False):
    tmp = p + '.tmp%d' % os.getpid()
    with open(tmp, 'wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8'})) as f:
        f.write(data)
    os.replace(tmp, p)


def list_projects():
    out = []
    for n in sorted(os.listdir(PROJ)):
        d = os.path.join(PROJ, n)
        if os.path.isfile(tl_path(d)):
            pj = read_pj(d)
            out.append(dict(name=n, audio=os.path.isfile(audio_path(d)), kind='template' if pj is not None else 'custom',
                            template=(pj or {}).get('template', ''), title=(pj or {}).get('title') or n,
                            artist=(pj or {}).get('artist', ''), aspect=(pj or {}).get('aspect', '21:9' if pj is None else '16:9'),
                            mtime=os.path.getmtime(tl_path(d))))
    out.sort(key=lambda p: -p['mtime'])
    return out


def read_pj(d):
    p = os.path.join(d, 'project.json')
    if not os.path.isfile(p):
        return None
    try:
        return load_json(p)
    except Exception:
        return {}


def write_pj(d, pj):
    write_atomic(os.path.join(d, 'project.json'), json.dumps(pj, indent=1, ensure_ascii=False))


SETTING_KEYS = ('template', 'aspect', 'title', 'artist', 'bg1', 'bg2', 'text', 'accent', 'font', 'size', 'position', 'logo_intro',
                'logo_outro', 'watermark', 'title_card', 'visualizer', 'particles', 'pulse', 'bg_image', 'bg_dim', 'logo', 'case')


def slug(s):
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s or '').strip('._')[:48] or 'video'
    name, k = s, 2
    while os.path.exists(os.path.join(PROJ, name)):
        name = '%s-%d' % (s[:44], k); k += 1
    return name


def new_project(body):
    title = (body.get('title') or '').strip() or 'Untitled'
    name = slug(title)
    d = os.path.join(PROJ, name)
    for sub in ('src', 'assets/fonts', 'analysis', 'out'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for f in os.listdir(os.path.join(HERE, 'engine')):
        if f.endswith('.py'):
            shutil.copyfile(os.path.join(HERE, 'engine', f), os.path.join(d, 'src', f))
    shutil.copyfile(os.path.join(HERE, 'look.py'), os.path.join(d, 'src', 'look.py'))
    for f in os.listdir(os.path.join(HERE, 'fonts')):
        shutil.copyfile(os.path.join(HERE, 'fonts', f), os.path.join(d, 'assets', 'fonts', f))
    shutil.copyfile(os.path.join(HERE, 'static', 'logo.png'), os.path.join(d, 'assets', 'logo.png'))
    pj = dict(template=body.get('template') or 'ink', aspect=body.get('aspect') or '16:9', title=title,
              artist=(body.get('artist') or '').strip(), lyrics=body.get('lyrics') or '')
    write_pj(d, pj)
    lines = [l.strip() for l in pj['lyrics'].splitlines() if l.strip()] or ['(add your lyrics)']
    words, t = [], 2.0
    for li, l in enumerate(lines):
        for wi, w in enumerate(l.split()):
            words.append(dict(line=li, wi=wi, text=w, start=t, end=t + 0.3)); t += 0.35
    T = dict(dur=max(t + 2, 10.0), bpm=120, t0=0.0, beat=0.5, beats=[], sections=[dict(name='song', start=0, end=max(t + 2, 10.0))],
             events=dict(kick=[], snare=[], hat=[]), curves_fps=100, curves=dict(vocal=[0.0] * 10, low=[0.0] * 10), lines=lines, words=words)
    write_atomic(tl_path(d), json.dumps(T))
    shutil.copyfile(tl_path(d), os.path.join(d, 'analysis', 'timeline_original.json'))
    return name


def analyze(name):
    d = pdir(name)
    if read_pj(d) is None:
        return dict(ok=True, skipped=True)
    if not os.path.isfile(audio_path(d)):
        raise ApiError(400, 'load the song first')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'studio_analyze.py'), d], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise ApiError(500, 'could not analyse the song: ' + ' | '.join(r.stderr.strip().splitlines()[-3:]))
    shutil.copyfile(tl_path(d), os.path.join(d, 'analysis', 'timeline_original.json'))
    return json.loads(r.stdout.strip().splitlines()[-1])


def get_settings(name):
    d = pdir(name); pj = read_pj(d)
    return dict(kind='custom') if pj is None else dict(kind='template', **{k: v for k, v in pj.items()})


def save_settings(name, body):
    d = pdir(name); pj = read_pj(d)
    if pj is None:
        raise ApiError(400, 'this project uses its own hand-built scenes; styles are not available')
    for k in SETTING_KEYS:
        if k in body:
            pj[k] = body[k]
    write_pj(d, pj)
    return dict(ok=True)


def set_lyrics(name, body):
    d = pdir(name); pj = read_pj(d)
    if pj is None:
        raise ApiError(400, 'line changes are only available for ADx template projects')
    pj['lyrics'] = body.get('lyrics', ''); write_pj(d, pj)
    if os.path.isfile(audio_path(d)):
        res = analyze(name)
        return res
    return dict(ok=True)


def upload_image(name, kind, data, fname):
    d = pdir(name); pj = read_pj(d)
    if pj is None:
        raise ApiError(400, 'only for ADx template projects')
    ext = os.path.splitext(fname or '')[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.webp'):
        raise ApiError(400, 'use a PNG, JPG or WebP image')
    rel = 'assets/%s%s' % ('background' if kind == 'bg' else 'logo_custom', ext)
    for old in os.listdir(os.path.join(d, 'assets')):
        if old.startswith('background' if kind == 'bg' else 'logo_custom'):
            os.remove(os.path.join(d, 'assets', old))
    write_atomic(os.path.join(d, rel), data, binary=True)
    pj['bg_image' if kind == 'bg' else 'logo'] = rel; write_pj(d, pj)
    return dict(ok=True, path=rel)


SYNCS = {}


def start_autosync(name):
    d = pdir(name)
    if not os.path.isfile(audio_path(d)):
        raise ApiError(400, 'load the song first')
    for j in SYNCS.values():
        if j['name'] == name and j['p'].poll() is None:
            raise ApiError(400, 'auto sync is already running for this project')
    try:
        import importlib.util
        if importlib.util.find_spec('sherpa_onnx') is None or importlib.util.find_spec('onnxruntime') is None:
            raise ImportError
    except ImportError:
        raise ApiError(400, 'auto sync components are missing: quit and reopen ADx Studio to install them '
                            '(or run: python3 -m pip install sherpa-onnx onnxruntime)')
    jid = uuid.uuid4().hex[:8]
    pf = os.path.join(d, 'out', 'studio', 'autosync_%s.json' % jid); os.makedirs(os.path.dirname(pf), exist_ok=True)
    shutil.copyfile(tl_path(d), os.path.join(d, 'analysis', 'timeline_before_autosync.json'))
    p = subprocess.Popen([sys.executable, os.path.join(HERE, 'studio_autosync.py'), d, pf], cwd=HERE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    SYNCS[jid] = dict(name=name, p=p, pf=pf, started=time.time())
    return jid


def autosync_status(jid):
    j = SYNCS.get(jid)
    if not j: raise ApiError(404, 'no such job')
    try:
        st = load_json(j['pf'])
    except Exception:
        st = dict(stage='start', frac=0.0, msg='Starting…')
    rc = j['p'].poll()
    if rc is not None and st.get('stage') != 'done':
        err = (j['p'].stderr.read() or '').strip().splitlines()
        st = dict(stage='error', frac=0, msg=st.get('msg') if st.get('stage') == 'error' else (err[-1] if err else 'auto sync failed'))
    st['elapsed'] = round(time.time() - j['started'], 1)
    return st


def delete_project(name):
    d = pdir(name)
    trash = os.path.join(PROJ, '.deleted'); os.makedirs(trash, exist_ok=True)
    os.replace(d, os.path.join(trash, '%s-%d' % (name, int(time.time()))))
    for k in [k for k in ENGINES if k[0] == name]:
        ENGINES.pop(k).close()
    return dict(ok=True)


def do_import(filename, data):
    base = os.path.splitext(os.path.basename(filename or 'project'))[0]
    base = re.sub(r'[^A-Za-z0-9_.-]', '_', base).strip('._') or 'project'
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ApiError(400, 'that file is not a zip')
    names = [n for n in zf.namelist() if not n.startswith('__MACOSX') and '/._' not in n]
    cands = [n for n in names if n == 'analysis/timeline.json' or n.endswith('/analysis/timeline.json')]
    if not cands:
        raise ApiError(400, 'no analysis/timeline.json in the zip: is this a KARAM project zip?')
    root = min(cands, key=len)[:-len('analysis/timeline.json')]
    name, k = base[:60], 2
    while os.path.exists(os.path.join(PROJ, name)):
        name = '%s-%d' % (base[:56], k); k += 1
    tmp = os.path.join(PROJ, '.import-' + uuid.uuid4().hex[:8])
    os.makedirs(tmp)
    real = os.path.realpath(tmp)
    try:
        for n in names:
            if not n.startswith(root) or n.endswith('/'):
                continue
            rel = n[len(root):]
            target = os.path.realpath(os.path.join(tmp, rel))
            if not target.startswith(real + os.sep):
                raise ApiError(400, 'unsafe path in zip: ' + n)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(n) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        orig = os.path.join(tmp, 'analysis', 'timeline_original.json')
        if not os.path.exists(orig):
            shutil.copyfile(tl_path(tmp), orig)
        os.replace(tmp, os.path.join(PROJ, name))
    finally:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return name


def project_data(name):
    d = pdir(name)
    ensure_hooks(d)
    T = load_json(tl_path(d))
    v = T['curves']['vocal']
    mx = max(v) or 1.0
    q = [int(min(255, max(0, round(x / mx * 255)))) for x in v]
    words = [dict(line=w['line'], wi=w['wi'], text=w['text'], start=w['start'], end=w['end']) for w in T['words']]
    orig = words
    op = os.path.join(d, 'analysis', 'timeline_original.json')
    if os.path.isfile(op):
        orig = [dict(line=w['line'], wi=w['wi'], text=w['text'], start=w['start'], end=w['end'])
                for w in load_json(op)['words']]
    st = get_settings(name)
    return dict(name=name, kind=st['kind'], settings=st, root=d, dur=T['dur'], fps=T['curves_fps'], t0=T['t0'], beat=T['beat'], vocal=q,
                kicks=[e[0] for e in T['events'].get('kick', [])], sections=T['sections'],
                lines=T['lines'], words=words, orig=orig, audio=os.path.isfile(audio_path(d)))


def save_timeline(name, body):
    d = pdir(name)
    T = load_json(tl_path(d))
    lines_in, words_in = body.get('lines'), body.get('words')
    if not isinstance(lines_in, list) or not isinstance(words_in, list):
        raise ApiError(400, 'lines and words are required')
    if len(lines_in) != len(T['lines']) and read_pj(d) is None:
        raise ApiError(400, 'the number of lyric lines cannot change in this prototype')
    old = {(w['line'], w['wi']): w for w in T['words']}
    per_line = {}
    out = []
    for w in words_in:
        try:
            li, wi = int(w['line']), int(w['wi'])
            s, e = float(w['start']), float(w['end'])
            text = str(w['text']).strip()
        except Exception:
            raise ApiError(400, 'malformed word entry')
        if not (0 <= li < len(lines_in)) or wi < 0 or not text or ' ' in text:
            raise ApiError(400, 'bad word at line %s word %s' % (li, wi))
        if not (math.isfinite(s) and math.isfinite(e)) or s < 0 or e <= s or e > T['dur'] + 1:
            raise ApiError(400, 'bad timing at line %d word %d (start %.3f end %.3f)' % (li, wi, s, e))
        per_line.setdefault(li, []).append(wi)
        o = old.get((li, wi), {})
        out.append(dict(line=li, wi=wi, text=text, start=round(s, 3), end=round(e, 3),
                        raw_start=o.get('raw_start', round(s, 3)), conf=o.get('conf', 0.0)))
    for li in range(len(lines_in)):
        if sorted(per_line.get(li, [])) != list(range(len(per_line.get(li, [])))) or not per_line.get(li):
            raise ApiError(400, 'line %d has missing or duplicate words' % li)
    out.sort(key=lambda w: (w['line'], w['wi']))
    lines_out = [' '.join(w['text'] for w in out if w['line'] == li) for li in range(len(lines_in))]
    T['words'], T['lines'] = out, lines_out
    write_atomic(tl_path(d), json.dumps(T))
    lp = os.path.join(d, 'assets', 'lyrics.txt')
    if os.path.isfile(lp):
        with open(lp, encoding='utf-8') as f:
            rows = f.read().split('\n')
        nonblank = [i for i, r in enumerate(rows) if r.strip()]
        if len(nonblank) == len(lines_out):
            for i, txt in zip(nonblank, lines_out):
                rows[i] = txt
            write_atomic(lp, '\n'.join(rows))
    return dict(ok=True, words=len(out))


def render_still(name, t, scale):
    d = pdir(name)
    st = os.stat(tl_path(d))
    key = hashlib.md5(('%d:%d:%.3f:%.3f' % (st.st_mtime_ns, st.st_size, t, scale)).encode()).hexdigest()[:16]
    cache = os.path.join(d, 'out', 'studio')
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, key + '.png')
    if os.path.isfile(out):
        return out
    with STILL_LOCK:
        if os.path.isfile(out):
            return out
        r = subprocess.run([sys.executable, os.path.join('src', 'main.py'), 'still', '%.3f' % t, '%.3f' % scale, out],
                           cwd=d, capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not os.path.isfile(out):
            tail = (r.stderr or r.stdout or '').strip().splitlines()[-6:]
            raise ApiError(500, 'renderer failed: ' + ' | '.join(tail))
        files = sorted((os.path.join(cache, f) for f in os.listdir(cache) if f.endswith('.png')), key=os.path.getmtime)
        for f in files[:-150]:
            try: os.remove(f)
            except OSError: pass
    return out


def start_clip(name, t0, t1, scale):
    # (frame size comes from the project's own W, H: 3840x1646 for KARAM projects)
    d = pdir(name)
    if not os.path.isfile(audio_path(d)):
        raise ApiError(400, 'load the song audio first (assets/karam.mp3)')
    # libx264 4:2:0 needs even frame sizes: nudge the scale until both dimensions are even
    for _ in range(40):
        if round(3840 * scale) % 2 == 0 and round(1646 * scale) % 2 == 0:
            break
        scale = round(scale - 0.001, 3)
    jid = uuid.uuid4().hex[:10]
    cache = os.path.join(d, 'out', 'studio')
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, 'clip_%s.mp4' % jid)
    job = dict(state='running', started=time.time(), out=out, error='')
    JOBS[jid] = job

    def run():
        try:
            with STILL_LOCK:
                r = subprocess.run([sys.executable, os.path.join('src', 'main.py'), 'clip', '%.3f' % t0, '%.3f' % t1,
                                    '%.3f' % scale, out, '30'], cwd=d, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0 and os.path.isfile(out):
                job['state'] = 'done'
            else:
                job['state'] = 'error'
                lines = (r.stderr or r.stdout or 'failed').strip().splitlines()
                keys = [l.strip() for l in lines if re.search(r'rror|not divisible|Invalid|No such', l) and 'BrokenPipe' not in l]
                job['error'] = ' | '.join((keys[:2] + lines[-1:]) if keys else lines[-4:])
        except Exception as e:
            job['state'] = 'error'; job['error'] = str(e)
            if 'ffmpeg' in str(e):
                job['error'] = 'ffmpeg is not installed. Run  pip3 install imageio-ffmpeg  in Terminal, then restart the server.'
        job['ended'] = time.time()
    threading.Thread(target=run, daemon=True).start()
    return jid


def export_zip(name, with_audio):
    d = pdir(name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for base, dirs, files in os.walk(d):
            rel_base = os.path.relpath(base, d)
            dirs[:] = [x for x in dirs if not (rel_base == '.' and x == 'out') and x != '__pycache__']
            for f in files:
                if f == '.DS_Store' or f.endswith('.pyc') or '.tmp' in f:
                    continue
                full = os.path.join(base, f)
                rel = os.path.normpath(os.path.join(rel_base, f))
                if rel == os.path.join('assets', 'karam.mp3') and not with_audio:
                    continue
                z.write(full, os.path.join(name, rel))
    return buf.getvalue()


# ---------------------------------------------------------------- v2: hooks, look, preview engine, renders
HOOK_MARK = '# studio-look-hook'


def ensure_hooks(d):
    """Copy look.py into the project and hook it into Renderer.post (idempotent, no visual change
    until a look is saved)."""
    src = os.path.join(d, 'src')
    try:
        shutil.copyfile(os.path.join(HERE, 'look.py'), os.path.join(src, 'look.py'))
        rp = os.path.join(src, 'render.py')
        with open(rp, encoding='utf-8') as f:
            s = f.read()
        if HOOK_MARK in s:
            return
        imp = 'from core import W, H, FPS, INK, BONE, EMBER, ROOT, clamp, rands\n'
        tail = "        arr[..., 3] = 255\n        return arr\n"
        if imp not in s or tail not in s:
            return
        s = s.replace(imp, imp + HOOK_MARK + '\ntry:\n    import look as _look\n    _LOOK = _look.load(ROOT)\n'
                      'except Exception:\n    _look, _LOOK = None, None\n', 1)
        s = s.replace(tail, "        if _look is not None and _LOOK is not None:  " + HOOK_MARK +
                      "\n            arr = _look.apply(arr, _LOOK)\n" + tail, 1)
        write_atomic(rp, s)
    except OSError:
        pass


def look_path(d): return os.path.join(d, 'analysis', 'look.json')


def get_look(name):
    d = pdir(name)
    sys.path.insert(0, HERE)
    import look
    try:
        return look.clean(load_json(look_path(d)))
    except Exception:
        return dict(look.DEFAULT)


def save_look(name, body):
    d = pdir(name)
    import look
    write_atomic(look_path(d), json.dumps(look.clean(body), indent=1))
    return dict(ok=True)


class Engine:
    """One persistent preview process per project; restarted when the timeline changes."""
    def __init__(self, d, grain):
        self.d, self.grain = d, grain
        self.key = self._key()
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, 'studio_worker.py'), d, grain],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1, cwd=d)
        first = self.p.stdout.readline()
        if not first:
            err = self.p.stderr.read()[-800:]
            raise ApiError(500, 'preview engine failed to start: ' + ' | '.join(err.strip().splitlines()[-4:]))

    def _key(self):
        st = os.stat(tl_path(self.d))
        srcs = [os.path.getmtime(os.path.join(self.d, 'src', f)) for f in os.listdir(os.path.join(self.d, 'src')) if f.endswith('.py')]
        return (st.st_mtime_ns, st.st_size, max(srcs) if srcs else 0)

    def alive(self): return self.p.poll() is None

    def stale(self): return self._key() != self.key

    def frame(self, req):
        self.p.stdin.write(json.dumps(req) + '\n'); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise ApiError(500, 'preview engine stopped: ' + ' | '.join(self.p.stderr.read()[-600:].strip().splitlines()[-3:]))
        r = json.loads(line)
        if not r.get('ok'):
            raise ApiError(500, 'render error: ' + r.get('err', '?'))
        return r

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


ENGINES = {}
ENGINE_LOCK = threading.Lock()
TICKETS = {}


def preview_frame(name, t, scale, fast, look_params, ticket, settings=None):
    d = pdir(name)
    TICKETS[name] = max(TICKETS.get(name, 0), ticket)
    grain = '0.4' if fast else '1.0'
    with ENGINE_LOCK:
        if ticket < TICKETS.get(name, 0):
            return None                      # a newer request is waiting: skip this one
        key = (name, grain)
        e = ENGINES.get(key)
        if e is not None and (not e.alive() or e.stale()):
            e.close(); e = None
        if e is None:
            for k in [k for k in ENGINES if k[0] == name and k != key]:
                if len(ENGINES) >= 2:
                    ENGINES.pop(k).close()
            e = ENGINES[key] = Engine(d, grain)
        cache = os.path.join(d, 'out', 'studio'); os.makedirs(cache, exist_ok=True)
        out = os.path.join(cache, 'live_%s.jpg' % ('f' if fast else 'q'))
        req = dict(t=t, scale=scale, fast=fast, look=look_params, out=out, q=80 if fast else 90)
        if settings is not None:
            req['settings'] = settings
        r = e.frame(req)
        with open(out, 'rb') as f:
            return f.read(), r['ms']


def snapshot(name, t, look_params):
    d = pdir(name)
    out_dir = os.path.join(d, 'out', 'stills'); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'frame_%07.3fs.png' % t)
    with ENGINE_LOCK:
        key = (name, '1.0')
        e = ENGINES.get(key)
        if e is None or not e.alive() or e.stale():
            if e: e.close()
            e = ENGINES[key] = Engine(d, '1.0')
        e.frame(dict(t=t, scale=1.0, fast=False, look=look_params, out=out))
    return out


def find_encoder():
    try:
        enc = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return 'libx264'
    return 'h264_videotoolbox' if 'h264_videotoolbox' in enc else 'libx264'


SYS = {}


def sysinfo():
    if not SYS:
        n = os.cpu_count() or 4
        SYS.update(cpus=n, encoder=find_encoder() if shutil.which('ffmpeg') else 'none',
                   default_workers=max(1, min(8, n - 2)), platform=sys.platform)
    return SYS


PRESETS = {
    'draft': dict(label='Draft 960 px, 24 fps', scale=0.25, fps=24, fast=True, vt='4M', crf='24'),
    'hd': dict(label='HD 1920 px, 30 fps', scale=0.5, fps=30, fast=False, vt='14M', crf='19'),
    'final': dict(label='Final 4K, 60 fps', scale=1.0, fps=60, fast=False, vt='60M', crf='18'),
}
RENDERS = {}


def start_render(name, body):
    d = pdir(name)
    info = sysinfo()
    if info['encoder'] == 'none':
        raise ApiError(400, 'ffmpeg is not installed (pip3 install imageio-ffmpeg, then restart)')
    pr = PRESETS.get(body.get('preset'))
    if not pr:
        raise ApiError(400, 'unknown preset')
    T = load_json(tl_path(d))
    t0 = max(0.0, float(body.get('t0', 0))); t1 = min(T['dur'], float(body.get('t1', T['dur'])))
    if t1 - t0 < 0.1:
        raise ApiError(400, 'range too short')
    workers = max(1, min(12, int(body.get('workers') or info['default_workers'])))
    fps = pr['fps']
    f0, f1 = int(round(t0 * fps)), int(math.ceil(t1 * fps))
    n = f1 - f0
    workers = min(workers, max(1, n // 24))
    jid = time.strftime('%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    wd = os.path.join(d, 'out', 'renders', '.work-' + jid); os.makedirs(wd)
    lookf = os.path.join(wd, 'look.json')
    lk = body.get('look')
    if lk:
        with open(lookf, 'w') as f: json.dump(lk, f)
    enc = info['encoder']
    rate = pr['vt'] if 'videotoolbox' in enc else pr['crf']
    bounds = [f0 + (n * i) // workers for i in range(workers + 1)]
    segs = []
    for i in range(workers):
        seg = os.path.join(wd, 'seg%02d.mp4' % i); prog = os.path.join(wd, 'seg%02d.prog' % i)
        p = subprocess.Popen([sys.executable, os.path.join(HERE, 'studio_render.py'), d, str(bounds[i]), str(bounds[i + 1]),
                              str(fps), str(pr['scale']), '1' if pr['fast'] else '0', enc, rate, seg, prog,
                              lookf if lk else '-'], cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        segs.append(dict(p=p, seg=seg, prog=prog, n=bounds[i + 1] - bounds[i]))
    fname = '%s_%s_%s-%ss.mp4' % (name, body.get('preset'), ('%.1f' % t0).replace('.', '_'), ('%.1f' % t1).replace('.', '_'))
    job = dict(state='running', started=time.time(), total=n, segs=segs, wd=wd, preset=body.get('preset'),
               out=os.path.join(d, 'out', 'renders', fname), file=fname, error='', t0=t0, t1=t1, workers=workers, enc=enc)
    RENDERS[jid] = job

    def finish():
        errs = []
        for s in segs:
            s['p'].wait()
            if s['p'].returncode != 0 and job['state'] != 'cancelled':
                errs.append(' | '.join((s['p'].stderr.read() or '').strip().splitlines()[-3:]))
        if job['state'] == 'cancelled':
            shutil.rmtree(wd, ignore_errors=True); return
        if errs:
            job['state'] = 'error'; job['error'] = errs[0] or 'a render process failed'; return
        job['state'] = 'muxing'
        lst = os.path.join(wd, 'list.txt')
        with open(lst, 'w') as f:
            for s in segs: f.write("file '%s'\n" % s['seg'])
        vid = os.path.join(wd, 'video.mp4')
        r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', vid],
                           capture_output=True, text=True)
        if r.returncode != 0:
            job['state'] = 'error'; job['error'] = r.stderr[-300:]; return
        if os.path.isfile(audio_path(d)):
            cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', vid, '-ss', '%.4f' % (f0 / fps), '-t', '%.4f' % (n / fps),
                   '-i', audio_path(d), '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k',
                   '-ar', '48000', '-shortest', '-movflags', '+faststart', job['out']]
        else:
            cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', vid, '-c', 'copy', '-movflags', '+faststart', job['out']]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            job['state'] = 'error'; job['error'] = r.stderr[-300:]; return
        shutil.rmtree(wd, ignore_errors=True)
        job['state'] = 'done'; job['ended'] = time.time()
    threading.Thread(target=finish, daemon=True).start()
    return jid


def render_status(jid):
    j = RENDERS.get(jid)
    if not j: raise ApiError(404, 'no such render')
    done = 0
    for s in j['segs']:
        try:
            with open(s['prog']) as f: done += int(f.read().strip() or 0)
        except (OSError, ValueError):
            pass
    if j['state'] in ('done', 'muxing'):
        done = j['total']
    el = time.time() - j['started']
    eta = (el / done * (j['total'] - done)) if done > 5 and j['state'] == 'running' else None
    return dict(state=j['state'], done=done, total=j['total'], elapsed=round(el, 1),
                eta=round(eta) if eta else None, file=j['file'], error=j['error'], workers=j['workers'], encoder=j['enc'])


def cancel_render(jid):
    j = RENDERS.get(jid)
    if not j: raise ApiError(404, 'no such render')
    j['state'] = 'cancelled'
    for s in j['segs']:
        try: s['p'].kill()
        except Exception: pass
    return dict(ok=True)


def list_renders(name):
    d = pdir(name); rd = os.path.join(d, 'out', 'renders')
    if not os.path.isdir(rd): return []
    out = []
    for f in sorted(os.listdir(rd), key=lambda f: -os.path.getmtime(os.path.join(rd, f))):
        if f.endswith('.mp4'):
            out.append(dict(file=f, mb=round(os.path.getsize(os.path.join(rd, f)) / 1e6, 1)))
    return out[:30]


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = 'ADxStudio/1.0'

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode('utf-8'), 'application/json')

    def _file(self, path, ctype):
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get('Range')
        if rng:
            m = re.match(r'bytes=(\d*)-(\d*)$', rng.strip())
            if m:
                a, b = m.groups()
                if a == '' and b:
                    start = max(0, size - int(b))
                elif a:
                    start = int(a)
                    if b: end = min(int(b), size - 1)
                code = 206
        if start > end or start >= size:
            self.send_response(416); self.send_header('Content-Range', 'bytes */%d' % size)
            self.send_header('Content-Length', '0'); self.end_headers(); return
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store')
        if code == 206:
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
        self.end_headers()
        try:
            with open(path, 'rb') as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk: break
                    self.wfile.write(chunk); left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_UPLOAD:
            raise ApiError(413, 'file too large')
        return self.rfile.read(n)

    def do_GET(self): self._route('GET')
    def do_POST(self): self._route('POST')

    def _route(self, method):
        try:
            u = urlparse(self.path); q = parse_qs(u.query); p = u.path
            if method == 'GET' and p in ('/', '/index.html'):
                return self._file(os.path.join(HERE, 'static', 'index.html'), 'text/html; charset=utf-8')
            if method == 'GET' and p.startswith('/static/'):
                rel = os.path.normpath(p[len('/static/'):])
                fp = os.path.join(HERE, 'static', rel)
                if rel.startswith('..') or not os.path.isfile(fp): raise ApiError(404, 'not found')
                ct = {'.png': 'image/png', '.jpg': 'image/jpeg', '.json': 'application/json', '.svg': 'image/svg+xml'}.get(os.path.splitext(fp)[1], 'application/octet-stream')
                return self._file(fp, ct)
            if method == 'GET' and p == '/api/ping':
                return self._json(dict(ok=True, app='ADx Studio'))
            if method == 'POST' and p == '/api/quit':
                self._json(dict(ok=True))
                threading.Thread(target=lambda: (time.sleep(0.3), [e.close() for e in list(ENGINES.values())], os._exit(0)), daemon=True).start()
                return
            if method == 'POST' and p == '/api/new':
                return self._json(dict(name=new_project(json.loads(self._body().decode('utf-8')))))
            if method == 'GET' and p == '/api/projects':
                return self._json(list_projects())
            if method == 'POST' and p == '/api/import':
                name = do_import(q.get('filename', ['project.zip'])[0], self._body())
                return self._json(dict(name=name))
            m = re.match(r'^/api/p/([^/]+)/([a-z]+)(?:/([A-Za-z0-9_.-]+))?$', p)
            if not m:
                raise ApiError(404, 'not found')
            name, action, arg = m.groups()
            if method == 'GET' and action == 'data':
                return self._json(project_data(name))
            if method == 'POST' and action == 'save':
                return self._json(save_timeline(name, json.loads(self._body().decode('utf-8'))))
            if method == 'GET' and action == 'audio':
                d = pdir(name)
                if not os.path.isfile(audio_path(d)): raise ApiError(404, 'no audio loaded')
                return self._file(audio_path(d), 'audio/mpeg')
            if method == 'POST' and action == 'audio':
                d = pdir(name)
                data = self._body()
                if len(data) < 1000: raise ApiError(400, 'that does not look like an audio file')
                os.makedirs(os.path.join(d, 'assets'), exist_ok=True)
                write_atomic(audio_path(d), data, binary=True)
                res = analyze(name) if read_pj(d) is not None else {}
                return self._json(dict(ok=True, bytes=len(data), analysis=res))
            if method == 'GET' and action == 'still' and not arg:
                t = float(q.get('t', ['0'])[0]); s = float(q.get('s', ['0.3'])[0])
                if not (0 <= t <= 1e5 and 0.05 <= s <= 1.0): raise ApiError(400, 'bad t or scale')
                return self._file(render_still(name, t, s), 'image/png')
            if method == 'POST' and action == 'clip':
                b = json.loads(self._body().decode('utf-8'))
                t0, t1, sc = float(b['t0']), float(b['t1']), float(b.get('scale', 0.25))
                if not (0 <= t0 < t1 <= t0 + 30 and 0.05 <= sc <= 0.6): raise ApiError(400, 'bad clip range or scale')
                return self._json(dict(job=start_clip(name, t0, t1, sc)))
            if method == 'GET' and action == 'job':
                j = JOBS.get(arg or '')
                if not j: raise ApiError(404, 'no such job')
                return self._json(dict(state=j['state'], error=j['error'], elapsed=round(time.time() - j['started'], 1)))
            if method == 'GET' and action == 'clipfile':
                j = JOBS.get((arg or '').replace('.mp4', ''))
                if not j or j['state'] != 'done': raise ApiError(404, 'clip not ready')
                return self._file(j['out'], 'video/mp4')
            if method == 'GET' and action == 'export':
                data = export_zip(name, q.get('audio', ['0'])[0] == '1')
                return self._send(200, data, 'application/zip',
                                  {'Content-Disposition': 'attachment; filename="%s_edited.zip"' % name})
            if method == 'GET' and action == 'settings':
                return self._json(get_settings(name))
            if method == 'POST' and action == 'settings':
                return self._json(save_settings(name, json.loads(self._body().decode('utf-8'))))
            if method == 'POST' and action == 'lyrics':
                return self._json(set_lyrics(name, json.loads(self._body().decode('utf-8'))))
            if method == 'POST' and action == 'analyze':
                return self._json(analyze(name))
            if method == 'POST' and action == 'image':
                return self._json(upload_image(name, q.get('kind', ['bg'])[0], self._body(), q.get('filename', [''])[0]))
            if method == 'GET' and action == 'asset' and arg:
                fp = os.path.join(pdir(name), 'assets', os.path.basename(arg))
                if not os.path.isfile(fp): raise ApiError(404, 'not found')
                return self._file(fp, 'image/png' if fp.endswith('.png') else 'image/jpeg')
            if method == 'POST' and action == 'autosync':
                return self._json(dict(job=start_autosync(name)))
            if method == 'GET' and action == 'autosync' and arg:
                return self._json(autosync_status(arg))
            if method == 'POST' and action == 'delete':
                return self._json(delete_project(name))
            if method == 'GET' and action == 'look':
                return self._json(get_look(name))
            if method == 'POST' and action == 'look':
                return self._json(save_look(name, json.loads(self._body().decode('utf-8'))))
            if method == 'GET' and action == 'frame':
                t = float(q.get('t', ['0'])[0]); s = float(q.get('s', ['0.15'])[0])
                fast = q.get('fast', ['1'])[0] == '1'
                lk = json.loads(q.get('look', ['null'])[0])
                if not (0 <= t <= 1e5 and 0.05 <= s <= 1.0): raise ApiError(400, 'bad t or scale')
                stg = json.loads(q.get('settings', ['null'])[0])
                r = preview_frame(name, t, s, fast, lk, int(q.get('ticket', ['0'])[0]), stg)
                if r is None:
                    self.send_response(204); self.send_header('Content-Length', '0'); self.end_headers(); return
                return self._send(200, r[0], 'image/jpeg', {'X-Render-Ms': str(r[1])})
            if method == 'POST' and action == 'snapshot':
                b = json.loads(self._body().decode('utf-8'))
                out = snapshot(name, float(b['t']), b.get('look'))
                return self._json(dict(file=os.path.basename(out)))
            if method == 'GET' and action == 'still' and arg:
                p2 = os.path.join(pdir(name), 'out', 'stills', os.path.basename(arg))
                if not os.path.isfile(p2): raise ApiError(404, 'no such still')
                return self._send(200, open(p2, 'rb').read(), 'image/png', {'Content-Disposition': 'attachment; filename="%s"' % os.path.basename(arg)})
            if method == 'POST' and action == 'render':
                return self._json(dict(job=start_render(name, json.loads(self._body().decode('utf-8')))))
            if method == 'GET' and action == 'render' and arg:
                return self._json(render_status(arg))
            if method == 'POST' and action == 'cancel' and arg:
                return self._json(cancel_render(arg))
            if method == 'GET' and action == 'renders':
                return self._json(dict(renders=list_renders(name), presets={k: v['label'] for k, v in PRESETS.items()}, sys=sysinfo()))
            if method == 'GET' and action == 'renderfile' and arg:
                p2 = os.path.join(pdir(name), 'out', 'renders', os.path.basename(arg))
                if not os.path.isfile(p2): raise ApiError(404, 'no such render')
                return self._file(p2, 'video/mp4')
            raise ApiError(404, 'not found')
        except ApiError as e:
            self._json(dict(error=e.msg), e.code)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._json(dict(error='%s: %s' % (type(e).__name__, e)), 500)


def ensure_ffmpeg():
    """Preview clips need ffmpeg. If it is not on PATH, fall back to the one bundled with the
    `imageio-ffmpeg` pip package (pip3 install imageio-ffmpeg), exposed to the renderer as `ffmpeg`."""
    if shutil.which('ffmpeg'):
        return True
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    bindir = os.path.join(HERE, '.bin')
    os.makedirs(bindir, exist_ok=True)
    link = os.path.join(bindir, 'ffmpeg')
    try:
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(exe, link)
    except OSError:
        shutil.copy2(exe, link)
    os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    os.makedirs(PROJ, exist_ok=True)
    if ensure_ffmpeg():
        print('ffmpeg:', shutil.which('ffmpeg'))
    else:
        print('NOTE: ffmpeg not found. Clip previews need it: run  pip3 install imageio-ffmpeg  (or brew install ffmpeg) and restart.')
    import atexit
    atexit.register(lambda: [e.close() for e in list(ENGINES.values())])
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    url = 'http://127.0.0.1:%d' % a.port
    print('ADx Studio running at', url, ' (Ctrl+C to stop)')
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('bye')


if __name__ == '__main__':
    main()

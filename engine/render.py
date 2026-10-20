"""ADx Studio template engine: renders a lyric video from analysis/timeline.json + project.json.

Every frame is a pure function of time t. Templates are small draw functions over a shared context
(timing, audio energy, beat, fonts, settings). Drawing happens in "base" pixels (3840 wide for 16:9)
and is scaled by the Renderer, so previews at low resolution are cheap.
"""
import json, math, os
import numpy as np
import skia
import cv2

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
try:
    import look as _look
    _LOOK = _look.load(ROOT)
except Exception:
    _look, _LOOK = None, None

ASPECTS = {'16:9': (3840, 2160), '9:16': (2160, 3840), '1:1': (2160, 2160), '4:5': (2160, 2700), '21:9': (3840, 1646)}

# ---------------------------------------------------------------- templates (metadata + defaults)
TEMPLATES = {
    'ink': dict(name='Ink & Ember', desc='Words appear as they are sung while a glowing pen draws a brush line under them.',
                font='Instrument Serif Italic', case='as', bg1='#0b0b0e', bg2='#17110d', text='#f1ece2', accent='#ff6a1a', size=1.0),
    'kinetic': dict(name='Kinetic Slam', desc='One huge word at a time, slamming in on the beat with camera shake.',
                    font='Big Shoulders Black', case='upper', bg1='#0a0a0c', bg2='#1b0f0a', text='#ffffff', accent='#ff6a1a', size=1.0),
    'neon': dict(name='Neon Night', desc='Glowing neon tube lettering that flickers on, over a pulsing synthwave grid.',
                 font='Big Shoulders', case='upper', bg1='#07040f', bg2='#2a0b3d', text='#ffffff', accent='#ff3df2', size=1.0),
    'karaoke': dict(name='Clean Karaoke', desc='The full line on screen, filling with colour word by word; next line below.',
                    font='Archivo Bold', case='as', bg1='#101422', bg2='#2b1638', text='#ffffff', accent='#ffcf3d', size=1.0),
    'typewriter': dict(name='Typewriter', desc='Letters are typed out on paper with a blinking cursor; the page scrolls.',
                       font='Courier Prime', case='as', bg1='#efe7d6', bg2='#e2d6bd', text='#1e1b17', accent='#b3261e', size=1.0),
    'cinematic': dict(name='Cinematic', desc='Elegant serif lines that resolve out of blur, letterbox bars and light leaks.',
                      font='Instrument Serif', case='as', bg1='#08080a', bg2='#1a1410', text='#f3eee4', accent='#ffb36b', size=1.0),
    'social': dict(name='Social Captions', desc='Bold short phrases with the active word on a colour block. Made for Reels and Shorts.',
                   font='Archivo Black', case='upper', bg1='#111111', bg2='#2a2a2a', text='#ffffff', accent='#ffd400', size=1.0),
}
DEFAULTS = dict(template='ink', aspect='16:9', title='', artist='', bg1='', bg2='', text='', accent='', font='', size=1.0,
                position='center', logo_intro=True, logo_outro=True, watermark=True, title_card=True, visualizer=False,
                particles=True, pulse=0.5, bg_image='', bg_dim=0.55, logo='assets/logo.png')

FONT_FILES = {
    'Archivo': ('Archivo[wdth,wght].ttf', dict(wght=500, wdth=100)),
    'Archivo Bold': ('Archivo[wdth,wght].ttf', dict(wght=800, wdth=100)),
    'Archivo Wide': ('Archivo[wdth,wght].ttf', dict(wght=800, wdth=125)),
    'Archivo Condensed': ('Archivo[wdth,wght].ttf', dict(wght=800, wdth=62)),
    'Archivo Black': ('Archivo[wdth,wght].ttf', dict(wght=900, wdth=100)),
    'Big Shoulders': ('BigShouldersDisplay[wght].ttf', dict(wght=500)),
    'Big Shoulders Black': ('BigShouldersDisplay[wght].ttf', dict(wght=900)),
    'Instrument Serif': ('InstrumentSerif-Regular.ttf', {}),
    'Instrument Serif Italic': ('InstrumentSerif-Italic.ttf', {}),
    'Fraunces': ('Fraunces[SOFT,WONK,opsz,wght].ttf', dict(wght=600)),
    'Courier Prime': ('CourierPrime-Regular.ttf', {}),
    'Courier Prime Bold': ('CourierPrime-Bold.ttf', {}),
    'Kalam': ('Kalam-Bold.ttf', {}),
}


def font_dirs():
    return [os.path.join(ROOT, 'assets', 'fonts'), os.environ.get('ADX_FONTS', ''), os.path.join(SRC, 'fonts')]


_TF, _KEEP = {}, []


def typeface(name):
    if name in _TF:
        return _TF[name]
    fname, axes = FONT_FILES.get(name, FONT_FILES['Archivo Bold'])
    path = next((os.path.join(d, fname) for d in font_dirs() if d and os.path.isfile(os.path.join(d, fname))), None)
    tf = skia.Typeface.MakeFromFile(path) if path else skia.Typeface.MakeDefault()
    if axes and path:
        C = skia.FontArguments.VariationPosition.Coordinate
        coords = skia.FontArguments.VariationPosition.Coordinates(
            [C(int.from_bytes(k.encode(), 'big'), float(v)) for k, v in axes.items()])
        pos = skia.FontArguments.VariationPosition(coords)
        fa = skia.FontArguments(); fa.setVariationDesignPosition(pos)
        tf = tf.makeClone(fa)
        _KEEP.append((coords, pos, fa))
    _TF[name] = tf
    return tf


def mkfont(name, size):
    f = skia.Font(typeface(name), size)
    f.setSubpixel(True); f.setEdging(skia.Font.Edging.kAntiAlias)
    return f


def hexc(h, a=1.0):
    h = (h or '#ffffff').lstrip('#')
    if len(h) == 3: h = ''.join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    except ValueError:
        r, g, b = 1, 1, 1
    return skia.Color4f(r, g, b, max(0.0, min(1.0, a)))


def clamp(x, a=0.0, b=1.0): return a if x < a else b if x > b else x
def smooth(u): u = clamp(u); return u * u * (3 - 2 * u)
def ease_out(u): u = clamp(u); return 1 - (1 - u) ** 3
def ease_out_back(u, s=1.7):
    u = clamp(u); return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2
def hrand(i, seed=0):
    x = (i * 374761393 + seed * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0xFFFFFF)
def paint(col, stroke=None, blur=0.0, cap=True):
    p = skia.Paint(AntiAlias=True, Color4f=col)
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(stroke)
        if cap: p.setStrokeCap(skia.Paint.kRound_Cap); p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


# ---------------------------------------------------------------- film (data + settings)
class Film:
    def __init__(self):
        self.T = json.load(open(os.path.join(ROOT, 'analysis', 'timeline.json'), encoding='utf-8'))
        try:
            self.base_settings = json.load(open(os.path.join(ROOT, 'project.json'), encoding='utf-8'))
        except Exception:
            self.base_settings = {}
        self.dur = float(self.T['dur'])
        self.fps_c = float(self.T.get('curves_fps', 100))
        c = self.T.get('curves', {})
        self.c_rms = np.asarray(c.get('vocal', [0.0]), dtype=np.float32)
        self.c_low = np.asarray(c.get('low', c.get('vocal', [0.0])), dtype=np.float32)
        self.beat = float(self.T.get('beat', 0.5)) or 0.5
        self.t0 = float(self.T.get('t0', 0.0))
        self.kicks = np.asarray([e[0] for e in self.T.get('events', {}).get('kick', [])], dtype=np.float64)
        lines = {}
        for w in self.T['words']:
            lines.setdefault(w['line'], []).append(w)
        self.lines = []
        for li in sorted(lines):
            ws = sorted(lines[li], key=lambda w: w['wi'])
            self.lines.append(dict(words=ws, start=ws[0]['start'], end=ws[-1]['end']))
        self.words = [w for L in self.lines for w in L['words']]
        self._img = {}
        self.set_settings(None)

    def set_settings(self, override):
        s = dict(DEFAULTS); s.update(self.base_settings or {}); s.update(override or {})
        tp = TEMPLATES.get(s.get('template'), TEMPLATES['ink'])
        s['template'] = s['template'] if s.get('template') in TEMPLATES else 'ink'
        for k in ('bg1', 'bg2', 'text', 'accent', 'font', 'case'):
            if not s.get(k):
                s[k] = tp.get(k, '')
        self.s = s
        self.W, self.H = ASPECTS.get(s.get('aspect'), ASPECTS['16:9'])
        self.U = min(self.W, self.H)

    def image(self, rel):
        if not rel:
            return None
        p = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            return None
        key = (p, os.path.getmtime(p))
        if key not in self._img:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            m = 2400.0 / max(img.shape[:2])
            if m < 1:
                img = cv2.resize(img, None, fx=m, fy=m, interpolation=cv2.INTER_AREA)
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self._img = {k: v for k, v in self._img.items() if k[0] != p}
            self._img[key] = skia.Image.fromarray(np.ascontiguousarray(rgba), colorType=skia.kRGBA_8888_ColorType)
        return self._img[key]

    def curve(self, arr, t):
        x = t * self.fps_c
        if x <= 0: return float(arr[0])
        if x >= len(arr) - 1: return float(arr[-1])
        i = int(x); f = x - i
        return float(arr[i] * (1 - f) + arr[i + 1] * f)

    def beat_pulse(self, t):
        ph = ((t - self.t0) / self.beat) % 1.0
        return math.exp(-ph * 6.0)

    def kick_env(self, t):
        if not len(self.kicks): return 0.0
        i = np.searchsorted(self.kicks, t, side='right') - 1
        return math.exp(-(t - self.kicks[i]) * 10.0) if i >= 0 else 0.0

    def first_start(self): return self.lines[0]['start'] if self.lines else self.dur
    def last_end(self): return self.lines[-1]['end'] if self.lines else 0.0


def disp(text, case):
    return text.upper() if case == 'upper' else text.lower() if case == 'lower' else text


# ---------------------------------------------------------------- context passed to templates
class Ctx:
    def __init__(self, film, t, canvas):
        self.f, self.t, self.c, self.s = film, t, canvas, film.s
        self.W, self.H, self.U = film.W, film.H, film.U
        self.e = film.curve(film.c_rms, t)
        self.low = film.curve(film.c_low, t)
        self.bp = film.beat_pulse(t)
        self.kick = film.kick_env(t)
        self.text = hexc(self.s['text']); self.accent = hexc(self.s['accent'])
        self.size = float(self.s.get('size') or 1.0)
        self.cy = self.H * (0.72 if self.s.get('position') == 'lower' else 0.30 if self.s.get('position') == 'upper' else 0.5)

    def col(self, which, a):
        c = self.text if which == 'text' else self.accent
        return skia.Color4f(c.fR, c.fG, c.fB, clamp(a))

    def line_at(self, lead=0.3, hold=1.2, gap=0.12):
        """(index, enter 0..1, exit 0..1) of the line on screen at t, or None."""
        L = self.f.lines; t = self.t
        i = -1
        for k, ln in enumerate(L):
            if ln['start'] - lead <= t: i = k
            else: break
        if i < 0: return None
        ln = L[i]
        end = ln['end'] + hold
        if i + 1 < len(L):
            end = min(end, max(L[i + 1]['start'] - lead - gap, ln['end'] + 0.1))
        if t > end: return None
        return i, clamp((t - (ln['start'] - lead)) / max(lead, 1e-3)), clamp((t - (end - 0.3)) / 0.3)

    def layout(self, words, font, maxw, case, space=None):
        """Wrap words into centred rows. Returns list of (word, x, width, row) and row count, line height."""
        sp = font.measureText(' ') if space is None else space
        items, rows, cur, curw = [], [], [], 0.0
        for w in words:
            txt = disp(w['text'], case); ww = font.measureText(txt)
            if cur and curw + sp + ww > maxw:
                rows.append((cur, curw)); cur, curw = [], 0.0
            curw = curw + (sp if cur else 0) + ww; cur.append((w, txt, ww))
        if cur: rows.append((cur, curw))
        for r, (row, rw) in enumerate(rows):
            x = (self.W - rw) / 2
            for w, txt, ww in row:
                items.append(dict(w=w, txt=txt, x=x, wd=ww, row=r)); x += ww + sp
        return items, len(rows)

    def fit_font(self, name, size, words, case, maxw, max_rows=3):
        f = mkfont(name, size)
        for _ in range(12):
            items, nr = self.layout(words, f, maxw, case)
            widest = max([it['wd'] for it in items] or [0])
            if nr <= max_rows and widest <= maxw: return f, items, nr
            size *= 0.88; f = mkfont(name, size)
        return f, items, nr


def prog(w, t): return clamp((t - w['start']) / max(0.04, w['end'] - w['start']))


# ---------------------------------------------------------------- shared layers
def draw_background(x):
    c, s, f = x.c, x.s, x.f
    g = skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(x.W * 0.3, x.H)], [hexc(s['bg1']), hexc(s['bg2'])])
    c.drawPaint(skia.Paint(Shader=g))
    img = f.image(s.get('bg_image'))
    if img is not None:
        k = max(x.W / img.width(), x.H / img.height()) * (1.06 + 0.04 * math.sin(x.t * 0.05))
        w, h = img.width() * k, img.height() * k
        dx = (x.W - w) / 2 + math.sin(x.t * 0.031) * x.W * 0.01; dy = (x.H - h) / 2
        c.drawImageRect(img, skia.Rect.MakeXYWH(dx, dy, w, h), skia.SamplingOptions(skia.FilterMode.kLinear))
        c.drawPaint(skia.Paint(Color4f=skia.Color4f(0, 0, 0, clamp(float(s.get('bg_dim', 0.55))))))
    pulse = float(s.get('pulse', 0.5))
    if pulse > 0:
        c.drawPaint(skia.Paint(Color4f=x.col('accent', 0.07 * pulse * x.bp * (0.4 + x.low))))


def draw_particles(x, n=70, color='accent'):
    c = x.c
    for i in range(n):
        sp = 0.01 + 0.03 * hrand(i, 3)
        px = (hrand(i, 1) + 0.02 * math.sin(x.t * 0.3 + i)) * x.W
        py = (1.0 - ((hrand(i, 2) + x.t * sp) % 1.15)) * x.H
        r = x.U * (0.0012 + 0.003 * hrand(i, 4)) * (1 + 0.6 * x.low)
        a = (0.25 + 0.5 * hrand(i, 5)) * (0.5 + 0.5 * math.sin(x.t * (1 + 2 * hrand(i, 6)) + i))
        c.drawCircle(px, py, r, paint(x.col(color, a * 0.8), blur=r * 0.8))


def draw_visualizer(x):
    c, f = x.c, x.f
    n = 64; bw = x.W * 0.8 / n; base = x.H * 0.94
    for k in range(n):
        d = abs(k - n / 2 + 0.5) / (n / 2)
        v = f.curve(f.c_low, x.t - d * 0.35) * 0.6 + f.curve(f.c_rms, x.t - d * 0.2) * 0.5
        h = x.H * 0.09 * v * (1 - 0.5 * d) + x.U * 0.004
        r = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x.W * 0.1 + k * bw + bw * 0.2, base - h, bw * 0.6, h), bw * 0.3, bw * 0.3)
        c.drawRRect(r, paint(x.col('accent', 0.75)))


def draw_logo(x, alpha, scale, cy=None, size=None):
    img = x.f.image(x.s.get('logo') or 'assets/logo.png')
    if img is None or alpha <= 0.003: return
    sz = (size or x.U * 0.42) * scale
    k = sz / max(img.width(), img.height()); w, h = img.width() * k, img.height() * k
    cy = x.H * 0.45 if cy is None else cy
    r = skia.Rect.MakeXYWH((x.W - w) / 2, cy - h / 2, w, h)
    glow = skia.Paint(Alphaf=alpha * 0.55, ImageFilter=skia.ImageFilters.Blur(sz * 0.03, sz * 0.03),
                      ColorFilter=skia.ColorFilters.Blend(x.col('accent', 1), skia.BlendMode.kSrcIn))
    x.c.drawImageRect(img, r, skia.SamplingOptions(skia.FilterMode.kLinear), glow)
    x.c.drawImageRect(img, r, skia.SamplingOptions(skia.FilterMode.kLinear), skia.Paint(Alphaf=alpha))


def draw_title_block(x, alpha, y):
    s = x.s
    if alpha <= 0.003: return
    if s.get('title'):
        f = mkfont('Archivo Wide', x.U * 0.055); tw = f.measureText(s['title'].upper())
        x.c.drawString(s['title'].upper(), (x.W - tw) / 2, y, f, paint(x.col('text', alpha)))
    if s.get('artist'):
        f2 = mkfont('Archivo', x.U * 0.03); txt = s['artist'].upper(); tw = f2.measureText(txt)
        x.c.drawString(txt, (x.W - tw) / 2, y + x.U * 0.07, f2, paint(x.col('accent', alpha)))


def intro_outro(x):
    """Logo intro (before the first lyric) and outro (after the last). Returns 'lyrics' visibility 0..1."""
    f, s, t = x.f, x.s, x.t
    vis = 1.0
    i_end = clamp(f.first_start() - 0.35, 0.0, 7.0)
    if (s.get('logo_intro') or s.get('title_card')) and i_end > 1.2 and t < i_end:
        a = smooth(t / 0.8) * (1 - smooth((t - (i_end - 0.6)) / 0.6))
        if s.get('logo_intro'):
            draw_logo(x, a, 0.94 + 0.06 * ease_out(t / i_end), cy=x.H * (0.42 if s.get('title_card') else 0.5))
        if s.get('title_card'):
            draw_title_block(x, a, x.H * (0.73 if s.get('logo_intro') else 0.5))
        vis = 0.0
    o_start = f.last_end() + 1.0
    if s.get('logo_outro') and t > o_start and f.dur - o_start > 1.5:
        a = smooth((t - o_start) / 0.9) * (1 - smooth((t - (f.dur - 0.8)) / 0.8))
        draw_logo(x, a, 1.0, cy=x.H * 0.42)
        draw_title_block(x, a, x.H * 0.73)
    if s.get('watermark') and i_end < t < o_start:
        img = f.image(s.get('logo') or 'assets/logo.png')
        if img is not None:
            sz = x.U * 0.08; k = sz / max(img.width(), img.height())
            r = skia.Rect.MakeXYWH(x.W - sz * 1.4, x.H - sz * 1.4, img.width() * k, img.height() * k)
            x.c.drawImageRect(img, r, skia.SamplingOptions(skia.FilterMode.kLinear), skia.Paint(Alphaf=0.35))
    return vis


# ---------------------------------------------------------------- templates
def t_ink(x):
    st = x.line_at(lead=0.15, hold=1.4)
    if not st: return
    i, enter, exit_ = st
    ln = x.f.lines[i]
    font, items, nr = x.fit_font(x.s['font'], x.U * 0.085 * x.size, ln['words'], x.s['case'], x.W * 0.8)
    lh = font.getSize() * 1.35
    top = x.cy - (nr - 1) * lh / 2
    c = x.c
    c.save(); c.translate(0, -exit_ * x.U * 0.03)
    la = 1 - exit_
    head = None
    for it in items:
        w = it['w']; y = top + it['row'] * lh
        a = clamp((x.t - w['start'] + 0.12) / 0.2) * la
        if a > 0:
            c.drawString(it['txt'], it['x'], y + (1 - ease_out(a)) * x.U * 0.02, font, paint(x.col('text', a)))
        p = prog(w, x.t)
        if p > 0:
            uy = y + font.getSize() * 0.2; x0 = it['x'] - font.getSize() * 0.1; x1 = it['x'] + (it['wd'] + font.getSize() * 0.2) * p
            path = skia.Path(); n = max(2, int((x1 - x0) / 18))
            for k in range(n + 1):
                px = x0 + (x1 - x0) * k / n
                py = uy + math.sin(px * 0.011 + i) * x.U * 0.0022
                path.moveTo(px, py) if k == 0 else path.lineTo(px, py)
            c.drawPath(path, paint(x.col('text', 0.85 * la), stroke=x.U * (0.0035 + 0.004 * x.e)))
            if p < 1: head = (x1, uy)
    if head:
        hx, hy = head
        rad = skia.GradientShader.MakeRadial(skia.Point(hx, hy), x.U * 0.3, [x.col('accent', 0.22), x.col('accent', 0)], [0, 1])
        c.drawCircle(hx, hy, x.U * 0.3, skia.Paint(Shader=rad))
        c.drawCircle(hx, hy, x.U * 0.012, paint(x.col('accent', 0.9), blur=x.U * 0.01))
        c.drawCircle(hx, hy, x.U * 0.005, paint(hexc('#fff3e0')))
    c.restore()


def t_kinetic(x):
    f, t = x.f, x.t
    cur = None
    for k, w in enumerate(f.words):
        if w['start'] - 0.04 <= t: cur = k
        else: break
    if cur is None: return
    w = f.words[cur]; nxt = f.words[cur + 1]['start'] if cur + 1 < len(f.words) else w['end'] + 0.8
    if t > min(nxt, w['end'] + 0.9): return
    c = x.c
    sh = x.kick * x.U * 0.008 * (0.5 + x.e)
    c.save(); c.translate(math.sin(t * 91) * sh, math.cos(t * 77) * sh)
    ln = next(L for L in f.lines if w in L['words'])
    idx = ln['words'].index(w)
    for back in (2, 1):
        if idx - back < 0: continue
        pw = ln['words'][idx - back]; txt = disp(pw['text'], x.s['case'])
        pf = mkfont(x.s['font'], x.U * 0.07 * x.size); tw = pf.measureText(txt)
        c.drawString(txt, (x.W - tw) / 2, x.cy - x.U * (0.12 + 0.085 * back), pf, paint(x.col('text', 0.22 / back)))
    txt = disp(w['text'], x.s['case'])
    u = (t - w['start'] + 0.04) / 0.2
    size = x.U * 0.22 * x.size
    fnt = mkfont(x.s['font'], size); tw = fnt.measureText(txt)
    if tw > x.W * 0.9: fnt = mkfont(x.s['font'], size * x.W * 0.9 / tw); tw = fnt.measureText(txt)
    sc = 1 + 0.6 * (1 - ease_out_back(u))
    rot = (hrand(cur, 7) - 0.5) * 7 * (1 - ease_out(u * 0.7))
    hot = len(w['text']) >= 6 or idx == len(ln['words']) - 1
    a = clamp(u * 3) * (1 - clamp((t - min(nxt, w['end'] + 0.9) + 0.12) / 0.12))
    c.save(); c.translate(x.W / 2, x.cy + fnt.getSize() * 0.33); c.rotate(rot); c.scale(sc, sc)
    col = x.col('accent' if hot else 'text', a)
    if hot:
        c.drawString(txt, -tw / 2, 0, fnt, paint(x.col('accent', a * 0.5), blur=fnt.getSize() * 0.08))
    c.drawString(txt, -tw / 2, 0, fnt, paint(col))
    c.restore(); c.restore()


def neon_grid(x):
    c = x.c; hy = x.H * 0.64; vx = x.W / 2
    a = 0.25 + 0.35 * x.low
    p = paint(x.col('accent', a), stroke=x.U * 0.002, cap=False)
    for k in range(-14, 15):
        c.drawLine(vx + k * x.W * 0.004, hy, vx + k * x.W * 0.11, x.H, p)
    sp = (x.t / x.f.beat) % 1.0
    for k in range(12):
        z = (k + sp) / 12.0
        y = hy + (x.H - hy) * z ** 2.2
        c.drawLine(0, y, x.W, y, paint(x.col('accent', a * z), stroke=x.U * 0.002, cap=False))
    c.drawLine(0, hy, x.W, hy, paint(x.col('accent', 0.8), stroke=x.U * 0.003, blur=x.U * 0.004))


def t_neon(x):
    neon_grid(x)
    st = x.line_at(lead=0.25, hold=1.2)
    if not st: return
    i, enter, exit_ = st
    ln = x.f.lines[i]
    font, items, nr = x.fit_font(x.s['font'], x.U * 0.11 * x.size, ln['words'], x.s['case'], x.W * 0.82)
    lh = font.getSize() * 1.15
    top = (x.H * 0.36 if x.s.get('position') == 'center' else x.cy) - (nr - 1) * lh / 2
    c = x.c; la = (1 - exit_) * enter
    sw = font.getSize() * 0.035
    for it in items:
        w = it['w']; y = top + it['row'] * lh + font.getSize() * 0.35
        lt = x.t - w['start']
        on = 0.12 if lt < 0 else 1.0
        if 0 <= lt < 0.2:
            on = (1.0, 0.1, 0.9, 0.3, 1.0)[int(lt / 0.04)]
        a = on * la
        blob = skia.TextBlob.MakeFromString(it['txt'], font)
        c.drawTextBlob(blob, it['x'], y, paint(x.col('accent', a * 0.75), stroke=sw * 3, blur=font.getSize() * 0.06))
        c.drawTextBlob(blob, it['x'], y, paint(x.col('accent', a), stroke=sw))
        c.drawTextBlob(blob, it['x'], y, paint(x.col('text', a * 0.9), stroke=sw * 0.35))


def soft_blobs(x):
    c = x.c
    for k in range(3):
        px = x.W * (0.5 + 0.35 * math.sin(x.t * 0.07 + k * 2.1)); py = x.H * (0.5 + 0.3 * math.cos(x.t * 0.05 + k * 1.7))
        r = x.U * (0.55 + 0.1 * k)
        g = skia.GradientShader.MakeRadial(skia.Point(px, py), r, [x.col('accent', 0.10 + 0.05 * x.low), x.col('accent', 0)], [0, 1])
        c.drawCircle(px, py, r, skia.Paint(Shader=g))


def t_karaoke(x):
    soft_blobs(x)
    st = x.line_at(lead=0.35, hold=1.0, gap=0.0)
    if not st: return
    i, enter, exit_ = st
    ln = x.f.lines[i]
    font, items, nr = x.fit_font(x.s['font'], x.U * 0.075 * x.size, ln['words'], x.s['case'], x.W * 0.84)
    lh = font.getSize() * 1.3
    top = x.cy - (nr - 1) * lh / 2 - x.U * 0.03
    c = x.c; la = ease_out(enter) * (1 - exit_)
    c.save(); c.translate(0, (1 - ease_out(enter)) * x.U * 0.04)
    for it in items:
        w = it['w']; y = top + it['row'] * lh
        c.drawString(it['txt'], it['x'], y, font, paint(x.col('text', 0.42 * la)))
        p = prog(w, x.t)
        if p > 0:
            c.save(); c.clipRect(skia.Rect(it['x'] - 2, y - font.getSize(), it['x'] + it['wd'] * p + 1, y + font.getSize() * 0.4))
            c.drawString(it['txt'], it['x'], y, font, paint(x.col('accent', la)))
            c.restore()
    c.restore()
    if i + 1 < len(x.f.lines):
        nf = mkfont(x.s['font'], font.getSize() * 0.6)
        nitems, nn = x.layout(x.f.lines[i + 1]['words'], nf, x.W * 0.84, x.s['case'])
        y0 = top + nr * lh + x.U * 0.04
        for it in nitems:
            c.drawString(it['txt'], it['x'], y0 + it['row'] * nf.getSize() * 1.25, nf, paint(x.col('text', 0.3 * la)))


def t_typewriter(x):
    c, f, t = x.c, x.f, x.t
    for k in range(260):
        px, py = hrand(k, 11) * x.W, hrand(k, 12) * x.H
        c.drawCircle(px, py, x.U * 0.0009, paint(x.col('text', 0.06)))
    cur = -1
    for k, ln in enumerate(f.lines):
        if ln['start'] - 0.05 <= t: cur = k
    if cur < 0: return
    if t > f.lines[cur]['end'] + 2.5 and cur == len(f.lines) - 1: return
    font = mkfont(x.s['font'], x.U * 0.055 * x.size)
    lh = font.getSize() * 1.6
    maxw = x.W * 0.8; left = x.W * 0.1
    rows = []
    for k in range(max(0, cur - 4), cur + 1):
        ws = f.lines[k]['words']; txt_rows = [[]]; wx = 0
        for w in ws:
            tw = font.measureText(disp(w['text'], x.s['case']) + ' ')
            if wx + tw > maxw and txt_rows[-1]: txt_rows.append([]); wx = 0
            txt_rows[-1].append(w); wx += tw
        rows += [(k, r) for r in txt_rows]
    base_y = x.cy + lh
    n = len(rows); cursor = None
    for r_i, (k, ws) in enumerate(rows):
        y = base_y - (n - 1 - r_i) * lh
        age = cur - k
        a = 1.0 if age == 0 else max(0.0, 0.6 - 0.15 * age)
        px = left
        for w in ws:
            txt = disp(w['text'], x.s['case'])
            if k == cur:
                p = prog(w, t); nch = len(txt) if p >= 1 else (int(p * len(txt)) + (1 if t >= w['start'] else 0))
                txt_show = txt[:max(0, nch)]
            else:
                txt_show = txt
            for j, ch in enumerate(txt_show):
                jy = (hrand(hash((k, w['wi'], j)) & 0xFFFF, 13) - 0.5) * x.U * 0.002
                c.drawString(ch, px, y + jy, font, paint(x.col('text', a)))
                px += font.measureText(ch)
            if k == cur and len(txt_show) < len(txt):
                cursor = (px, y); break
            if k == cur and t >= w['start']:
                cursor = (px, y)
            px += font.measureText(' ' if k != cur or len(txt_show) == len(txt) else '')
            if k == cur and len(txt_show) < len(txt): break
            px += font.measureText(' ') * 0 if True else 0
    if cursor and (t * 2) % 1 < 0.6:
        cxp, cyp = cursor
        c.drawRect(skia.Rect(cxp + font.getSize() * 0.05, cyp - font.getSize() * 0.75, cxp + font.getSize() * 0.6, cyp + font.getSize() * 0.12), paint(x.col('accent', 0.85)))


def t_cinematic(x):
    c = x.c
    for k in range(3):
        px = x.W * (0.2 + 0.6 * hrand(k, 21) + 0.1 * math.sin(x.t * 0.05 + k)); py = x.H * (0.3 + 0.4 * hrand(k, 22))
        r = x.U * (0.6 + 0.3 * hrand(k, 23))
        g = skia.GradientShader.MakeRadial(skia.Point(px, py), r, [x.col('accent', 0.12 * (0.6 + 0.5 * x.e)), x.col('accent', 0)], [0, 1])
        c.drawCircle(px, py, r, skia.Paint(Shader=g))
    st = x.line_at(lead=0.2, hold=1.8, gap=0.2)
    if st:
        i, enter, exit_ = st
        ln = x.f.lines[i]
        font, items, nr = x.fit_font(x.s['font'], x.U * 0.075 * x.size, ln['words'], x.s['case'], x.W * 0.76)
        lh = font.getSize() * 1.3
        cy = x.H * 0.74 if x.s.get('position') == 'center' and x.W > x.H else x.cy
        top = cy - (nr - 1) * lh / 2
        la = 1 - smooth(exit_)
        for it in items:
            w = it['w']; u = clamp((x.t - w['start'] + 0.1) / 0.45)
            if u <= 0: continue
            bl = (1 - ease_out(u)) * font.getSize() * 0.12
            p = paint(x.col('text', ease_out(u) * la))
            if bl > 0.5: p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, bl))
            c.drawString(it['txt'], it['x'], top + it['row'] * lh - (1 - ease_out(u)) * x.U * 0.01, font, p)
    if x.W > x.H * 1.5 and x.W / x.H < 2.0:
        bh = x.H * 0.11
        c.drawRect(skia.Rect(0, 0, x.W, bh), paint(hexc('#000000')))
        c.drawRect(skia.Rect(0, x.H - bh, x.W, x.H), paint(hexc('#000000')))


def t_social(x):
    f, t = x.f, x.t
    cur = None
    for k, w in enumerate(f.words):
        if w['start'] - 0.05 <= t: cur = k
        else: break
    if cur is None: return
    w = f.words[cur]
    ln = next(L for L in f.lines if w in L['words'])
    if t > ln['end'] + 0.7: return
    font = mkfont(x.s['font'], x.U * (0.11 if x.H > x.W else 0.085) * x.size)
    maxw = x.W * 0.86
    groups, g, gw = [], [], 0
    for lw in ln['words']:
        ww = font.measureText(disp(lw['text'], x.s['case']) + ' ')
        if g and (len(g) >= 3 or gw + ww > maxw): groups.append(g); g, gw = [], 0
        g.append(lw); gw += ww
    if g: groups.append(g)
    grp = next(gg for gg in groups if w in gg)
    items, nr = x.layout(grp, font, maxw, x.s['case'], space=font.getSize() * 0.32)
    lh = font.getSize() * 1.35
    top = x.cy - (nr - 1) * lh / 2 + font.getSize() * 0.35
    c = x.c
    for it in items:
        ww = it['w']; u = (t - ww['start'] + 0.05) / 0.16
        if u <= 0: continue
        sc = 0.6 + 0.4 * ease_out_back(u)
        y = top + it['row'] * lh
        active = ww is w and t <= ww['end'] + 0.25
        c.save(); cxp = it['x'] + it['wd'] / 2; c.translate(cxp, y - font.getSize() * 0.35); c.scale(sc, sc); c.translate(-cxp, -(y - font.getSize() * 0.35))
        if active:
            pad = font.getSize() * 0.14
            r = skia.RRect.MakeRectXY(skia.Rect(it['x'] - pad, y - font.getSize() * 0.86, it['x'] + it['wd'] + pad, y + font.getSize() * 0.2), pad, pad)
            c.drawRRect(r, paint(x.col('accent', 1)))
            c.drawString(it['txt'], it['x'], y, font, paint(hexc('#111111')))
        else:
            c.drawString(it['txt'], it['x'], y, font, paint(hexc('#000000', 0.6), stroke=font.getSize() * 0.12))
            c.drawString(it['txt'], it['x'], y, font, paint(x.col('text', 1)))
        c.restore()


DRAW = dict(ink=t_ink, kinetic=t_kinetic, neon=t_neon, karaoke=t_karaoke, typewriter=t_typewriter, cinematic=t_cinematic, social=t_social)


# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, film, scale=1.0):
        self.film = film
        self.sc = scale
        self._mk()

    def _mk(self):
        f = self.film
        self.size = (f.W, f.H)
        self.pw = max(2, int(round(f.W * self.sc / 2)) * 2); self.ph = max(2, int(round(f.H * self.sc / 2)) * 2)
        self.surf = skia.Surface(self.pw, self.ph)
        rng = np.random.default_rng(5)
        self.grain = [np.clip(rng.normal(0, 4.0, (self.ph // 2, self.pw // 2)), -12, 12).astype(np.int16) for _ in range(4)]

    def frame(self, t, samples=None):
        f = self.film
        if self.size != (f.W, f.H):
            self._mk()
        c = self.surf.getCanvas()
        c.save(); c.resetMatrix(); c.scale(self.pw / f.W, self.ph / f.H)
        x = Ctx(f, t, c)
        draw_background(x)
        if f.s.get('particles'):
            draw_particles(x, color='text' if f.s['template'] == 'typewriter' else 'accent')
        vis = intro_outro(x)
        if vis > 0:
            DRAW.get(f.s['template'], t_ink)(x)
        if f.s.get('visualizer'):
            draw_visualizer(x)
        c.restore()
        arr = self.surf.makeImageSnapshot().toarray(colorType=skia.kBGRA_8888_ColorType)
        if f.s['template'] in ('cinematic', 'typewriter', 'ink'):
            g = cv2.resize(self.grain[int(t * 24) % 4], (self.pw, self.ph), interpolation=cv2.INTER_NEAREST)
            arr[..., :3] = np.clip(arr[..., :3].astype(np.int16) + g[..., None], 0, 255).astype(np.uint8)
        if _look is not None and _LOOK is not None:
            arr = _look.apply(arr, _LOOK)
        arr[..., 3] = 255
        return arr

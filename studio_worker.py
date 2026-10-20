"""ADx Studio persistent preview engine: builds the film once, then renders frames on request (one JSON line each).

    python3 studio_worker.py <project_dir> <grain>
stdin : {"t": 12.3, "scale": 0.15, "fast": true, "look": {...} | null, "out": "/path.jpg"}
stdout: {"ok": true, "ms": 41} | {"ok": false, "err": "..."}
"""
import json, os, sys, time, traceback

proj = os.path.abspath(sys.argv[1])
os.environ['KARAM_GRAIN'] = sys.argv[2] if len(sys.argv) > 2 else '1.0'
os.chdir(proj)
sys.path.insert(0, os.path.join(proj, 'src'))
import cv2  # noqa: E402
cv2.setNumThreads(2)
import main as M  # noqa: E402  (project's src/main.py)
import render as RM  # noqa: E402

F = M.build_film()
renderers = {}


def renderer(scale):
    key = round(scale, 3)
    if key not in renderers:
        if len(renderers) > 3:
            renderers.clear()
        renderers[key] = RM.Renderer(F, key)
    return renderers[key]


print(json.dumps(dict(ok=True, ready=True)), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        t0 = time.time()
        R = renderer(float(req.get('scale', 0.3)))
        if 'settings' in req and hasattr(F, 'set_settings'):
            F.set_settings(req.get('settings'))
        RM._LOOK = req.get('look')          # grade applied inside Renderer.post (studio hook)
        arr = R.frame(float(req['t']), samples=1 if req.get('fast') else None)
        out = req['out']
        if out.endswith('.png'):
            ok = cv2.imwrite(out, arr[..., :3])
        else:
            ok = cv2.imwrite(out, arr[..., :3], [cv2.IMWRITE_JPEG_QUALITY, int(req.get('q', 86))])
        print(json.dumps(dict(ok=bool(ok), ms=int((time.time() - t0) * 1000))), flush=True)
    except Exception as e:
        print(json.dumps(dict(ok=False, err='%s: %s' % (type(e).__name__, e),
                              tb=traceback.format_exc()[-600:])), flush=True)

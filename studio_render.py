"""Render one contiguous segment of frames to a video-only file (used in parallel by Karam Studio).

    python3 studio_render.py <project_dir> <f0> <f1> <fps> <scale> <fast 0|1> <encoder> <bitrate|crf> <out.mp4> <progress_file> <look.json|->
Frame k is at time k / fps (absolute frame numbers on the film's timeline).
"""
import json, os, subprocess, sys, time

proj, f0, f1, fps, scale, fast, enc, rate, out, prog, lookp = sys.argv[1:12]
f0, f1, fps, scale, fast = int(f0), int(f1), int(fps), float(scale), fast == '1'
os.environ['KARAM_FPS'] = str(fps)
os.environ['KARAM_GRAIN'] = '0.5' if scale < 0.4 else '1.0'
os.chdir(proj)
sys.path.insert(0, os.path.join(proj, 'src'))
import main as M  # noqa: E402
import render as RM  # noqa: E402

if lookp != '-':
    try:
        RM._LOOK = json.load(open(lookp))
    except Exception:
        pass
F = M.build_film()
R = RM.Renderer(F, scale)
vf = 'crop=trunc(iw/2)*2:trunc(ih/2)*2'
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', '%dx%d' % (R.pw, R.ph),
       '-r', str(fps), '-i', '-', '-vf', vf]
if 'videotoolbox' in enc:
    cmd += ['-c:v', enc, '-b:v', rate, '-pix_fmt', 'nv12', '-allow_sw', '1']
else:
    preset = 'medium' if scale >= 0.99 else 'veryfast'
    cmd += ['-c:v', 'libx264', '-preset', preset, '-crf', rate, '-pix_fmt', 'yuv420p', '-threads', '2']
cmd += ['-an', out]
ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
t0 = time.time()
for k in range(f0, f1):
    ff.stdin.write(R.frame(k / fps, samples=1 if fast else None).tobytes())
    if (k - f0) % 6 == 0 or k == f1 - 1:
        with open(prog + '.tmp', 'w') as f:
            f.write(str(k - f0 + 1))
        os.replace(prog + '.tmp', prog)
ff.stdin.close()
sys.exit(ff.wait())

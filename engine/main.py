"""ADx Studio project entry point (template engine).

  python3 src/main.py still <t> [scale] [out.png]
  python3 src/main.py clip  <t0> <t1> [scale] [out.mp4] [fps]
"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R

ROOT = R.ROOT
AUDIO = os.path.join(ROOT, 'assets', 'karam.mp3')


def build_film():
    return R.Film()


def main():
    mode = sys.argv[1]
    F = build_film()
    if mode == 'still':
        t = float(sys.argv[2]); sc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ROOT, 'out', 'still_%.2f.png' % t)
        import cv2
        cv2.imwrite(out, R.Renderer(F, sc).frame(t)[..., :3]); print(out)
    elif mode == 'clip':
        t0, t1 = float(sys.argv[2]), float(sys.argv[3])
        sc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
        out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(ROOT, 'out', 'clip.mp4')
        fps = int(sys.argv[6]) if len(sys.argv) > 6 else 30
        Rr = R.Renderer(F, sc); n = int(round((t1 - t0) * fps))
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', '%dx%d' % (Rr.pw, Rr.ph),
               '-r', str(fps), '-i', '-']
        if os.path.isfile(AUDIO):
            cmd += ['-ss', '%.4f' % t0, '-t', '%.4f' % (n / fps), '-i', AUDIO, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '192k']
        cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-shortest', out]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k in range(n):
            ff.stdin.write(Rr.frame(t0 + k / fps).tobytes())
        ff.stdin.close(); ff.wait()


if __name__ == '__main__':
    main()

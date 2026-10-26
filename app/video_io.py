import os
import subprocess

import imageio
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg(*args):
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *map(str, args)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()}")


def write_frames(frames, path, fps):
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                           macro_block_size=1, ffmpeg_log_level="error")
    for f in frames:
        w.append_data(np.asarray(f.convert("RGB")))
    w.close()
    return path


def fit_clip(src, dst, duration, fps, out_fps, width, height, generated_seconds):
    """Trim (or slow down, if the model clip is too short) to exactly `duration` seconds."""
    stretch = max(1.0, duration / max(generated_seconds, 1e-3))
    vf = f"setpts={stretch:.5f}*PTS,fps={out_fps},scale={width}:{height}:flags=lanczos,setsar=1"
    ffmpeg("-i", src, "-vf", vf, "-t", f"{duration:.3f}", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", dst)
    return dst


def concat_with_audio(clips, audio, dst, start=0.0, duration=None):
    list_path = os.path.splitext(dst)[0] + "_clips.txt"
    with open(list_path, "w", encoding="utf-8") as fh:
        for c in clips:
            fh.write(f"file '{os.path.abspath(c).replace(os.sep, '/')}'\n")
    audio_opts = ["-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else [])
    ffmpeg("-f", "concat", "-safe", "0", "-i", list_path, *audio_opts, "-i", audio,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
           "-shortest", "-movflags", "+faststart", dst)
    return dst

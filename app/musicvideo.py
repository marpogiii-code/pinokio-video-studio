import os
import time

import numpy as np

from engine import ENGINE, MODELS, frames_for_seconds, snap_size
from video_io import concat_with_audio, fit_clip, write_frames

ENERGY_WORDS = {
    "low": "calm, slow gentle camera movement, soft dreamy lighting",
    "mid": "steady flowing camera movement, cinematic lighting",
    "high": "energetic, fast dynamic camera motion, vivid dramatic lighting",
}
PLAN_HEADERS = ["#", "start", "end", "energy", "prompt"]


def analyze_audio(path, start=0.0, max_duration=0.0):
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True, offset=float(start) or 0.0,
                         duration=float(max_duration) if max_duration else None)
    duration = len(y) / sr
    if duration < 0.5:
        raise ValueError("Audio is too short (or start offset is past the end).")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    return {
        "duration": float(duration),
        "tempo": float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0,
        "beats": [float(b) for b in np.atleast_1d(beats)],
        "rms": rms,
        "onset": onset,
        "times": times,
    }


def plan_cuts(duration, beats, target, max_len):
    """Cut points snapped to beats, each segment within [target/2, max_len]."""
    target = max(0.5, min(float(target), max_len))
    min_len = target * 0.5
    beats = [b for b in beats if 0 < b < duration]
    cuts, t = [0.0], 0.0
    while duration - t > min(target + min_len, max_len):
        hi = min(t + max_len, duration - min_len)
        cands = [b for b in beats if t + target * 0.6 <= b <= min(t + target * 1.5, hi)]
        nxt = min(cands, key=lambda b: abs(b - (t + target))) if cands else min(t + target, hi)
        cuts.append(nxt)
        t = nxt
    cuts.append(duration)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _energy_levels(analysis, segments):
    times, rms = analysis["times"], analysis["rms"]
    vals = []
    for s, e in segments:
        m = (times >= s) & (times < e)
        vals.append(float(rms[m].mean()) if m.any() else 0.0)
    if len(vals) < 3:
        return ["mid"] * len(vals)
    lo, hi = np.percentile(vals, [33, 67])
    return ["low" if v <= lo else "high" if v >= hi else "mid" for v in vals]


def build_plan(audio_path, model_key, scenes_text, style, clip_seconds, start=0.0, max_duration=0.0,
               energy_prompts=True):
    if not audio_path:
        raise ValueError("Upload an audio file first.")
    spec = MODELS[model_key]
    analysis = analyze_audio(audio_path, start, max_duration)
    max_len = (spec.max_frames - 1) / spec.fps
    segments = plan_cuts(analysis["duration"], analysis["beats"], clip_seconds, max_len)
    levels = _energy_levels(analysis, segments)
    scenes = [s.strip() for s in (scenes_text or "").splitlines() if s.strip()] or ["a cinematic music video scene"]
    rows = []
    for i, ((s, e), lvl) in enumerate(zip(segments, levels)):
        scene = scenes[min(int(i * len(scenes) / len(segments)), len(scenes) - 1)]
        parts = [scene, style.strip() if style else "", ENERGY_WORDS[lvl] if energy_prompts else ""]
        rows.append([i + 1, round(s, 3), round(e, 3), lvl, ", ".join(p for p in parts if p)])
    info = (f"Duration {analysis['duration']:.1f}s | Tempo ~{analysis['tempo']:.0f} BPM | "
            f"{len(analysis['beats'])} beats | {len(rows)} clips")
    return rows, info


def _rows(plan):
    if hasattr(plan, "values"):
        plan = plan.values.tolist()
    rows = []
    for r in plan or []:
        if len(r) < 5 or str(r[4]).strip() == "":
            continue
        rows.append((float(r[1]), float(r[2]), str(r[4]).strip()))
    if not rows:
        raise ValueError("The scene plan is empty. Click 'Analyze audio' first.")
    return rows


def render(audio_path, plan, model_key, negative, start_image, continuity, width, height, steps, guidance,
           seed, low_vram, start=0.0, max_duration=0.0, out_root="outputs", progress=None):
    spec = MODELS[model_key]
    rows = _rows(plan)
    if (start_image is not None or continuity) and not spec.i2v:
        raise ValueError(f"{spec.label} is text-only; disable 'start image' / 'continuity' or pick another model.")
    width, height = snap_size(spec, width), snap_size(spec, height)
    out_dir = os.path.join(out_root, time.strftime("musicvideo_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    base_seed = int(seed) if seed is not None and int(seed) >= 0 else int(time.time()) % (2**31)
    clips, prev_last = [], None
    total_dur = rows[-1][1] - rows[0][0]

    for i, (s, e, prompt) in enumerate(rows):
        dur = e - s
        init = start_image if i == 0 else (prev_last if continuity else None)
        n = frames_for_seconds(spec, dur)

        def on_step(k, total, i=i):
            if progress:
                progress((i + k / total) / len(rows), desc=f"Clip {i + 1}/{len(rows)} step {k}/{total}")

        frames, fps, _ = ENGINE.generate(model_key, prompt, negative, init, width, height, n, steps,
                                         guidance, base_seed + i, low_vram, on_step)
        prev_last = frames[-1]
        raw = write_frames(frames, os.path.join(out_dir, f"clip_{i + 1:03d}_raw.mp4"), fps)
        clips.append(fit_clip(raw, os.path.join(out_dir, f"clip_{i + 1:03d}.mp4"), dur, fps, spec.fps,
                              width, height, len(frames) / fps))

    if progress:
        progress(1.0, desc="Muxing audio")
    final = os.path.join(out_dir, "music_video.mp4")
    concat_with_audio(clips, audio_path, final, start=float(start or 0.0) + rows[0][0],
                      duration=total_dur if (max_duration or rows[0][0] > 0) else None)
    return final, clips

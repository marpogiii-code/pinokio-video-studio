import argparse
import os
import time

import gradio as gr

import musicvideo
from engine import ENGINE, MODELS, NEGATIVE_DEFAULT, model_choices
from video_io import write_frames

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
DEFAULT_MODEL = "ltx"


def _defaults(key):
    m = MODELS[key]
    return (gr.update(value=m.width), gr.update(value=m.height),
            gr.update(value=m.default_frames, maximum=m.max_frames, step=m.frame_step),
            gr.update(value=m.steps), gr.update(value=m.guidance))


def _size_defaults(key):
    m = MODELS[key]
    return gr.update(value=m.width), gr.update(value=m.height), gr.update(value=m.steps), gr.update(value=m.guidance)


def _run(model, prompt, negative, image, width, height, frames, steps, guidance, seed, low_vram, prefix, progress):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    try:
        out, fps, used_seed = ENGINE.generate(
            model, prompt, negative, image, width, height, frames, steps, guidance, seed, low_vram,
            on_step=lambda k, n: progress(k / n, desc=f"Step {k}/{n}"))
    except ValueError as e:
        raise gr.Error(str(e))
    path = os.path.join(OUT, time.strftime(f"{prefix}_%Y%m%d_%H%M%S.mp4"))
    write_frames(out, path, fps)
    return path, f"Seed {used_seed} | {len(out)} frames @ {fps} fps | {out[0].width}x{out[0].height}"


def t2v(model, prompt, negative, width, height, frames, steps, guidance, seed, low_vram, progress=gr.Progress()):
    return _run(model, prompt, negative, None, width, height, frames, steps, guidance, seed, low_vram, "t2v", progress)


def i2v(model, image, prompt, negative, width, height, frames, steps, guidance, seed, low_vram,
        progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a start image.")
    return _run(model, prompt, negative, image, width, height, frames, steps, guidance, seed, low_vram, "i2v",
                progress)


def mv_analyze(audio, model, scenes, style, clip_seconds, start, max_dur, energy):
    try:
        rows, info = musicvideo.build_plan(audio, model, scenes, style, clip_seconds, start, max_dur, energy)
    except ValueError as e:
        raise gr.Error(str(e))
    return rows, info


def mv_render(audio, plan, model, negative, start_image, continuity, width, height, steps, guidance, seed,
              low_vram, start, max_dur, progress=gr.Progress()):
    if not audio:
        raise gr.Error("Upload an audio file first.")
    try:
        final, clips = musicvideo.render(audio, plan, model, negative, start_image, continuity, width, height,
                                         steps, guidance, seed, low_vram, start, max_dur, OUT, progress)
    except ValueError as e:
        raise gr.Error(str(e))
    return final, f"Saved {final} ({len(clips)} clips)"


def gen_controls(key, frames=True):
    m = MODELS[key]
    with gr.Accordion("Settings", open=False):
        negative = gr.Textbox(label="Negative prompt", value=NEGATIVE_DEFAULT, lines=2)
        with gr.Row():
            width = gr.Slider(256, 1536, value=m.width, step=32, label="Width")
            height = gr.Slider(256, 1536, value=m.height, step=32, label="Height")
        nframes = gr.Slider(9, m.max_frames, value=m.default_frames, step=m.frame_step,
                            label="Frames", visible=frames)
        with gr.Row():
            steps = gr.Slider(4, 80, value=m.steps, step=1, label="Steps")
            guidance = gr.Slider(1.0, 12.0, value=m.guidance, step=0.5, label="Guidance (CFG)")
        with gr.Row():
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            low_vram = gr.Checkbox(value=False, label="Low VRAM (sequential offload, much slower)")
    return negative, width, height, nframes, steps, guidance, seed, low_vram


with gr.Blocks(title="Video Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Video Studio\nText-to-video, image-to-video and audio-driven music videos. "
                "Models download from Hugging Face on first use.")
    with gr.Tabs():
        with gr.Tab("Text to Video"):
            with gr.Row():
                with gr.Column():
                    t_model = gr.Dropdown(model_choices(), value=DEFAULT_MODEL, label="Model")
                    t_prompt = gr.Textbox(label="Prompt", lines=4,
                                          placeholder="A red fox running through a snowy forest at sunrise, cinematic")
                    t_ctrl = gen_controls(DEFAULT_MODEL)
                    t_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    t_out = gr.Video(label="Result")
                    t_info = gr.Markdown()
            t_model.change(_defaults, t_model, [t_ctrl[1], t_ctrl[2], t_ctrl[3], t_ctrl[4], t_ctrl[5]])
            t_btn.click(t2v, [t_model, t_prompt, *t_ctrl], [t_out, t_info])

        with gr.Tab("Image to Video"):
            i2v_models = [c for c in model_choices() if MODELS[c[1]].i2v]
            with gr.Row():
                with gr.Column():
                    i_model = gr.Dropdown(i2v_models, value=DEFAULT_MODEL, label="Model")
                    i_image = gr.Image(type="pil", label="Start image")
                    i_prompt = gr.Textbox(label="Prompt (describe the motion)", lines=3,
                                          placeholder="The camera slowly pushes in as her hair blows in the wind")
                    i_ctrl = gen_controls(DEFAULT_MODEL)
                    i_btn = gr.Button("Animate", variant="primary")
                with gr.Column():
                    i_out = gr.Video(label="Result")
                    i_info = gr.Markdown()
            i_model.change(_defaults, i_model, [i_ctrl[1], i_ctrl[2], i_ctrl[3], i_ctrl[4], i_ctrl[5]])
            i_btn.click(i2v, [i_model, i_image, i_prompt, *i_ctrl], [i_out, i_info])

        with gr.Tab("Music Video"):
            gr.Markdown("1. Upload a song and describe the scenes. 2. **Analyze audio** to split it into "
                        "beat-aligned clips (edit any prompt in the table). 3. **Render** — clips are generated, "
                        "cut to the beat and muxed with your audio.")
            with gr.Row():
                with gr.Column():
                    m_audio = gr.Audio(type="filepath", label="Audio file (mp3/wav/flac/m4a)")
                    m_model = gr.Dropdown(model_choices(), value=DEFAULT_MODEL, label="Model")
                    m_scenes = gr.Textbox(
                        label="Scenes / storyline (one per line, spread across the song in order)", lines=5,
                        placeholder="a lone singer on a rooftop at night, city lights\n"
                                    "neon-lit streets in the rain, crowds dancing\n"
                                    "sunrise over the ocean, the singer walking on the beach")
                    m_style = gr.Textbox(label="Global style (appended to every clip)",
                                         value="cinematic music video, 35mm film, high detail")
                    with gr.Row():
                        m_clip = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Target clip length (s)")
                        m_energy = gr.Checkbox(value=True, label="Adapt prompts to song energy")
                    with gr.Row():
                        m_start = gr.Number(value=0, label="Audio start (s)")
                        m_maxdur = gr.Number(value=0, label="Max duration (s, 0 = full song)")
                    m_image = gr.Image(type="pil", label="Optional start image (first clip uses image-to-video)")
                    m_cont = gr.Checkbox(value=False,
                                         label="Continuity: start each clip from the previous clip's last frame")
                    with gr.Accordion("Settings", open=False):
                        m_neg = gr.Textbox(label="Negative prompt", value=NEGATIVE_DEFAULT, lines=2)
                        with gr.Row():
                            m_w = gr.Slider(256, 1536, value=MODELS[DEFAULT_MODEL].width, step=32, label="Width")
                            m_h = gr.Slider(256, 1536, value=MODELS[DEFAULT_MODEL].height, step=32, label="Height")
                        with gr.Row():
                            m_steps = gr.Slider(4, 80, value=MODELS[DEFAULT_MODEL].steps, step=1, label="Steps")
                            m_cfg = gr.Slider(1.0, 12.0, value=MODELS[DEFAULT_MODEL].guidance, step=0.5,
                                              label="Guidance (CFG)")
                        with gr.Row():
                            m_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                            m_low = gr.Checkbox(value=False, label="Low VRAM")
                    m_an = gr.Button("1. Analyze audio")
                with gr.Column():
                    m_info = gr.Markdown()
                    m_plan = gr.Dataframe(headers=musicvideo.PLAN_HEADERS,
                                          datatype=["number", "number", "number", "str", "str"],
                                          interactive=True, wrap=True, label="Scene plan (prompts are editable)")
                    m_btn = gr.Button("2. Render music video", variant="primary")
                    m_out = gr.Video(label="Music video")
                    m_status = gr.Markdown()
            m_model.change(_size_defaults, m_model, [m_w, m_h, m_steps, m_cfg])
            m_an.click(mv_analyze, [m_audio, m_model, m_scenes, m_style, m_clip, m_start, m_maxdur, m_energy],
                       [m_plan, m_info])
            m_btn.click(mv_render, [m_audio, m_plan, m_model, m_neg, m_image, m_cont, m_w, m_h, m_steps, m_cfg,
                                    m_seed, m_low, m_start, m_maxdur], [m_out, m_status])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port,
                                                   allowed_paths=[OUT])

import gc
import os
import random
import threading
from dataclasses import dataclass

from PIL import Image

FAKE = os.environ.get("VIDEO_STUDIO_FAKE") == "1"

NEGATIVE_DEFAULT = (
    "worst quality, low quality, blurry, jittery, distorted, deformed, static, "
    "watermark, text, subtitles, overexposed, ugly, extra limbs, bad anatomy"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    family: str
    fps: int
    frame_step: int
    max_frames: int
    default_frames: int
    size_mult: int
    t2v: bool
    i2v: bool
    width: int
    height: int
    steps: int
    guidance: float


MODELS = {
    m.key: m
    for m in [
        ModelSpec("ltx", "LTX-Video (fast, 12GB+ VRAM)", "Lightricks/LTX-Video", "ltx",
                  24, 8, 257, 97, 32, True, True, 768, 512, 40, 3.0),
        ModelSpec("wan22_5b", "Wan 2.2 TI2V 5B (quality, 16GB+ VRAM)", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan",
                  24, 4, 121, 121, 32, True, True, 1280, 704, 40, 5.0),
        ModelSpec("wan21_1_3b", "Wan 2.1 T2V 1.3B (low VRAM 8GB+, text only)", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan",
                  16, 4, 81, 81, 16, True, False, 832, 480, 30, 5.0),
    ]
}


def model_choices():
    return [(m.label, m.key) for m in MODELS.values()]


def snap_frames(spec, n):
    n = max(1, min(int(n), spec.max_frames))
    return ((n - 1) // spec.frame_step) * spec.frame_step + 1


def frames_for_seconds(spec, seconds):
    """Smallest valid frame count covering `seconds`, capped at the model max."""
    need = max(2, int(seconds * spec.fps + 0.999) + 1)
    k = -(-(need - 1) // spec.frame_step)
    return min(k * spec.frame_step + 1, snap_frames(spec, spec.max_frames))


def snap_size(spec, v):
    return max(spec.size_mult, int(v) // spec.size_mult * spec.size_mult)


def fit_image(img, width, height):
    """Resize + center-crop to exactly width x height."""
    img = img.convert("RGB")
    scale = max(width / img.width, height / img.height)
    w, h = round(img.width * scale), round(img.height * scale)
    img = img.resize((w, h), Image.LANCZOS)
    left, top = (w - width) // 2, (h - height) // 2
    return img.crop((left, top, left + width, top + height))


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.pipe = None
        self.loaded = None

    def unload(self):
        self.pipe = None
        self.loaded = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load(self, spec, mode, low_vram):
        sig = (spec.key, mode, low_vram)
        if self.loaded == sig:
            return self.pipe
        self.unload()
        import torch
        import diffusers

        dtype = torch.bfloat16
        if spec.family == "ltx":
            cls = diffusers.LTXImageToVideoPipeline if mode == "i2v" else diffusers.LTXPipeline
            pipe = cls.from_pretrained(spec.repo, torch_dtype=dtype)
        else:
            vae = diffusers.AutoencoderKLWan.from_pretrained(spec.repo, subfolder="vae", torch_dtype=torch.float32)
            cls = diffusers.WanImageToVideoPipeline if mode == "i2v" else diffusers.WanPipeline
            pipe = cls.from_pretrained(spec.repo, vae=vae, torch_dtype=dtype)

        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if not torch.cuda.is_available():
            pipe.to("cpu")
        elif low_vram:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        self.pipe, self.loaded = pipe, sig
        return pipe

    def generate(self, model_key, prompt, negative="", image=None, width=None, height=None,
                 num_frames=None, steps=None, guidance=None, seed=-1, low_vram=False, on_step=None):
        """Returns (frames: list[PIL.Image], fps, seed)."""
        spec = MODELS[model_key]
        mode = "i2v" if image is not None else "t2v"
        if mode == "i2v" and not spec.i2v:
            raise ValueError(f"{spec.label} does not support image-to-video. Pick another model.")
        width = snap_size(spec, width or spec.width)
        height = snap_size(spec, height or spec.height)
        num_frames = snap_frames(spec, num_frames or spec.default_frames)
        steps = int(steps or spec.steps)
        guidance = float(spec.guidance if guidance is None else guidance)
        seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**31 - 1)
        if image is not None:
            image = fit_image(image, width, height)

        with self.lock:
            if FAKE:
                return _fake_frames(prompt, image, width, height, num_frames, steps, seed, on_step), spec.fps, seed
            import torch

            pipe = self._load(spec, mode, bool(low_vram))
            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative or NEGATIVE_DEFAULT,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator("cpu").manual_seed(seed),
            )
            if image is not None:
                kwargs["image"] = image
            if spec.family == "ltx":
                kwargs.update(decode_timestep=0.05, decode_noise_scale=0.025)
            if on_step:
                def _cb(_pipe, i, _t, cb_kwargs):
                    on_step(i + 1, steps)
                    return cb_kwargs
                kwargs["callback_on_step_end"] = _cb
            with torch.inference_mode():
                frames = pipe(**kwargs).frames[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        frames = [f if isinstance(f, Image.Image) else Image.fromarray(_to_uint8(f)) for f in frames]
        return frames, spec.fps, seed


def _to_uint8(arr):
    import numpy as np
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    return arr


def _fake_frames(prompt, image, width, height, num_frames, steps, seed, on_step):
    """CPU stand-in used for UI/pipeline testing without a GPU (VIDEO_STUDIO_FAKE=1)."""
    import numpy as np
    from PIL import ImageDraw

    for i in range(steps):
        if on_step:
            on_step(i + 1, steps)
    rng = np.random.default_rng(seed)
    base = np.asarray(image, dtype=np.float32) if image is not None else \
        np.tile(rng.uniform(40, 215, 3), (height, width, 1)).astype(np.float32)
    frames = []
    for f in range(num_frames):
        shift = int(f * width / max(num_frames, 1))
        arr = np.roll(base, shift, axis=1).astype(np.uint8)
        img = Image.fromarray(arr)
        ImageDraw.Draw(img).text((10, 10), f"{prompt[:60]} | f{f}", fill=(255, 255, 255))
        frames.append(img)
    return frames


ENGINE = Engine()

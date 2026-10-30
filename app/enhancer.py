import json
import os
import re
import threading
import urllib.request

FAKE = os.environ.get("VIDEO_STUDIO_FAKE") == "1"
LOCAL_MODEL = os.environ.get("VIDEO_STUDIO_ENHANCER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
BACKENDS = ["Local (Qwen2.5 1.5B)", "Ollama", "OpenAI-compatible"]

SYSTEM = (
    "You are an expert prompt writer for AI text-to-video and image-to-video models. "
    "Rewrite the user's idea into one vivid, concrete video prompt of 60-120 words. "
    "Describe in order: the main subject and its appearance, the action and motion over time, "
    "the setting and background, camera movement and framing, lighting, colors and mood, and visual style. "
    "Keep every element the user asked for; do not add text, logos or subtitles. "
    "Write a single paragraph in plain present tense. Output only the prompt, no preamble, quotes or lists."
)
I2V_HINT = (
    " The video starts from a provided still image, so focus on motion, camera movement and how the "
    "scene evolves rather than re-describing static details."
)
MV_HINT = " This is one shot of a music video; keep it a single continuous shot."


class Enhancer:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.tok = None

    def _local(self, messages, max_new_tokens=220):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        if self.model is None:
            self.tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL, torch_dtype=torch.bfloat16 if cuda else torch.float32)
        device = "cuda" if cuda else "cpu"
        self.model.to(device)
        try:
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                          do_sample=True, temperature=0.7, top_p=0.9,
                                          pad_token_id=self.tok.eos_token_id)
            return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        finally:
            if cuda:
                self.model.to("cpu")
                torch.cuda.empty_cache()

    def enhance(self, prompt, mode="t2v", backend=BACKENDS[0], url="", model="", api_key=""):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Enter a prompt to enhance.")
        system = SYSTEM + (I2V_HINT if mode == "i2v" else "") + (MV_HINT if mode == "mv" else "")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        with self.lock:
            if FAKE:
                text = f"{prompt}, cinematic shot, smooth camera movement, detailed lighting"
            elif backend == "Ollama":
                text = _ollama(messages, url or "http://127.0.0.1:11434", model or "llama3.2")
            elif backend == "OpenAI-compatible":
                text = _openai(messages, url or "https://api.openai.com/v1", model or "gpt-4o-mini", api_key)
            else:
                text = self._local(messages)
        return clean(text) or prompt


def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#`]+", "", text)
    text = re.sub(r"^(here is|here's|enhanced prompt|prompt)[^:\n]*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\w[\w ]{0,24}:\s+", "", text)
    return " ".join(text.strip().strip('"').split())


def _post(url, payload, headers=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())
    except OSError as e:
        raise ValueError(f"Prompt enhancer request to {url} failed: {e}") from e


def _ollama(messages, url, model):
    data = _post(url.rstrip("/") + "/api/chat",
                 {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.7}})
    return data.get("message", {}).get("content", "")


def _openai(messages, url, model, api_key):
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    data = _post(url.rstrip("/") + "/chat/completions",
                 {"model": model, "messages": messages, "temperature": 0.7},
                 {"Authorization": f"Bearer {key}"} if key else None)
    return data["choices"][0]["message"]["content"]


ENHANCER = Enhancer()

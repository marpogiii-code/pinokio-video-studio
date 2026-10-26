module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: { PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True" },
        message: ["python app.py --port {{port}}"],
        on: [{ event: "/(http:\/\/[0-9.:]+)/", done: true }],
      },
    },
    {
      method: "local.set",
      params: { url: "{{input.event[1]}}" },
    },
  ],
}

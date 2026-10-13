# ❓ FAQ

### Is it really compatible with OpenAI's API?

For chat and text completions, yes: same routes, same request and response shapes, same streaming
format, same error format, so the official SDKs work unchanged. What it doesn't do (tools, logprobs,
JSON schemas, `n` > 1, images) it refuses with a clear `400` instead of pretending. The full list
is in [Using the API](api.md).

### Which models can I run?

Any GGUF model that llama.cpp supports, in any quantisation: Llama, Mistral, Qwen, Gemma, Phi,
DeepSeek, Command-R and many more. One model per server; run several servers for several models.

### How is this different from llama.cpp's `llama-server` or Ollama?

All three run GGUF models with llama.cpp. LiSa's differences:

- **The sandbox.** Neither of the others sandboxes the code that parses the model. LiSa never opens
  the model in the network-facing process, and the process that does is locked down.
- **Built for exposure.** Hashed API keys, per-key rate limits, a fair bounded queue, mutual TLS,
  readiness probes, Prometheus metrics and graceful drain come built in.
- **Faster under concurrent load** than `llama-server` in [our benchmark](performance.md), sandbox
  included.
- **Smaller scope.** No model downloads, no model switching, no web UI, no tool calling (yet).
  It's a server, not a model manager.

### Does the sandbox slow it down?

Not measurably. A single sandboxed request runs as fast as `llama-server` in-process, and under
load LiSa is faster. See [Performance](performance.md).

### Do I need a GPU?

No. LiSa is fast on CPUs, especially with small quantised models. NVIDIA GPUs are supported through
the CUDA image or a CUDA build, but GPU mode runs without the sandbox.

### Can I use it on macOS or Windows?

Not natively: the sandbox uses Linux kernel features. The container image runs in Docker Desktop
(the Linux VM inside it does the work), on Apple silicon too.

### Is my data sent anywhere?

No. LiSa makes no outbound connections at all: no telemetry, no update checks, no model
downloads. With the sandbox on, the process that sees your prompts and the model can't even open a
network connection.

### How many users can it serve?

As many as you give it slots and memory for. Each of the `parallel_slots` generates at the same
time, `max_queued_requests` more wait their turn, and the rest are told to retry. Throughput grows
with concurrency up to roughly your core count; past that, add instances behind a load balancer.

### Can I run several models?

One per process. Run one LiSa per model (on different ports, or as different Compose services) and
route by URL in your proxy.

### Why does it refuse to start on `0.0.0.0` with my key?

Because the key is the placeholder `changeme` or shorter than 16 characters, and that's how servers
end up open to the internet. Use `openssl rand -hex 24`.

### Where does the name come from?

**Li**ghtweight **Sa**ndboxed LLM server. Also, every lion-sized sandbox deserves a name.

### How do I report a security problem?

Privately, through GitHub's [Report a vulnerability](https://github.com/Yog-Sotho/LiSa-LLM/security/advisories/new)
form. See [SECURITY.md](../SECURITY.md).

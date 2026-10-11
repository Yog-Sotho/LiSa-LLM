
<div align="center">
  <img src="assets/Logo_LiSa.png" alt="LiSa Logo" width="420" style="margin-bottom: 20px;">
  <h1>LiSa-LLM – Lightweight, Sandboxed LLM Inference Server</h1>
  <p><em>“Because your cat‑sized model deserves a lion‑sized sandbox.”</em></p>
  <p><strong>LiSa-LLM is a small, self-contained C++20 inference server for GGUF models, built on llama.cpp's <code>libllama</code> with optional CUDA offload. It provides streaming Server-Sent Events, API-key authentication and TLS. A hardened Linux sandbox for running untrusted models is under active development (see Project status).</strong></p>
  <p>
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/C++20-ready-blue.svg" alt="C++20 Ready">
    <img src="https://img.shields.io/badge/GPU-CUDA-green.svg" alt="GPU Required">
    <a href="https://github.com/sponsors/Yog-Sotho" target="_blank" rel="noopener">
      <img src="https://img.shields.io/badge/Sponsor❤️-30363D.svg?logo=githubsponsors&logoColor=EA4AAA" alt="Sponsor on GitHub">
    </a>
  </p>
</div>

## Project status

LiSa-LLM is being rebuilt following the [codebase audit](docs/project/codebase_audit.md).
**Phases P0 and P1 are done**: the project builds reproducibly, inference runs on
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `libllama`, the API is OpenAI-compatible
with chat templates, and CI runs sanitizers, static analysis and end-to-end model and HTTP
tests. Next up: a sandbox that really isolates the model (P2).

## Features

- 🧠 **Any GGUF model llama.cpp supports** – LLaMA 2/3, Mistral, Qwen, Gemma, Phi and more,
  every quantisation type; the tokenizer is read from the GGUF file.
- ⚡ **GPU offload** – CUDA build option with configurable layer count (`gpu_layers`).
- 🔌 **OpenAI-compatible API** – `/v1/chat/completions`, `/v1/completions`, `/v1/models`; works
  with the official OpenAI SDKs. Chat templates come from the GGUF file.
- 📡 **Real-time streaming** via Server-Sent Events; generation stops when the client disconnects.
- 🎛️ **Sampling** – temperature, top-p, top-k, min-p, presence/frequency/repeat penalties,
  seed and stop sequences.
- 🌐 **HTTPS and API keys** – optional TLS, Bearer/`X-API-Key` authentication with
  constant-time comparison, strict request validation and a 1 MiB body limit.
- 🧪 **Tested** – unit tests plus end-to-end generation tests against a real model under
  ASan/UBSan and TSan; clang-tidy, cppcheck and CodeQL in CI.
- 🔒 **Sandbox (experimental, off by default)** – namespaces, seccomp, cgroup v2 and
  `pivot_root` code exists but **does not isolate the server yet**; it is being redesigned
  as a broker/worker split (roadmap P2).

## Build & Run

### Dependencies

- CMake ≥ 3.28 and Ninja (or Make)
- C++20 compiler (GCC 13+, Clang 18+)
- yaml-cpp (built from source if not installed)
- OpenSSL (for TLS; `-DLISA_TLS=OFF` to build without)
- libseccomp (for the sandbox; `-DLISA_SANDBOX=OFF` to build without)
- CUDA toolkit (optional, for `-DLISA_CUDA=ON`)

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install build-essential cmake ninja-build pkg-config libssl-dev libyaml-cpp-dev libseccomp-dev
```

**Fedora / RHEL:**
```bash
sudo dnf install gcc-c++ cmake ninja-build openssl-devel yaml-cpp-devel libseccomp-devel
```

**Arch Linux:**
```bash
sudo pacman -S base-devel cmake ninja openssl yaml-cpp libseccomp
```

llama.cpp, cpp-httplib, nlohmann/json and Catch2 are fetched by CMake at configure time,
pinned to a release tag whose commit (or archive SHA-256) is verified. For offline builds,
point `FETCHCONTENT_SOURCE_DIR_<NAME>` at local checkouts.

### Compile

```bash
cmake --preset release          # portable optimised build (presets: dev, asan-ubsan, tsan, release, cuda)
cmake --build --preset release
ctest --preset release          # unit tests; set LISA_TEST_MODEL=/path/to/model.gguf for engine tests
```

Main CMake options: `LISA_CUDA` (OFF), `LISA_TLS` (ON), `LISA_SANDBOX` (ON), `LISA_NATIVE`
(ON, tune for the build CPU), `LISA_HARDENING` (ON), `LISA_SANITIZERS` (e.g. `address,undefined`).

### Prepare a model

Download any GGUF model, for example:

```bash
mkdir -p models
curl -L -o models/model.gguf \
  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

Then set `model_path` in `config.yaml`.

### Run

```bash
./build/release/lisa --config config.yaml
```

You should see logs like:

```json
{"level":"info","message":"Loading model","path":"./models/model.gguf","timestamp":1791636189245}
{"gpu":false,"level":"info","message":"Model loaded","model":"llama 7B Q4_K - Medium","n_ctx":4096,"n_ctx_train":4096,"n_layer":32,"n_params":6738415616,"timestamp":1791636189259}
{"bind":"127.0.0.1:8080","level":"info","message":"Starting server","timestamp":1791636189259,"tls":false}
```

`SIGINT`/`SIGTERM` stop the server cleanly.

### Use the API

LiSa-LLM speaks the **OpenAI API**, so existing OpenAI clients and SDKs work unchanged:

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, formatted with the model's chat template |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | The loaded model |
| `GET /healthz` | Liveness check (no key needed) |

Authenticate with `Authorization: Bearer <api_key>` (or `X-API-Key: <api_key>`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="changeme")
reply = client.chat.completions.create(
    model="llama-2-7b-chat.Q4_K_M",            # any value works; one model is loaded
    messages=[{"role": "user", "content": "Write a haiku about sandboxes."}],
    max_tokens=100,
    stream=True,
)
for chunk in reply:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer changeme" -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

**Supported request fields:** `messages` (roles `system`/`developer`, `user`, `assistant`;
text content) or `prompt`; `max_tokens` / `max_completion_tokens`; `temperature` (0 = greedy);
`top_p`; `presence_penalty`; `frequency_penalty`; `seed`; `stop` (up to 4 sequences);
`stream` and `stream_options.include_usage`. llama.cpp extensions: `top_k`, `min_p`,
`repeat_penalty`. Responses include `usage` and `finish_reason` (`stop` / `length`).

Features the server doesn't implement (`n` > 1, `logprobs`, `tools`, JSON `response_format`,
`logit_bias`, `echo`, `suffix`) are rejected with `400` rather than silently ignored. Errors
use OpenAI's format: `{"error": {"message", "type", "param", "code"}}`. Without `max_tokens`,
generation is limited by `max_new_tokens_default` and the free context.

**Chat templates.** The template stored in the GGUF file is detected automatically (54
built-in formats: Llama 2/3/4, ChatML, Mistral, Gemma, Phi, DeepSeek, Qwen, Command-R, …). Set
`chat_template` in the config to force one (e.g. `llama3`, `chatml`); models without a
template fall back to ChatML. Message content that spells a special token (e.g.
`<|im_end|>`) is rejected, so users cannot forge role boundaries.

### Docker

```bash
docker build -t lisa-llm .     # CUDA: see the build args at the top of the Dockerfile
docker run --rm -p 8080:8080 \
  -v ./models:/models:ro \
  -v ./my-config.yaml:/etc/lisa/config.yaml:ro \
  lisa-llm
```

The image runs as an unprivileged user and listens on `0.0.0.0` inside the container. It
refuses to start with the placeholder `api_key`, so mount a config with a strong key.

## Configuration

See [`config.yaml`](config.yaml) for every option with comments:

```yaml
model_path: "./models/model.gguf"
model_alias: ""                  # model id in API responses; default: file name
chat_template: ""                # built-in template name; default: from the GGUF file

listen_addr: "127.0.0.1"
listen_port: 8080
enable_tls: false
cert_file: ""
key_file: ""
api_key: "changeme"          # set to empty string to disable auth

n_threads: 8
max_context: 4096
temperature_default: 0.8
top_p_default: 0.95
max_new_tokens_default: 256

memory_limit_bytes: 4294967296   # 4 GiB (sandbox only)
sandbox_enabled: false           # experimental, see Project status
use_gpu: true
gpu_layers: -1                   # -1 = all layers (CUDA builds)
```

## Security notes

- The sandbox is **not** a security boundary yet; don't rely on it to contain untrusted
  models until roadmap phase P2 lands.
- If you expose the server beyond localhost, enable TLS and set a strong API key (the server
  refuses the placeholder key on non-loopback addresses).
- Run the binary under a dedicated, unprivileged user (the Docker image does).

## Limitations

- Requests are served one at a time (continuous batching is planned, roadmap P4).
- Chat templates use llama.cpp's 54 built-in formats; arbitrary Jinja templates, tool calls
  and structured (JSON) output are not supported yet.
- GPU support is NVIDIA CUDA only for now (llama.cpp's Vulkan/Metal/ROCm backends are not wired up).

Contributing

Fork, hack, and open a PR. Keep the humour and the seccomp filters tight.

License

MIT – because sharing is caring, even when sandboxing.

---

<div align="center">
  Made with ❤️ by Yog-Sotho<br>
  <i>“May your logits be low‑entropy and your cgroups never OOM.”</i>
</div>

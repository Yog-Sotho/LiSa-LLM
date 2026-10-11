
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
**Phase P0 is done**: the project builds reproducibly, inference runs on
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `libllama`, and CI runs sanitizers,
static analysis and end-to-end model tests. Next up: OpenAI-compatible API and chat
templates (P1), then a sandbox that really isolates the model (P2).

## Features

- 🧠 **Any GGUF model llama.cpp supports** – LLaMA 2/3, Mistral, Qwen, Gemma, Phi and more,
  every quantisation type; the tokenizer is read from the GGUF file.
- ⚡ **GPU offload** – CUDA build option with configurable layer count (`gpu_layers`).
- 📡 **Real-time streaming** via Server-Sent Events; generation stops when the client disconnects.
- 🌐 **HTTP API** with optional TLS and API-key authentication; request validation and a
  1 MiB body limit.
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

### Test the API

```bash
# Health check
curl http://localhost:8080/healthz

# Generate a completion (streaming)
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9}'
```

The response is a Server-Sent Events stream:

```
data: {"token":","}
data: {"token":" there"}
data: {"token":" was"}
...
data: [DONE]
```

Request fields: `prompt` (required), `max_new_tokens` (1 … `max_context`), `temperature`
(0 … 2, 0 = greedy), `top_p` (0 < p ≤ 1). Invalid requests and prompts that don't fit the
context window get `400` with a JSON `error`.

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
- Raw completions only; chat templates and the OpenAI-compatible API come in P1/P3.
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

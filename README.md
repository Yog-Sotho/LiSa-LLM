
<div align="center">
  <img src="assets/Logo_LiSa.png" alt="LiSa Logo" width="420" style="margin-bottom: 20px;">
  <h1>LiSa-LLM – Lightweight, Sandboxed LLM Inference Server</h1>
  <p><em>“Because your cat‑sized model deserves a lion‑sized sandbox.”</em></p>
  <p><strong>LiSa-LLM is a small, self-contained C++20 inference server for GGUF models, built on llama.cpp's <code>libllama</code> with optional CUDA offload. It provides streaming Server-Sent Events, API-key authentication and TLS, and parses every model inside a locked-down worker process (namespaces, empty read-only root, Landlock, seccomp), so a malicious GGUF file can't reach your system.</strong></p>
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
**Phases P0–P4 are done**: the project builds reproducibly, inference runs on
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `libllama`, the API is OpenAI-compatible
with chat templates, the model runs in an isolated worker process, and the API is hardened
for exposure (multiple hashed keys, rate limits, a bounded queue, timeouts, mTLS, Prometheus
metrics, graceful shutdown). CI runs sanitizers, static analysis, end-to-end model and HTTP
tests, escape attempts against a live worker, and libFuzzer on the request and IPC decoders.
With continuous batching and a prompt cache it is faster than llama.cpp's own `llama-server`
under concurrent load, sandbox included (see [Performance](#performance)). Next up: project
hygiene (P5: releases, SBOM, coverage).

## Features

- 🧠 **Any GGUF model llama.cpp supports** – LLaMA 2/3, Mistral, Qwen, Gemma, Phi and more,
  every quantisation type; the tokenizer is read from the GGUF file.
- 🚀 **Continuous batching** – up to `parallel_slots` requests generate together in one
  batch; new prompts are evaluated alongside running streams, so nobody waits for a long
  generation to finish.
- ♻️ **Prompt cache** – a request reuses the longest matching prefix already in any slot
  (a shared system prompt, the earlier turns of a chat) and only evaluates the rest;
  reported as `usage.prompt_tokens_details.cached_tokens`.
- ⚡ **GPU offload** – CUDA build option with configurable layer count (`gpu_layers`);
  flash attention and quantised (`q8_0`/`q4_0`) KV cache options.
- 🔌 **OpenAI-compatible API** – `/v1/chat/completions`, `/v1/completions`, `/v1/models`; works
  with the official OpenAI SDKs. Chat templates come from the GGUF file.
- 📡 **Real-time streaming** via Server-Sent Events; generation stops when the client disconnects.
- 🎛️ **Sampling** – temperature, top-p, top-k, min-p, presence/frequency/repeat penalties,
  seed and stop sequences; JSON mode (`response_format: {"type": "json_object"}`) constrained
  by a grammar, so the output is always JSON.
- 🌐 **Ready for exposure** – TLS 1.2+/1.3 with forward-secret AEAD suites, optional mutual
  TLS, HSTS; multiple API keys stored as SHA-256 digests; per-key rate limits; a bounded
  first-come-first-served queue (429 + `Retry-After` when full); timeouts; strict request
  validation and a 1 MiB body limit.
- 📈 **Operable** – `/healthz`, `/readyz` (for load balancers and Kubernetes), Prometheus
  `/metrics` (latency, queue wait, time to first token, tokens/s, rejections, worker
  restarts), JSON access logs with request ids (`X-Request-Id`), graceful drain on SIGTERM.
- 🧪 **Tested** – unit tests plus end-to-end generation tests against a real model under
  ASan/UBSan and TSan, TLS/mTLS tests with generated certificates, libFuzzer targets for
  untrusted input; clang-tidy, cppcheck and CodeQL in CI.
- 🔒 **Sandboxed inference (on by default)** – the network-facing broker never parses the
  model. A separate worker does, with no network, no visible files except the model, no other
  processes, no capabilities, and a seccomp allow-list that kills it on anything else. Crashed
  workers are logged and restarted. Optional `model_sha256` pinning. See the
  [threat model](docs/project/threat_model.md).

## Build & Run

### Dependencies

- CMake ≥ 3.28 and Ninja (or Make)
- C++20 compiler (GCC 13+, Clang 18+)
- yaml-cpp (built from source if not installed)
- OpenSSL 3 (libcrypto; model SHA-256, and TLS)
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
# or without a config file:
./build/release/lisa --model models/model.gguf --port 8080
```

| Option | Purpose |
|---|---|
| `-c, --config <path>` | YAML config (default: `config.yaml` if present) |
| `-m, --model <path>` | Model file, overrides `model_path` |
| `--host <addr>`, `-p, --port <n>` | Listen address and port |
| `--check-config` | Validate the configuration and the model file (and its SHA-256), then exit |
| `--hash-key` | Read an API key on stdin and print its `sha256:` entry for `api_keys` |

Every config key can also be set as an environment variable `LISA_<KEY>`, e.g.
`LISA_LISTEN_PORT=9000` or `LISA_API_KEYS=key1,key2`. Precedence: command line >
environment > config file > defaults.

You should see logs like:

```json
{"level":"info","message":"LiSa-LLM starting","sandbox":true,"timestamp":1791764103710,"version":"1.0.0"}
{"level":"info","message":"Loading model","path":"./models/model.gguf","sandbox":true,"timestamp":1791764103710}
{"landlock_abi":7,"level":"info","message":"Sandboxed worker ready","pid":5456,"timestamp":1791764108919}
{"chat_template":"model (auto-detected)","gpu":false,"level":"info","message":"Model loaded","model":"qwen2 1B Q4_K - Medium","n_ctx":2048,"n_ctx_train":32768,"n_layer":24,"n_params":630167424,"timestamp":1791764108919}
{"bind":"127.0.0.1:8080","level":"info","message":"Starting server","model":"model","timestamp":1791764108919,"tls":false}
```

`SIGINT`/`SIGTERM` drain the server: new requests get 503, `/readyz` turns 503, running and
queued requests finish (up to `shutdown_timeout_seconds`, then they are cancelled). A second
signal stops at once.

### Sandbox requirements

The sandbox needs Linux with **unprivileged user namespaces** (kernel ≥ 5.13 recommended for
Landlock). If it can't be set up, the server refuses to start rather than run unprotected.

- **Ubuntu 23.10+** restricts user namespaces through AppArmor:
  `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0` (or an AppArmor profile for `lisa`).
- **Docker**: add `--security-opt seccomp=unconfined --security-opt apparmor=unconfined`
  (the worker installs a much stricter seccomp filter of its own) and set
  `sandbox_enabled: true`; the bundled container config has it off.
- **GPU offload** can't be sandboxed (drivers need device nodes and `ioctl`): it requires
  `sandbox_enabled: false`, which logs a warning.

`memory_limit_bytes` caps the worker's heap (weights repacked for the CPU, KV cache, compute
buffers); raise it for large models. For hard limits, point `sandbox_cgroup` at a delegated
cgroup v2 directory (e.g. a systemd unit with `Delegate=yes`).

### Use the API

LiSa-LLM speaks the **OpenAI API**, so existing OpenAI clients and SDKs work unchanged:

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, formatted with the model's chat template |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | The loaded model |
| `GET /healthz` | Liveness check (no key needed) |
| `GET /readyz` | Readiness: 503 while draining or when the worker keeps crashing (no key) |
| `GET /metrics` | Prometheus metrics (needs a key when keys are configured) |

Authenticate with `Authorization: Bearer <api_key>` (or `X-API-Key: <api_key>`). Every response
carries an `X-Request-Id` (yours if you send a plain one, else a random one) that also appears
in the server's access log.

When the queue is full (`max_queued_requests`) or a key exceeds `rate_limit_per_minute`, the
server answers `429` with a `Retry-After` header, which the OpenAI SDKs honour.

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
`stream` and `stream_options.include_usage`; `response_format` `text` or `json_object`.
llama.cpp extensions: `top_k`, `min_p`, `repeat_penalty`. Responses include `usage` (with
`prompt_tokens_details.cached_tokens`) and `finish_reason` (`stop` / `length`).

Features the server doesn't implement (`n` > 1, `logprobs`, `tools`, `json_schema`
response formats, `logit_bias`, `echo`, `suffix`) are rejected with `400` rather than silently ignored. Errors
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
refuses to start with the placeholder `api_key`, so mount a config with a strong key. To run
the sandbox in the container, see [Sandbox requirements](#sandbox-requirements).

## Performance

Throughput and time to first token (TTFT) at increasing concurrency, measured with
[`bench/bench.py`](bench/bench.py) on the same machine and model for both servers:
Qwen2.5-0.5B-Instruct Q4_K_M, 4 vCPUs (Xeon @ 2.8 GHz, AVX2), 4 threads, 64 tokens per
request, 8 requests per level, every prompt distinct. LiSa runs **with the sandbox on**, 8
slots; `llama-server` is built from the same llama.cpp commit with `-np 8`.

| Concurrency | LiSa tok/s | llama-server tok/s | LiSa TTFT p95 | llama-server TTFT p95 |
|---:|---:|---:|---:|---:|
| 1 | 37.4 | 35.6 | 0.31 s | 0.36 s |
| 2 | 48.8 | 42.8 | 0.77 s | 0.77 s |
| 4 | 61.9 | 47.2 | 1.07 s | 1.41 s |
| 8 | 66.7 | 59.2 | 2.38 s | 2.56 s |

With a long prompt shared by every request (`--shared-prefix`, e.g. a system prompt), the
prompt cache copies the prefix between slots: 90.4 tok/s and 0.20 s TTFT p95 at
concurrency 8 (`llama-server`: 82.8 tok/s, 0.12 s). A single request runs at the same speed
as `llama-server`, so the sandbox costs nothing measurable. Before continuous batching
(P3), LiSa stayed at about 37 tok/s whatever the concurrency. Expect ±5 % between runs.

Reproduce with any OpenAI-compatible server:

```bash
python3 bench/bench.py --url http://127.0.0.1:8080 --key "$KEY" --concurrency 1,2,4,8
```

Tuning: `parallel_slots` sets how many requests share a batch (each reserves KV cache for
`max_context` tokens); `kv_cache_type: q8_0` halves KV memory; keep `n_threads` at or below
the number of cores.

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
client_ca_file: ""               # require client certificates from this CA (mTLS)
tls_min_version: "1.2"           # or "1.3"

api_key: "changeme"              # see also api_keys, api_keys_file, LISA_API_KEYS
api_keys: []                     # keys or "sha256:<hex>" digests (lisa --hash-key)
api_keys_file: ""                # one key or digest per line

max_queued_requests: 16          # more waiting requests get 429
rate_limit_per_minute: 0         # per key; 0 = unlimited
rate_limit_burst: 10
max_prompt_tokens: 0             # 0 = context window only
read_timeout_seconds: 30
write_timeout_seconds: 60
keep_alive_timeout_seconds: 5
shutdown_timeout_seconds: 30

n_threads: 0                     # 0 = one per available CPU
parallel_slots: 4                # requests batched together (KV cache: slots × max_context)
flash_attention: "auto"          # auto | on | off
kv_cache_type: "f16"             # f16 | q8_0 | q4_0
max_context: 4096
temperature_default: 0.8
top_p_default: 0.95
max_new_tokens_default: 256

model_sha256: ""                 # optional: refuse to start on any other file

sandbox_enabled: true            # parse the model only in the isolated worker
memory_limit_bytes: 4294967296   # worker heap limit, 4 GiB
sandbox_cgroup: ""               # optional delegated cgroup v2 dir (memory/pids limits)
sandbox_seccomp_log: false       # diagnose only: log blocked syscalls instead of killing
use_gpu: true
gpu_layers: -1                   # -1 = all layers (CUDA builds)
```

Unknown keys are rejected, so a typo can't silently leave a setting at its default.

## Security notes

- The sandbox is designed to contain a malicious model file; read the
  [threat model](docs/project/threat_model.md) for what it does and doesn't cover (kernel
  bugs, CPU exhaustion, the unsandboxed network broker, GPU mode).
- If you expose the server beyond localhost, enable TLS and use random keys of at least 16
  characters: off loopback the server refuses the placeholder `changeme` and shorter keys.
  Prefer `sha256:` entries (`lisa --hash-key`) in `api_keys_file` over plaintext keys.
- Set `rate_limit_per_minute` and keep `max_queued_requests` small so one client can't
  monopolise the model; put `/readyz` behind your load balancer's health check.
- Run the binary under a dedicated, unprivileged user (the Docker image does).

## Limitations

- Prompt caching works within the running process (no disk or RAM-tier cache across
  restarts); speculative decoding and multi-GPU split are not wired up.
- Chat templates use llama.cpp's 54 built-in formats; arbitrary Jinja templates, tool calls
  and JSON-schema-constrained output (`json_schema`) are not supported yet (JSON mode is).
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

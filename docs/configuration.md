# ⚙️ Configuration

LiSa-LLM reads its settings from three places. When the same setting is in more than one, the
first one in this list wins:

1. **The command line**: `--model`, `--host`, `--port`
2. **Environment variables**: `LISA_<KEY>` for every key below, e.g. `LISA_LISTEN_PORT=9000`
3. **A YAML file**: `--config path.yaml`, or `config.yaml` in the current folder if there is one
4. Built-in defaults

The sample [`config.yaml`](../config.yaml) lists every key with a comment. Unknown keys are an
error, so a typo can't quietly leave a setting at its default. Check a configuration without
starting the server:

```bash
lisa --config config.yaml --check-config     # prints "configuration OK", or what's wrong
```

---

## Command line

| Option | What it does |
|---|---|
| `-c, --config <path>` | YAML configuration file (default: `config.yaml` if present) |
| `-m, --model <path>` | GGUF model file (overrides `model_path`) |
| `--host <addr>` | Address to listen on (overrides `listen_addr`) |
| `-p, --port <n>` | Port to listen on (overrides `listen_port`) |
| `--check-config` | Validate the configuration and the model file (and its SHA-256), then exit |
| `--health-check` | Ask the configured server whether it is ready; exit 0 if so, 1 if not |
| `--hash-key` | Read an API key on stdin and print its `sha256:` entry |
| `-v, --version` | Print the version |
| `-h, --help` | Show the help |

---

## Every setting

### Model

| Key | Default | What it does |
|---|---|---|
| `model_path` | *(required)* | The GGUF file to serve |
| `model_alias` | file name without `.gguf` | Model id shown by `/v1/models` and in responses |
| `chat_template` | from the GGUF file (ChatML if it has none) | Force a built-in format: `llama3`, `llama2`, `chatml`, `mistral-v7`, `gemma`, `phi4`, `deepseek3`, … |
| `model_sha256` | empty | 64 hex digits; the server refuses to start on any other file |

### Network and TLS

| Key | Default | What it does |
|---|---|---|
| `listen_addr` | `127.0.0.1` | Address to listen on; `0.0.0.0` for every interface |
| `listen_port` | `8080` | Port (1–65535) |
| `enable_tls` | `false` | Serve HTTPS (needs `cert_file` and `key_file`) |
| `cert_file`, `key_file` | empty | PEM certificate (chain) and private key |
| `client_ca_file` | empty | Require client certificates signed by this CA (mutual TLS) |
| `tls_min_version` | `"1.2"` | `"1.2"` or `"1.3"`. TLS 1.2 only offers forward-secret AEAD suites |

### API keys

| Key | Default | What it does |
|---|---|---|
| `api_key` | empty | One key (the sample config ships the placeholder `changeme`) |
| `api_keys` | `[]` | A list of keys |
| `api_keys_file` | empty | A file with one key per line (`#` comments and blank lines are ignored) |

All of these add up, together with `LISA_API_KEYS` (comma-separated). Clients may use any of the
keys. With no key anywhere, authentication is off (fine on localhost; warned about elsewhere).

**Don't store keys in plain text.** Any entry can be the key's SHA-256 digest instead, written as
`sha256:<hex>`. The server only ever keeps digests in memory and compares them in constant time.

```bash
openssl rand -hex 24 | tee my-key.txt             # make a key, give it to your client
lisa --hash-key < my-key.txt                      # → sha256:6066dbde0a65…
echo "sha256:6066dbde0a65…" >> /etc/lisa/keys     # api_keys_file: "/etc/lisa/keys"
```

Off loopback, the server **refuses to start** with the placeholder `changeme` or any plain-text
key shorter than 16 characters.

### Admission, limits and timeouts

| Key | Default | What it does |
|---|---|---|
| `max_queued_requests` | `16` | Requests allowed to wait for a free slot (0–4096); more get `429` with `Retry-After` |
| `rate_limit_per_minute` | `0` (off) | Requests per minute per key (per client address without keys) |
| `rate_limit_burst` | `10` | Requests a key may send at once before the per-minute rate applies |
| `max_prompt_tokens` | `0` (off) | Longest prompt accepted; 0 = only the context window limits it |
| `read_timeout_seconds` | `30` | Time to receive a request |
| `write_timeout_seconds` | `60` | Time for each write of a response (protects against slow readers) |
| `keep_alive_timeout_seconds` | `5` | Idle time before a kept-alive connection is closed |
| `shutdown_timeout_seconds` | `30` | How long a shutdown waits for running requests before cancelling them |

Requests are admitted first come, first served: up to `parallel_slots` generate at once, the next
`max_queued_requests` wait in line, the rest are turned away straight away with a `Retry-After`
hint instead of piling up.

### Engine

| Key | Default | What it does |
|---|---|---|
| `n_threads` | `0` | CPU threads; 0 = one per CPU the process may use. More than the cores only slows it down |
| `parallel_slots` | `4` | Requests generated together (1–64). Each slot reserves KV cache for `max_context` tokens |
| `max_context` | `2048` (sample config: 4096) | Context window per slot, capped at the model's training context |
| `flash_attention` | `"auto"` | `auto`, `on` or `off` |
| `kv_cache_type` | `"f16"` | `f16`, `q8_0` (half the KV memory) or `q4_0` (a quarter); quantised types need flash attention |
| `temperature_default` | `0.8` | When a request doesn't set `temperature` |
| `top_p_default` | `0.95` | When a request doesn't set `top_p` |
| `max_new_tokens_default` | `128` (sample config: 256) | When a request doesn't set `max_tokens` (also capped by the free context) |
| `use_gpu` | `true` | Offload to the GPU in CUDA builds (requires `sandbox_enabled: false`) |
| `gpu_layers` | `-1` | Layers to offload; -1 = all |

See [Performance](performance.md) for how to choose `parallel_slots`, `max_context` and
`kv_cache_type`.

### Sandbox

| Key | Default | What it does |
|---|---|---|
| `sandbox_enabled` | `true` | Parse and run the model only in the isolated worker. If the sandbox can't be set up, the server refuses to start |
| `memory_limit_bytes` | `4294967296` (4 GiB) | Heap limit of the worker (`RLIMIT_DATA`, and `memory.max` with `sandbox_cgroup`): weights repacked for the CPU, KV cache, buffers. Raise it for big models |
| `sandbox_cgroup` | empty | A delegated cgroup v2 folder; the worker gets a child cgroup with hard memory and process-count limits |
| `sandbox_seccomp_log` | `false` | For diagnosis only: log blocked system calls instead of killing the worker |

More in [Security and the sandbox](security.md).

---

## Environment variables

Every key has an environment variable: `LISA_` plus the key in capitals.

```bash
LISA_MODEL_PATH=/models/model.gguf
LISA_LISTEN_ADDR=0.0.0.0
LISA_API_KEYS=sha256:6066dbde…,sha256:9f86d081…     # comma-separated list
LISA_PARALLEL_SLOTS=8
LISA_SANDBOX_ENABLED=true
```

This is how the container images and [`docker-compose.yml`](../docker-compose.yml) are configured.
An empty value (`LISA_API_KEY=`) sets the key to empty, which is how you clear the image's
placeholder key.

## Logs

LiSa writes one JSON object per line: information to stdout, warnings and errors to stderr. Each
line has `timestamp` (milliseconds since 1970), `level` and `message`, plus fields that depend on
the event. HTTP requests are logged once each, when they finish (see
[Using the API → Request ids](api.md#request-ids)). Ship them to Loki, Elasticsearch or
`journalctl -o cat | jq` as they are.

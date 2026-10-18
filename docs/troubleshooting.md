# 🩺 Troubleshooting

The errors you're most likely to meet, what they mean and how to fix them. LiSa logs one JSON
object per line; the interesting part is usually the `error` field. Two commands catch most
problems before they happen:

```bash
lisa --config config.yaml --check-config    # settings and model file
lisa --config config.yaml --health-check    # is the running server ready?
```

---

## Starting up

### ❌ Sandbox: user namespaces look blocked

```json
{"error":"sandbox: unshare namespaces: Operation not permitted (unprivileged user namespaces look blocked: on Ubuntu 23.10+ run 'sysctl kernel.apparmor_restrict_unprivileged_userns=0' or add an AppArmor profile for lisa; in Docker run with '--security-opt seccomp=unconfined --security-opt apparmor=unconfined'; or set sandbox_enabled: false)","level":"error","message":"Model loading failed",…}
```

The system doesn't let normal users create the namespaces the sandbox is built on. LiSa refuses to
start rather than run the model unprotected. The step named in the message may differ
(`uid_map`, `make mounts private`, …); the fix is the same.

**Fix**, depending on where you run:
- **Ubuntu 23.10+:** `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`; to keep it
  after a reboot, put `kernel.apparmor_restrict_unprivileged_userns=0` in
  `/etc/sysctl.d/60-lisa.conf`.
- **Docker / Podman:** add `--security-opt seccomp=unconfined --security-opt apparmor=unconfined`
  (see [Deployment](deployment.md#sandbox-in-a-container)).
- **Other distributions:** check `sysctl kernel.unprivileged_userns_clone` (Debian-based kernels)
  or `user.max_user_namespaces` (must be > 0).
- **Can't change any of that:** `sandbox_enabled: false`, and only serve models you trust.

### ❌ Refusing to listen on a non-loopback address

```json
{"level":"error","listen_addr":"0.0.0.0","message":"Refusing to listen on a non-loopback address: the placeholder API key 'changeme' is configured; use a random key of at least 16 characters",…}
```

You asked LiSa to listen on the network with the sample config's placeholder key (or a key shorter
than 16 characters). **Fix:** set a real key, e.g. `LISA_API_KEYS="$(openssl rand -hex 24)"`, and
clear the placeholder (`api_key: ""` in the config, or `LISA_API_KEY=`). In containers the server
always listens on `0.0.0.0`, so this applies there too.

### ❌ Unknown key in config

```json
{"error":"unknown key 'max_ctx' in config 'config.yaml'","level":"error","message":"Invalid configuration",…}
```

A typo, or a key from another server. **Fix:** compare with the [settings list](configuration.md#every-setting).
Unknown `LISA_*` environment variables are ignored, not rejected, so check their spelling too.

### ❌ This build needs an x86-64 CPU with AVX2, FMA and F16C

The release binary targets CPUs from 2013 on (Intel Haswell, AMD Excavator). **Fix:**
[build from source](installation.md#build-from-source) on that machine with
`cmake --preset release -DLISA_NATIVE=ON`, which tunes for the CPU it runs on (the plain
`release` preset uses the same portable x86-64-v3 flags as the release binary).

### ❌ Sandboxed worker failed to start

```
sandboxed worker failed to start (killed by SIGABRT); if the log shows an allocation failure, raise memory_limit_bytes or lower parallel_slots or max_context
```

Usually memory: the model plus the KV cache for every slot didn't fit in the worker's limit
(`memory_limit_bytes`, 4 GiB by default). **Fix:** raise `memory_limit_bytes`, or lower
`parallel_slots` / `max_context`, or use `kv_cache_type: q8_0` with flash attention. The
[memory section](performance.md#memory) helps size it. If the status says
**`(blocked system call)`** instead, see [the worker keeps dying](#-the-worker-keeps-dying-blocked-system-call).

### ❌ Failed to create inference context

```
failed to create inference context (if memory ran out, lower parallel_slots or max_context, or use kv_cache_type q8_0)
```

The same problem without the sandbox: not enough RAM (or VRAM) for every slot's KV cache. Same fix.

### ❌ Model SHA-256 mismatch

```
model SHA-256 mismatch: expected 3b1a…, got 9c4f…
```

The file isn't the one you pinned with `model_sha256`. Either it was replaced (find out why!) or
you upgraded the model on purpose; then update the checksum (`sha256sum model.gguf`).

### ❌ GPU offload inside the sandbox is not supported yet

You're running a CUDA build with `use_gpu: true` and the sandbox on. GPU drivers can't work inside
it. **Fix:** `sandbox_enabled: false` for GPU inference (see [Security](security.md#when-the-sandbox-is-off)),
or `use_gpu: false` to stay sandboxed on the CPU.

### ❌ Cannot listen on 127.0.0.1:8080

Something else uses the port, or a port below 1024 needs privileges. **Fix:** `--port 9000`, or
find the other process with `ss -ltnp | grep 8080`.

### ❌ Unknown chat_template

```
unknown chat_template 'llama-3'; built-in templates: bailing, bailing-think, bailing2, chatglm3, …
```

**Fix:** use one of the names listed in the error, or leave `chat_template` empty to use the
model's own.

---

## While it runs

### ❌ The worker keeps dying (blocked system call)

```json
{"error":"…","level":"error","message":"Inference worker died","status":"killed by SIGSYS (blocked system call)",…}
```

The worker tried a system call that isn't on its allow-list, and the seccomp filter killed it. On
a known-good model and a normal platform this shouldn't happen, so treat it as suspicious: it is
exactly what a model exploit attempt looks like. If you trust the model, run once with
`sandbox_seccomp_log: true` to log the call instead of killing (the kernel's audit log shows it),
and [report it](https://github.com/Yog-Sotho/LiSa-LLM/issues) with your platform details. After
repeated crashes (more than 3 in a minute) LiSa stops restarting the worker: `/readyz` turns
`503` and requests fail with `500` until the minute is up, then the next request restarts it.

### ❌ 429 Too Many Requests

Either your key went over `rate_limit_per_minute`, or all slots are busy and `max_queued_requests`
are already waiting. The `Retry-After` header says when to try again (the OpenAI SDKs wait and
retry by themselves). **Fix on the server side:** more `parallel_slots`, a higher queue limit, a
higher rate limit, or another instance behind a load balancer. `lisa_requests_rejected_total`
shows which reason it is.

### ❌ 400 context_length_exceeded

```json
{"error":{"code":"context_length_exceeded","message":"prompt (2190 tokens) plus max_tokens (256) exceeds the context window of 2048 tokens",…}}
```

The prompt plus the answer you asked for don't fit in `max_context`. **Fix:** a shorter prompt, a
smaller `max_tokens` (or leave it out: the answer then stops when the context is full), or a
bigger `max_context`.

### ❌ 400 for a field your client sends

```json
{"error":{"message":"tools is not supported","param":"tools","type":"invalid_request_error",…}}
```

LiSa refuses OpenAI features it doesn't implement instead of quietly ignoring them. The list is in
[Using the API](api.md#not-supported-and-rejected-rather-than-ignored). Many clients have a setting
to stop sending the field.

### ❌ Answers stream in one big chunk

A proxy between you and LiSa is buffering. **Fix:** `proxy_buffering off;` in nginx (see
[Deployment → Behind a reverse proxy](deployment.md#behind-a-reverse-proxy)), or the equivalent
in your proxy.

### ❌ Slow answers

- Check `n_threads`: 0 (automatic) is best; more threads than cores hurts.
- Under load, look at `lisa_queue_wait_seconds`: if requests wait long, add slots; if every
  request is slow, the machine is at its limit, so a smaller or more heavily quantised model
  (Q4_K_M) helps.
- A shared system prompt? Keep it identical and first, so the [prompt cache](performance.md#how-it-gets-there)
  can reuse it.

### ❌ The chat output looks garbled or never stops

The chat template doesn't match the model, typically for a model whose GGUF has no template (the
startup log then warns that it falls back to ChatML). **Fix:** set `chat_template` to the model's
format (`llama3`, `mistral-v7`, `gemma`, …).

---

Still stuck? [Open an issue](https://github.com/Yog-Sotho/LiSa-LLM/issues) with your OS and kernel
version (`uname -r`), how you run it (binary, Docker, flags), your config with keys removed, and
the log lines around the error. For anything that smells like a security problem, use the private
form in [SECURITY.md](https://github.com/Yog-Sotho/LiSa-LLM/blob/main/SECURITY.md) instead.

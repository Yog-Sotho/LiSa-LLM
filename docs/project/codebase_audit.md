# Codebase Audit Report — LiSa-LLM

**Date:** 2026-10-10
**Project:** LiSa-LLM ("Lightweight, Sandboxed LLM Inference Server")
**Commit audited:** `6659c8e` (branch `main`)
**Language / Framework:** C++20, GGML (vendored), cpp-httplib 0.40.0, nlohmann/json 3.12, yaml-cpp, OpenSSL, libseccomp, CUDA (optional)
**Project Type:** Web service (HTTP inference server) — all 9 dimensions apply
**Audit Mode:** Global (single `src/`, ~1.1 kLOC first-party; ~63 kLOC vendored)

---

## Progress log

### 2026-10-11 — P1 complete (OpenAI-compatible API, chat templates, sampling)

Done:
- **API:** `/v1/chat/completions`, `/v1/completions` (both streaming and non-streaming, with
  `usage` and `finish_reason`, plus `stream_options.include_usage`) and `/v1/models`. Errors use
  OpenAI's `{"error": {message, type, param, code}}` shape, including JSON 404/413/500 bodies.
  OpenAI features that aren't implemented (`n`>1, `logprobs`, `tools`, JSON `response_format`,
  `logit_bias`, `echo`, `suffix`) return 400 instead of being silently ignored. Checked against
  the official OpenAI Python SDK 3.28.
- **Chat templates:** chosen in this order: `chat_template` from the config (any of libllama's
  54 built-in names), then the template stored in the GGUF (auto-detected), then a ChatML
  fallback with a warning. Full Jinja was considered and left out: llama.cpp keeps it in
  `common/` behind an unstable internal API.
- **Prompt-injection guard:** message content that spells a special token (detected by
  tokenizing with and without special-token parsing) is rejected, so users can't forge role
  boundaries. The template's own markers are still parsed as special tokens.
- **Sampling:** llama.cpp sampler chain penalties → top-k → top-p → min-p → temperature → seeded
  draw (greedy at T=0). Supports `presence_penalty`, `frequency_penalty`, `repeat_penalty`,
  `top_k`, `min_p` and `seed`. Up to 4 `stop` sequences, held back across token boundaries so
  they're never emitted.
- **Context handling:** an explicit `max_tokens` that doesn't fit returns 400
  `context_length_exceeded`. Without one, the default is capped to the free context, as OpenAI
  does.
- **Auth (pulled forward from P3):** `Authorization: Bearer` as well as `X-API-Key`,
  constant-time comparison, on every `/v1/*` route.
- **Fix found by the new tests:** cpp-httplib's default `getaddrinfo_a` runs on threads glibc
  creates internally, and ThreadSanitizer crashes on them. It's now built with
  `HTTPLIB_USE_NON_BLOCKING_GETADDRINFO=OFF`.
- **Tests:** 42 Catch2 cases (325 assertions): request parsing, stop matcher, OpenAI body shapes,
  engine (seeded sampling, stop sequences, chat templates, injection guard), and HTTP tests
  against a live server on an ephemeral port. Green under GCC, Clang, ASan+UBSan and TSan, plus
  a concurrent-load run under TSan.

Estimated scores after P1 (self-assessed): Security 6, Code Principles 8, Observability 4,
Lifecycle 6, Code Quality 8 → **health ≈ 7.0/10, SOTA ≈ 5.5/10**. P2 (the sandbox redesign) is
next and is the biggest remaining gap.

### 2026-10-10 — P0 complete (build, engine, CI)

Done:
- **Engine:** Option A adopted. `libllama` from llama.cpp **v0.5.0** (commit `7fe450e1`) replaces
  the custom GGUF parser, transformer graph and tokenizer (`model.*`, `tokenizer.*`,
  `model_loader.cpp` and the partial `third_party/ggml` copy were deleted). Generation does one
  prefill, then decodes token by token, detects EOG from the vocab and uses llama.cpp's sampler
  chain (greedy at T=0). It assembles UTF-8 incrementally and stops when the client disconnects.
- **Build:** CMake 3.28 rewrite. Dependencies are fetched by tag with the commit verified
  (archives by SHA-256): cpp-httplib **v0.60.1** (built without compression support),
  nlohmann/json 3.12.0, Catch2 3.16.0; yaml-cpp comes from the system or 0.9.0. Options
  `LISA_CUDA/TLS/SANDBOX/NATIVE/HARDENING/SANITIZERS`; `-Wall -Wextra -Wpedantic -Wconversion
  -Wshadow …` with `-Werror` in presets; hardening flags (stack protector, FORTIFY_SOURCE=3, CET,
  full RELRO, PIE). `CMakePresets.json`: dev, asan-ubsan, tsan, release, cuda.
- **Code:** `server.hpp`/`server.cpp` restored (real header, implementation in the `.cpp`);
  everything is in namespace `lisa`. Request validation returns 400 for bad JSON, types or ranges
  and for context overflow. 1 MiB body limit. TLS goes through `httplib::SSLServer`. The server
  refuses the placeholder `changeme` key on non-loopback binds. `--config/--version/--help` CLI.
  SIGINT/SIGTERM shut down cleanly. `waitpid` status is decoded. The config is validated.
- **Tests:** 17 Catch2 cases (config, request validation, UTF-8, engine end-to-end on
  `stories260K.gguf`) plus 3 CLI tests. Verified locally: green under GCC 13 and Clang 18,
  ASan+UBSan and TSan (engine instrumented too), plus a concurrent-request TSan run of the live
  server. clang-tidy (warnings as errors), cppcheck and hadolint are clean.
- **CI:** `.github/workflows/ci.yml` runs a gcc/clang × asan-ubsan/release matrix plus TSan,
  static analysis (clang-format, clang-tidy, cppcheck) and a Docker build with a smoke test.
  CodeQL runs through the repository's GitHub default setup. Actions are pinned by commit SHA.
- **Docker:** `Dockerfile` (renamed) on Ubuntu 24.04 with CPU or CUDA build args, a non-root
  UID 10001, the runtime libraries, `ENTRYPOINT --config /etc/lisa/config.yaml` and a
  container config that binds `0.0.0.0`. Image built and smoke-tested locally (141 MB).

Deliberately left for later phases:
- The sandbox still has the design flaws listed below. It now **defaults to off** and warns
  that it doesn't isolate anything (P2).
- API-key comparison isn't constant-time yet; no timeouts or rate limits; no `/readyz` or
  `/metrics`; no OpenAI-compatible routes or chat templates (P1/P3).
- Model SHA-256 verification is still not wired (P2).

Estimated scores after P0 (self-assessed, not a full re-audit; "Build" assumes the first
GitHub CI run is green): Build 8, Security 5, Concurrency 5, Principles 7, Dependencies 8,
Observability 4, Lifecycle 6, Quality 7, Dead code 9 → **health ≈ 6.6/10, SOTA ≈ 4/10**.
P0 went further than planned (engine swap, validation, graceful shutdown), so it is ahead of the
trajectory table. The SOTA score stays low until P1–P4 add the OpenAI API, isolation and batching.

---

## Executive Summary

LiSa-LLM has a strong idea: a small C++ GGUF inference server where the model runs inside a
defense-in-depth Linux sandbox. The current code does not do that yet. **It does not configure,
compile or link** (verified: every first-party translation unit fails, and so does the bundled
`ggml.c`). Even with the compile errors fixed, the design has fatal runtime flaws: the sandbox
`pivot_root`s into an empty tmpfs *before* the model is opened. It also puts the HTTP server in
a fresh network namespace that the host cannot reach, and it calls `setuid()` inside an unmapped
user namespace, so it always exits. On top of that, the GGUF parser uses the wrong type codes
and has no bounds checks, and the transformer graph is never actually computed. Several README
claims ("production-grade", "SHA-256 integrity checks", "works for LLaMA-2/3", "~30 syscalls
seccomp", mTLS) are not true of the current code.

**Top-priority action:** decide on the engine strategy (recommended: use `libllama` from
llama.cpp, pinned), get a green CI build, then redesign the sandbox as a broker/worker split.

## SOTA Score: **1.5 / 10**

Measured against current open-source C/C++ inference servers (llama.cpp `llama-server`,
Ollama, vLLM for features; bubblewrap/minijail for sandboxing). The codebase-health score
below is the mechanical average; the SOTA score also counts the feature gap (no OpenAI API,
no batching, no chat templates, one architecture, a tokenizer that can't work).

### Codebase-health score: **2.2 / 10** (hard caps: Security CRITICAL → ≤4.0, Build CRITICAL → ≤5.0)

| Dimension | Score | Priority | Status |
|---|---|---|---|
| Security | 1/10 | CRITICAL | FAIL |
| Build & Types | 0/10 | CRITICAL | FAIL |
| Concurrency | 4/10 | HIGH | WARN |
| Code Principles | 2/10 | HIGH | FAIL |
| Dependencies | 3/10 | MEDIUM | FAIL |
| Observability | 3/10 | MEDIUM | WARN |
| Lifecycle | 2/10 | MEDIUM | FAIL |
| Code Quality | 2/10 | MEDIUM | FAIL |
| Dead Code | 3/10 | LOW | WARN |

---

## Strengths

1. **Good product idea.** Few servers offer a sandboxed inference process. Namespaces, seccomp,
   cgroup v2 and privilege drop are the right building blocks, and they make LiSa different
   from llama-server and Ollama.
2. **Structured JSON logging** from day one (`src/utils.hpp:13-33`), with timestamps and extra
   fields merged in. This is easy to ship to Loki/ELK.
3. **Some modern C++ already:** `Model` deletes copy and implements `noexcept` moves
   (`src/model.hpp:21-26`), uses RAII-style `unload()` in the destructor, and has a mutex around
   generation (`src/inference.cpp:211`).
4. **Header-only third-party libs are vendored** (httplib, json), and a permissive MIT license
   matches all vendored components.
5. **Picks the right model format (GGUF)** and has a clean, small config surface
   (`config.yaml`).

---

## Verification method

- `cmake -S . -B build` → **fails**: `Cannot find source file: third_party/ggml/ggml-nn.c`.
- `g++-13 -std=c++20 -fsyntax-only -Wall -Wextra` per TU (with system yaml-cpp 0.8, libseccomp
  2.5.5, OpenSSL 3.0): **every TU except `utils.cpp` / `model_loader.cpp` fails**; `ggml.c`
  fails (`ggml-backend.h: No such file`).
- GGUF facts checked against the spec (`ggml-org/ggml/docs/gguf.md`); httplib API checked in the
  vendored header; llama.cpp API checked against `include/llama.h` (Context7); HF repo
  contents checked through the Hub API.

---

## Findings by Dimension

### Build & Types [0/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| CRITICAL | `CMakeLists.txt:28` | `ggml-nn.c` doesn't exist → configure fails. | Replace the vendored ggml with a pinned `FetchContent`/submodule of upstream (see Roadmap P0). |
| CRITICAL | `third_party/ggml/` | Partial snapshot: `ggml.c` needs `ggml-backend.h`, `ggml-impl.h`, `ggml-cpu.h`, `ggml-quants.h`, `ggml-threading.h`; `ggml-cuda.cu` includes ~80 missing `ggml-cuda/*.cuh`. Not buildable. | Same as above; never hand-copy ggml files. |
| CRITICAL | `src/server.hpp` | Holds the **implementation** of `HttpServer` and `#include "server.hpp"` itself → infinite include recursion. `class HttpServer` is declared nowhere. | Restore a real header (class declaration) and move the code to `server.cpp`. |
| CRITICAL | `src/server.cpp` | Is a near-copy of `sandbox.cpp` → duplicate `sandbox_init` symbol at link time. | Delete; put the server implementation here. |
| CRITICAL | `src/server.hpp:2` | Includes `cpp-httplib.h`; the vendored file is `third_party/httplib.h`. | `#include <httplib.h>`. |
| CRITICAL | `CMakeLists.txt:35,39` | Include dirs `third_party/cpp-httplib` and `third_party/json/single_include` don't exist; code includes `<nlohmann/json.hpp>` but the file is `third_party/json.hpp`. | Lay out `third_party/nlohmann/json.hpp`, or better, use `FetchContent`/`find_package(nlohmann_json)`. |
| CRITICAL | `src/model.hpp:34-40`, `src/model.cpp:49-70` | Uses `ggml_backend_t`, `ggml_backend_cpu_init`, `ggml_backend_cuda_init`, `ggml_backend_is_cuda` and others without `ggml-backend.h` / `ggml-cpu.h` / `ggml-cuda.h`. | Include the proper headers, or use libllama. |
| CRITICAL | `src/inference.cpp:104,138` | Calls `engine->model()` and `engine->kv_cache()`; neither exists and the members are private. | Make `build_graph` a member (it's already declared at `inference.hpp:37` with another signature). |
| CRITICAL | `src/inference.cpp:170` | `ggml_attn` doesn't exist in ggml. | `ggml_flash_attn_ext`, or `mul_mat → soft_max_ext(mask) → mul_mat`. |
| CRITICAL | `src/inference.cpp:226` | Calls the static `build_graph(this, ctx, ids, pos)`, which resolves to the member declaration → type error. | See above. |
| CRITICAL | `src/server.hpp:48-57` | Uses `Response::send`, which doesn't exist on `httplib::Response` (only on `DataSink`/clients). | `res.set_chunked_content_provider("text/event-stream", …)`. |
| CRITICAL | `src/server.hpp:64` | `Server::set_ssl_cert_file_and_private_key_file` doesn't exist. TLS needs `httplib::SSLServer(cert, key[, client_ca])` and `CPPHTTPLIB_OPENSSL_SUPPORT`; CMake only defines `LISA_ENABLE_TLS`. | Use `SSLServer` behind `#ifdef CPPHTTPLIB_OPENSSL_SUPPORT`. |
| CRITICAL | `src/tokenizer.cpp:22` | `"\xC2\x%02X"` → compile error (`\x used with no following hex digits`). | See the Tokenizer findings. Use the GGUF-embedded tokenizer. |
| CRITICAL | `src/sandbox.cpp:91` | `pivot_root` has no glibc wrapper. | `syscall(SYS_pivot_root, ".", ".")`. |
| HIGH | `CMakeLists.txt` | `-DENABLE_CUDA=ON` (README, Dockerfile) isn't an option; `ggml-cuda.cu` is never compiled; the CUDA language is never enabled; `GGML_USE_CUDA` is never defined → the GPU path is dead. | `option(LISA_CUDA)` → `GGML_CUDA=ON` on upstream ggml/llama.cpp. |
| HIGH | `src/sandbox.cpp:7` | Includes `<seccomp.h>` unconditionally → the build breaks with `-DENABLE_SANDBOX=OFF` or without libseccomp. | Guard the include with `LISA_ENABLE_SANDBOX`. |
| HIGH | `CMakeLists.txt:46-48,82` | OpenSSL is only found when `ENABLE_TLS`, but `model.cpp` always needs `<openssl/sha.h>`. | Always link `OpenSSL::Crypto` (or use a small vendored SHA-256). |
| MEDIUM | `CMakeLists.txt` | No warnings, no `-Werror`, no sanitizer or `CMakePresets.json`, global `add_definitions(-DHAVE_SECCOMP)` (unused). | Add a warnings target, ASan/UBSan/TSan presets and target-scoped definitions. |
| MEDIUM | `src/model.cpp:16-22` | Deprecated OpenSSL 3 `SHA256_*` API (warnings). | `EVP_DigestInit_ex(EVP_sha256())`. |
| MEDIUM | `src/tokenizer.cpp:74` | `INT_MAX` without `<climits>`; `std::isspace(char)` without `<cctype>`. | Add includes. |

### Security [1/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| CRITICAL | `src/model.cpp:29-141` | **GGUF parser has no bounds checks** (the README says it is meant for *untrusted* models). `read_string` takes the length from the file, `n_dims`/`dims`/`offset` aren't validated, and `data + offset` + `memcpy(nbytes)` isn't checked against `size`. A crafted file gives out-of-bounds reads, huge allocations or integer overflow. | Use the parser of the pinned upstream engine (`gguf.h` / libllama, which is fuzzed upstream), or a cursor type with `remaining()` checks, `n_dims ≤ 4`, overflow-checked `ne` products, `offset+nbytes ≤ data_size`, a string length ≤ 1 MiB, and a libFuzzer target. |
| CRITICAL | `src/main.cpp:26-48` + `src/sandbox.cpp:75-102` | **Sandbox can't work:** (a) after `unshare(CLONE_NEWUSER)` there is no `uid_map`/`gid_map`, so `setgid(65534)` fails with EINVAL → `_exit(1)` every time. (b) `pivot_root` into an empty tmpfs happens **before** `model.load(cfg.model_path)` and before the tokenizer files are opened → ENOENT. (c) `CLONE_NEWNET` gives the child a namespace with only a down loopback, so the listener is unreachable from the host. (d) `CLONE_NEWPID` with `unshare` affects only *children*; the worker itself isn't in the new PID namespace. | Redesign as broker/worker (Roadmap P2): open the model fd and device nodes first, use `clone3`/`clone` with namespaces + id maps (`setgroups=deny`), keep networking in an unprivileged broker and talk to the worker over a `socketpair`. |
| CRITICAL | `src/sandbox.cpp:28-39` | Seccomp allow-list is missing syscalls any multi-threaded C++/glibc program needs: `futex`, `mprotect`, `madvise`, `clone`/`clone3`, `openat`, `rt_sigreturn`, `exit`, `getrandom`, `clock_nanosleep`, `poll`/`ppoll`, `sched_yield`, `set_robust_list`, `rseq`, `ioctl` (CUDA). It does allow `socket`/`connect` (exfiltration). It uses `SCMP_ACT_KILL` (= kill *thread*). | Build the allow-list from `strace -f` of the real worker; default `SCMP_ACT_KILL_PROCESS`; no `socket`/`connect` in the worker; filter `clone` flags to threads only; add Landlock as a second layer. |
| HIGH | `src/sandbox.cpp:99-101` | After `pivot_root(".", ".")` it calls `chroot(".")` and then `umount2("/")` and **ignores failure**. If the detach fails, the old root stays reachable. | Canonical sequence: `pivot_root(".", ".")` → `umount2(".", MNT_DETACH)` (fatal on error) → `chdir("/")`. |
| HIGH | `src/sandbox.cpp:56-73` | cgroup v2: writing to `/sys/fs/cgroup/lisa_<pid>` needs a delegated subtree. Errors are silently ignored (so there's **no memory limit, without any warning**), dirs leak and are never `rmdir`'d, and there's no `pids.max`/`cpu.max`. | Use a systemd-delegated cgroup (`Delegate=yes`) or a pre-created group; fail closed when `sandbox.require_cgroup=true`; also apply `RLIMIT_AS/NOFILE/NPROC/CORE`. |
| HIGH | `config.yaml:10`, README | Ships with `api_key: "changeme"`; an empty key disables auth silently. | Refuse to start with a known-weak key on a non-loopback bind; read keys from an env var or file; support `Authorization: Bearer`. |
| HIGH | `src/server.hpp:21` | API key compared with `!=` (not constant-time). | `CRYPTO_memcmp` on SHA-256 digests of the key. |
| HIGH | `src/server.hpp:35-38` | No input validation: `max_new_tokens` may be negative or 10⁹, `temperature`/`top_p` NaN or out of range; a wrong JSON type throws `type_error`. No body-size cap, timeouts or concurrency/rate limits → a single request can hold the global mutex indefinitely (DoS). | Validate ranges; `set_payload_max_length`, `set_read_timeout`/`set_write_timeout`, a bounded queue that returns 429, a per-key token bucket. |
| HIGH | `third_party/httplib.h` (0.40.0) | Upstream is at **0.60.1** (2026-10-08). Releases since 0.40 include request-smuggling fixes (0.56–0.59: TE+CL, chunk-size control chars, request-target control chars, invalid Content-Length), multipart CPU-DoS caps (0.54) and decompression limits (0.53), plus CVE-2026-46527 (fixed in 0.44.0). | Pin ≥ 0.60.1 via FetchContent, with Dependabot/Renovate. |
| HIGH | README, `src/model.cpp:143` | "Model integrity checks (SHA-256)" is advertised, but `load()` is always called without a hash and `Config` has no `model_sha256`. | Add `model_sha256` (required when `sandbox.untrusted_model=true`); hash the *same fd* that is mmapped (no TOCTOU). |
| MEDIUM | `src/server.hpp:55-57` | Exception text is streamed to clients. | Generic error + request id; full detail only in logs. |
| MEDIUM | `src/sandbox.cpp` | No `PR_SET_DUMPABLE 0`, no capability-bounding-set drop, no `close_range()` of inherited fds, no `setsid`. | Add them (bubblewrap/minijail practice). |
| MEDIUM | `dockerfile` | Runs as root; base image not digest-pinned; no `.dockerignore` (`COPY . .` copies `.git`). | Non-root user, pinned digest, `.dockerignore`, read-only rootfs guidance. |

### Concurrency [4/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| HIGH | `src/inference.cpp:211`, `src/server.hpp:48` | One global mutex. Every httplib worker thread blocks on it, there's no queue limit, and a client disconnect doesn't cancel generation (the full answer is generated before anything is sent). | Bounded request queue + N decoding slots (continuous batching); stop when `DataSink::is_writable()` is false. |
| MEDIUM | `src/inference.cpp:223-226` | Each step builds a graph in the **model's weight context**, so metadata grows until the 200 MB context overflows (`GGML_ASSERT`). | A per-step compute context or `ggml_gallocr`/`ggml_backend_sched` with a reserved graph (handled by libllama). |
| LOW | `src/main.cpp:12` | `child_pid` is a plain global (harmless now, since the handler doesn't read it). | `volatile sig_atomic_t`, or `signalfd`. |

### Code Principles [2/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| CRITICAL | `src/inference.cpp:11-36` | RoPE is applied on the CPU to `q->data` / `k->data` **while building the graph**. Those tensors haven't been computed (`data` is NULL with `no_alloc`) → segfault, or nonsense on a GPU. | `ggml_rope_ext(…, LLAMA_ROPE_TYPE_NORM, …)` inside the graph. |
| CRITICAL | `src/inference.cpp:224-236` | `ggml_new_graph()` is never filled (`ggml_build_forward_expand` is never called) → the compute does nothing; logits are garbage. The `ggml_cpy`s into the KV cache aren't in the graph either. | `ggml_build_forward_expand(gf, logits)` plus the cache copies. |
| CRITICAL | `src/inference.cpp:155` | `kv_cache.current_len += n_tokens` runs **inside the per-layer loop** (×n_layer per step). | Advance once per decode, after the graph. |
| CRITICAL | `src/inference.cpp:143-167` | KV views: the `nb2` stride is wrong and the offset is always `0` → new K/V overwrite position 0, and `k_full` strides are inconsistent. | Correct strides (`nb1=row`, `nb2=row*n_ctx`, `offset=row*cache_len`). |
| CRITICAL | `src/inference.cpp:221-248` | Each step re-feeds the **whole** prompt and every generated token at a nonsense `pos` while also appending to the cache → wrong output plus O(n²) cost. | Prefill once, then decode one token at `pos = n_past`. |
| CRITICAL | `src/inference.cpp:119,176,186` | Uses `ggml_norm` (LayerNorm) instead of RMSNorm and never multiplies by `attn_norm`/`ffn_norm`/`output_norm` weights. No causal mask. No GQA (`head_count_kv`) → LLaMA-2-70B, LLaMA-3, Mistral and Qwen all break. | `ggml_rms_norm` × weight, a causal mask, a `n_head_kv` broadcast — or use libllama (100+ architectures). |
| CRITICAL | `src/model.cpp:81-111` | GGUF KV parsing uses the wrong type codes (spec: 4=UINT32, 5=INT32, 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64); STRING and ARRAY throw "unknown kv type" → **no real GGUF file loads** (`general.name`, `tokenizer.ggml.tokens`). Known keys are read with `read_u64` whatever their type; `eps` takes the float's **bit pattern as an integer**; `rope_theta` (f32) is read as u64. The key for eps is wrong (`llama.attention.layer_norm_rms_epsilon`). | Type-dispatching reader or upstream `gguf.h`. |
| CRITICAL | `src/model.cpp:136` | Tensor data offset is relative to the file start; the spec says it's relative to the **aligned start of the tensor-data section** (`general.alignment`, default 32). | Compute `data_start = align(header_end, alignment)`. |
| CRITICAL | `src/model.cpp:115,138` | `no_alloc = true`, then `memcpy(tensor->data, …)` into a NULL `data`. `create_backend_and_buffer` sizes the backend buffer with `ggml_get_mem_size(ctx)` (metadata size), does arithmetic on `void*`, and copies all weights (no zero-copy mmap, so RAM is doubled). | `ggml_backend_alloc_ctx_tensors`, or mmap-backed CPU buffers; libllama does both. |
| HIGH | `src/inference.cpp:41-49,197` | KV cache allocated in the weight context (no_alloc → `ggml_set_zero` on NULL data). Sized from the model's `n_ctx` (128k for LLaMA-3 → tens of GB in F32), ignoring `cfg.max_context`. | Allocate a backend buffer; `n_ctx = min(cfg.max_context, n_ctx_train)`; F16/Q8 cache. |
| HIGH | `src/inference.cpp:243` | EOS hard-coded to `2` (wrong for GPT-2 `50256` and LLaMA-3 `128001/128009`). | Read `tokenizer.ggml.eos_token_id` and EOG set; add `stop` strings. |
| HIGH | `src/inference.cpp:65-97` | Sampler: `temperature ≤ 0` becomes `1.0` (should be greedy). The nucleus isn't renormalised, so `r∈[0,1)` often falls past the selected mass and returns token `0`. Full `std::sort` of the vocab every token; `unordered_set` lookups. | Greedy at T=0, renormalise, `partial_sort`/top-k first, min-p, repetition penalties, seed — or the `llama_sampler_chain`. |
| HIGH | `src/tokenizer.cpp` | Not a working tokenizer: the byte→unicode map is wrong (GPT-2 maps to code points 256+n). `bpe()` runs on raw chars rather than byte-encoded symbols. There's no GPT-2/LLaMA-3 pre-tokenizer regex. The whole word is looked up as one id → mostly `<unk>`, and `.at("<unk>")` **throws** for GPT-2/LLaMA-3 vocabs (no `<unk>`). LLaMA-2 uses **SentencePiece**, not vocab.json/merges.txt. `std::isspace(char)` on UTF-8 bytes is UB. Per-token `decode` splits multi-byte UTF-8 in the stream. | Use the tokenizer embedded in the GGUF (`tokenizer.ggml.*`) via `llama_tokenize`/`llama_token_to_piece`, and an incremental UTF-8 assembler for streaming. |

### Dependencies [3/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| CRITICAL | `third_party/ggml` | Hand-copied, incomplete and mixed snapshot, with no version or commit recorded. | Pin `ggml-org/llama.cpp` (tag `bNNNN`) via FetchContent/submodule; record it in `third_party/VERSIONS.md`. |
| HIGH | `third_party/httplib.h` | 0.40.0 vs 0.60.1 upstream (see Security). | Pin ≥ 0.60.1. |
| LOW | `third_party/json.hpp` | 3.12.0 is current; fine, but wrong include layout. | Keep; fix the path. |
| MEDIUM | README | Tells users to download `vocab.json`/`merges.txt` from `TheBloke/Llama-2-7B-Chat-GGUF`. **They don't exist** there (the repo has only `config.json` + GGUF files). | Drop external tokenizer files. |
| LOW | — | No Dependabot/Renovate, no SBOM, no third-party license notice file. | Add them. |

### Observability [3/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| MEDIUM | `src/server.hpp:14` | `/healthz` always returns ok, even when the model isn't loaded or the worker is dead. | `/healthz` (liveness) + `/readyz` (model loaded, worker alive). |
| MEDIUM | — | No metrics (tokens/s, time to first token, queue depth, 4xx/5xx, sandbox kills). | Prometheus `/metrics` (opt-in, auth-protected). |
| MEDIUM | `src/utils.hpp` | No log levels or filter, no request id, no access log; `std::endl` flushes on every line; logging from two processes interleaves. | Levels + `LISA_LOG_LEVEL`, request id propagated broker→worker, `'\n'` + line-buffered output. |
| LOW | README | The sample log line `"Loading model"` doesn't exist in the code. | Keep the docs in sync. |

### Lifecycle [2/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| HIGH | `src/main.cpp:17` | Config path hard-coded to `./config.yaml`; no CLI. In Docker, `WORKDIR /models` + `COPY config.yaml /etc/lisa/` → the config isn't found. `YAML::Exception` isn't caught → `terminate`. | `--config` (default `/etc/lisa/config.yaml`), env overrides, validated config with clear errors. |
| HIGH | `src/main.cpp:47-55`, `src/server.hpp:66` | Without the sandbox, SIGINT/SIGTERM set `g_terminate` but nothing reads it; `svr.listen` blocks for ever → only SIGKILL stops it, and in-flight requests aren't drained. | Signal thread → `svr.stop()`; drain with a deadline. |
| HIGH | `src/main.cpp:37` | Returns the raw `waitpid` status (not `WEXITSTATUS`/`WTERMSIG`); there's no restart policy and no log of seccomp kills (SIGSYS). | Decode the status, log the signal, optional supervised restart with backoff. |
| HIGH | `dockerfile` | The runtime image lacks `libyaml-cpp0.8` and `libseccomp2` → the binary won't start. `listen_addr: 127.0.0.1` inside the container → unreachable through `EXPOSE`. The in-container sandbox needs userns/seccomp permissions that Docker's default profile blocks. No `HEALTHCHECK`. | Install runtime libs (or link statically), `LISA_LISTEN_ADDR=0.0.0.0` in the image, document `--security-opt`, add `HEALTHCHECK`. |
| MEDIUM | `src/sandbox.cpp:77` | Leaks a `/tmp/lisa_sandbox_*` dir on every start (plus the cgroup dir). | Create it under a private mount namespace before pivoting; clean up in the broker. |

### Code Quality [2/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| HIGH | repo | **No tests** (`assets/test` and `third_party/ggml/test` are placeholder files), no CI, no formatter or linter config. | GoogleTest/Catch2 + GitHub Actions (see Roadmap P0/P5). |
| MEDIUM | `src/*.cpp` | C-style casts everywhere, `int` for sizes and counts (`n_tokens * n_vocab * sizeof(float)` can overflow `int`), magic numbers (`0x46554747`, `65534`, `2`, `200 MB`). | `static_cast`, `size_t`/`int64_t`, named `constexpr` constants. |
| MEDIUM | `src/tokenizer.cpp:121-149` | Leftover stream-of-consciousness comments ("Actually…", "This is complex; the full implementation is ~500 lines. We'll assume…") show the code is unfinished. | Remove them together with the code. |
| LOW | `src/inference.hpp:24` | `const Config& cfg_` reference member (dangling-reference risk, no assignment). | Store by value. |

### Dead Code [3/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| MEDIUM | `src/model_loader.cpp` | Never compiled; duplicates `sha256_file`; defines a conflicting `struct Model` (ODR violation if ever added); its comment "mapped memory is released by ggml_free" is false (leaks the mmap). | Delete. |
| MEDIUM | `src/server.cpp` | Copy of `sandbox.cpp` (see Build). | Replace. |
| LOW | `src/tokenizer.hpp:19,22-26` | `merges_`, `byte_encode` result, `build_byte_encoder/decoder` declared but unused or undefined. | Delete with the tokenizer. |
| LOW | `src/inference.hpp:37`, `CMakeLists.txt:52` | Undefined member `build_graph` declaration; unused `HAVE_SECCOMP`. | Delete. |

---

## Advisory Findings

- `[High cohesion module]` `src/sandbox.cpp` — one clear responsibility. The problem is
  correctness, not size or structure.
- `[Single consumer]` `utils.hpp` logging helpers duplicate each other (`log_info`/`log_error`):
  under 5 lines each, so no DRY penalty.

---

## Roadmap to ≥ 8/10

### Key decision: engine strategy

| Option | What | Effort | Risk | SOTA ceiling |
|---|---|---|---|---|
| **A (recommended)** | Use **`libllama`** (llama.cpp, pinned tag) as the engine: GGUF loader, tokenizer from GGUF, chat templates, 100+ architectures, GQA, all quant types, CUDA/Vulkan/Metal offload (`n_gpu_layers`), KV cache, `llama_sampler_chain`, grammar/JSON output. Delete `model.cpp`, `inference.cpp` graph code, `tokenizer.cpp`, `third_party/ggml`. LiSa's value moves to the **sandbox + hardened API** layer, which nobody else ships. | ~1–2 weeks | Low (upstream API churn → pin + CI) | 8.5–9 |
| B | Keep your own transformer on upstream ggml (fix all the Code Principles findings: `ggml_rope_ext`, RMSNorm with weights, causal mask, GQA, `ggml_backend_sched`, upstream `gguf.h`, a real BPE/SPM tokenizer). | 6–10 weeks | High (correctness, one architecture) | ~6.5 |

The plan below assumes **Option A**. Steps that differ under B are marked *(B)*.

### P0 — Make it build and keep it green (→ ~3.5/10)

1. Restore `server.hpp` as a header (`class HttpServer { explicit HttpServer(InferenceEngine&, Config); void start(); void stop(); … }`) and move the implementation to `server.cpp`; delete the duplicated sandbox code and `model_loader.cpp`.
2. CMake rewrite: `FetchContent` for llama.cpp (pinned tag, `LLAMA_BUILD_TESTS/EXAMPLES/SERVER=OFF`, `GGML_CUDA=${LISA_CUDA}`), cpp-httplib ≥ 0.60.1 (`HTTPLIB_REQUIRE_OPENSSL` when TLS), nlohmann_json 3.12; options `LISA_CUDA`, `LISA_TLS`, `LISA_SANDBOX`; guard `<seccomp.h>`; always link `OpenSSL::Crypto`; target-scoped warnings `-Wall -Wextra -Wpedantic -Wconversion -Wshadow` (+`-Werror` in CI); `CMakePresets.json` with `dev`, `asan-ubsan`, `tsan`, `release`, `cuda`.
3. Add `.clang-format`, `.clang-tidy` (bugprone-*, cert-*, cppcoreguidelines-*, modernize-*, performance-*), `.gitignore`, `.dockerignore`.
4. GitHub Actions: matrix gcc-13/clang-18 × {Debug+ASan/UBSan, Release}; clang-tidy; cppcheck; CodeQL for C++; hadolint; build the Docker image.

### P1 — Correct inference (→ ~5/10)

1. `Engine` wrapper over libllama: `llama_model_load_from_file` (`n_gpu_layers = cfg.gpu_layers` only if `use_gpu`), `llama_init_from_model` with `n_ctx = min(cfg.max_context, n_ctx_train)`; prefill with `llama_batch_get_one`, then decode token by token; EOG through `llama_vocab_is_eog`; stop strings; `llama_token_to_piece` + an incremental UTF-8 assembler. *(B: fix every Code Principles finding; replace the parser with `gguf.h`; RMSNorm, causal mask, GQA, `ggml_rope_ext`, `ggml_backend_sched`.)*
2. Sampler: `llama_sampler_chain` (penalties → top-k → top-p → min-p → temp → dist(seed)); greedy at `temperature == 0`.
3. Chat: `/v1/chat/completions` applying the model's chat template (`llama_chat_apply_template` or llama.cpp's Jinja support).
4. Real streaming: `set_chunked_content_provider`, one SSE event per piece, `stream:false` JSON mode, stop when `!sink.is_writable()` (client disconnected).
5. Tests: unit tests (sampler params, UTF-8 assembler, request validation, config parsing) + integration tests in CI with `ggml-org/models/tinyllamas/stories260K.gguf` (1.2 MB) for deterministic greedy output, prompt → N tokens, EOS stop, context overflow → 400.

### P2 — Sandbox that actually isolates (→ ~6.5/10)

Architecture: **broker** (unprivileged, owns the network/TLS/auth/rate limiting) ↔ **worker** (inference only, no network) over a `socketpair` with a length-prefixed JSON/CBOR protocol and per-message size limits.

1. Broker opens the model **fd** (and, if GPU, `/dev/nvidia*`, `/dev/dri/*`), verifies `model_sha256` on that fd (EVP), and passes it with `SCM_RIGHTS`, or keeps it open across `clone`.
2. Worker: `clone3(CLONE_NEWUSER|NEWNS|NEWPID|NEWIPC|NEWUTS|NEWNET|NEWCGROUP)`; the parent writes `setgroups=deny`, `uid_map`, `gid_map`; then mount tmpfs → `syscall(SYS_pivot_root, ".", ".")` → `umount2(".", MNT_DETACH)` (fatal) → `chdir("/")`; `close_range(3, ~0U, CLOSE_RANGE_UNSHARE)` except the needed fds; `PR_SET_DUMPABLE 0`; drop the bounding set and ambient caps; `PR_SET_NO_NEW_PRIVS`; `setrlimit` (AS, NOFILE, NPROC, CORE=0, FSIZE=0).
3. Landlock (ABI ≥ 1; kernel ≥ 5.13): no filesystem access at all (best-effort, with a log line when unavailable).
4. Seccomp: default `SCMP_ACT_KILL_PROCESS`; allow-list from `strace -f` of the real worker on CPU and CUDA; `clone` only with thread flags; **no** `socket`/`connect`/`execve`; `ioctl` only when `sandbox.gpu=true`, documented as a weaker profile. Log SIGSYS kills in the broker.
5. cgroup v2: use a delegated subtree (systemd unit with `Delegate=yes`, or `--cgroup-parent`); set `memory.max`, `memory.swap.max`, `pids.max`, `cpu.max` before moving the worker in; fail closed if `sandbox.require_limits=true`; `rmdir` on exit.
6. Tests: a test-only escape-probe request (compiled only in test builds) that tries `open("/etc/passwd")`, `socket()` and `execve()` → expects a kill/EPERM; plus a CI job on an Ubuntu runner with userns enabled.
7. Docs: threat model (what the sandbox protects against, GPU caveats, running under Docker/K8s with the needed `securityContext`/seccomp profile).

### P3 — Hardened, OpenAI-compatible API (→ ~7.5/10)

1. Endpoints: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/healthz`, `/readyz`, `/metrics` (Prometheus; auth-gated); `usage` token counts; `stop`, `seed`, `max_tokens` (alias `max_new_tokens`), `stream`.
2. Auth: `Authorization: Bearer` + `X-API-Key`; keys from `LISA_API_KEYS` / `api_keys_file`, stored as SHA-256 and compared with `CRYPTO_memcmp`; refuse weak or default keys on non-loopback binds.
3. Limits: `set_payload_max_length` (1 MiB default), read/write/idle timeouts, max prompt tokens, bounded queue → 429 + `Retry-After`, per-key token bucket.
4. TLS: `SSLServer` (TLS 1.2+ only, modern ciphers), optional mTLS with `client_ca_file`; HSTS when TLS is on.
5. Errors: OpenAI-style error JSON with no internal details; request ids in the logs.
6. CLI (`--config`, `--model`, `--host`, `--port`, `--check-config`, `--version`) + `LISA_*` env overrides + strict config validation (unknown keys → error).
7. Graceful shutdown: signal → stop accepting → drain (deadline) → `svr.stop()` → worker SIGTERM → reap.
8. Fuzzing: libFuzzer targets for request JSON → validated params and for the broker↔worker protocol decoder; nightly CI run (OSS-Fuzz-style).

### P4 — Performance at SOTA level (→ ~8.5/10)

1. Continuous batching: N slots (`n_seq_max`), one `llama_batch` with multiple `seq_id`s per step; prefix/prompt cache reuse (`llama_memory_seq_*`).
2. Flash attention on, quantised KV cache (`q8_0`) as options; mmap zero-copy (default in libllama); multi-GPU split mode.
3. Optional speculative decoding (draft model) and grammar/JSON-schema constrained output.
4. Benchmarks: a `bench/` script comparing tokens/s and TTFT against `llama-server` on the same GGUF; publish the numbers in the README. Goal: ≥ 90 % of llama-server throughput with the sandbox on.

### P5 — Project hygiene (needed for a credible 8+)

1. Honest README: remove the claims that aren't true yet; add a threat model, an architecture diagram, a support matrix (archs, GPUs, kernels) and a config reference.
2. `SECURITY.md` (reporting process), `CONTRIBUTING.md`, `CHANGELOG.md`, SemVer tags, release workflow (static Linux x86_64/arm64 binaries + CUDA image, SBOM via syft, cosign signatures, provenance).
3. Docker: rename to `Dockerfile`, non-root user, runtime libs, pinned digests, `HEALTHCHECK`, `0.0.0.0` bind inside the image, `docker-compose.yml` with GPU and a documented seccomp/userns setup.
4. Coverage gate (≥ 80 % on first-party code via gcovr/llvm-cov) in CI.

### Expected score trajectory

| After | Build | Security | Concurrency | Principles | Deps | Observability | Lifecycle | Quality | Dead code | **Overall** |
|---|---|---|---|---|---|---|---|---|---|---|
| Today | 0 | 1 | 4 | 2 | 3 | 3 | 2 | 2 | 3 | **2.2** (SOTA 1.5) |
| P0 | 8 | 2 | 4 | 4 | 7 | 3 | 3 | 5 | 8 | ~4.0 (capped by security) |
| P1 | 8 | 3 | 5 | 7 | 8 | 4 | 4 | 7 | 9 | ~5.0 |
| P2 | 8 | 7 | 6 | 7 | 8 | 5 | 6 | 7 | 9 | ~6.6 |
| P3 | 9 | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 9 | ~8.1 |
| P4+P5 | 9 | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8.5 | 9 | **~8.7** |

### Priority-ordered action list

1. [CRITICAL] Choose the engine (Option A) and pin llama.cpp; fix the swapped `server.hpp`/`server.cpp`; delete `model_loader.cpp`; rewrite CMake → green CI with ASan/UBSan.
2. [CRITICAL] Remove or replace the custom GGUF parser (an out-of-bounds read on untrusted input).
3. [CRITICAL] Rewrite the generation loop (prefill/decode, KV cache, EOS, sampler) and the streaming endpoint.
4. [CRITICAL] Redesign the sandbox as broker/worker with id maps, the model fd opened before pivot, a correct pivot sequence, a real seccomp allow-list, Landlock and rlimits.
5. [HIGH] Upgrade httplib to ≥ 0.60.1; input validation; body, time and concurrency limits; constant-time auth; no default key.
6. [HIGH] Wire `model_sha256`; fix the Docker runtime (libs, bind address, config path, non-root).
7. [HIGH] Tests: unit + tiny-GGUF integration + sandbox escape probes + fuzzers.
8. [MEDIUM] OpenAI-compatible API, `/readyz`, `/metrics`, log levels, request ids, graceful shutdown, CLI.
9. [MEDIUM] Continuous batching, prompt cache, flash attention, KV quantisation, benchmarks.
10. [LOW] README honesty pass, SECURITY/CONTRIBUTING/CHANGELOG, signed releases, SBOM.

---

## Sources Consulted

- GGUF specification — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp `include/llama.h` and `examples/simple/simple.cpp` (via Context7 `/ggml-org/llama.cpp`)
- cpp-httplib releases — https://github.com/yhirose/cpp-httplib/releases (latest v0.60.1)
- CVE-2026-46527 (cpp-httplib < 0.44.0) — https://www.sentinelone.com/vulnerability-database/cve-2026-46527/
- CVE-2026-34441 (fixed in 0.40.0) — https://cvefeed.io/vuln/detail/CVE-2026-34441
- user_namespaces(7) — https://man.archlinux.org/man/user_namespaces.7.en
- passt sandbox ordering (uid_map → setgroups deny → gid_map) — https://git.phyllo.me/External/passt/src/tag/2022_09_23.d6f865a/isolation.c
- PR_SET_NO_NEW_PRIVS before seccomp/Landlock — https://forge.lthn.ai/core/core-agent-ide/commit/2259031d64644f74ca757cc535db61c2d47c378c
- llama.cpp server features (OpenAI API, continuous batching) — https://www.mintlify.com/ggml-org/llama.cpp/inference/server
- Hugging Face Hub: `TheBloke/Llama-2-7B-Chat-GGUF` file listing (no vocab.json/merges.txt); `ggml-org/models/tinyllamas/stories260K.gguf`

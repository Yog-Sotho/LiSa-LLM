# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0]

First tagged release. The code base was rebuilt following the
[codebase audit](docs/project/codebase_audit.md); nothing of the earlier custom engine
remains.

### Added

- **Engine**: llama.cpp's `libllama` (pinned and verified) replaces the custom GGUF parser
  and transformer; any GGUF model llama.cpp supports, every quantisation type, optional CUDA
  offload.
- **OpenAI-compatible API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
  streaming (SSE), `usage` with cached-token details, `finish_reason`, OpenAI error format;
  54 built-in chat templates with forged-special-token protection; sampling controls
  (temperature, top-p/k, min-p, penalties, seed, stop sequences); JSON mode.
- **Sandbox** (on by default): the model is parsed only in a worker process with its own
  user/mount/network/PID/IPC/UTS/cgroup namespaces, an empty read-only root holding only
  the model, no capabilities, Landlock, a two-stage seccomp allow-list and resource limits;
  crashed workers are logged and restarted; optional `model_sha256` pinning.
- **Performance**: continuous batching (`parallel_slots`), a prompt cache that reuses and
  copies prefixes between slots, flash attention and quantised KV cache options; faster than
  `llama-server` under concurrent load in the published benchmark.
- **Hardened API**: multiple API keys stored as SHA-256 digests, per-key rate limits, a
  bounded first-come-first-served queue, timeouts, TLS 1.2+/1.3 with AEAD suites only,
  optional mutual TLS, HSTS, strict config validation, `LISA_*` environment overrides.
- **Operations**: `/healthz`, `/readyz`, Prometheus `/metrics`, JSON access logs with request
  ids, graceful drain on SIGTERM, `--check-config`, `--hash-key`, `--health-check`.
- **Distribution**: portable Linux binaries (x86-64-v3 and arm64; every dependency but glibc
  linked in), multi-arch CPU and CUDA container images, SPDX SBOMs, Sigstore signatures and
  SLSA provenance for every artifact.
- **Quality**: unit, end-to-end, TLS, sandbox-escape and hostile-worker tests under
  ASan/UBSan and TSan; libFuzzer targets for the request and worker-reply decoders;
  clang-tidy, cppcheck, CodeQL and a coverage gate in CI.

### Security

- The earlier sandbox did not isolate anything (missing UID maps, model opened after
  `pivot_root`, wrong namespace order) and the custom GGUF parser had out-of-bounds reads on
  untrusted input; both are gone. See the audit report for the full list.

[1.0.0]: https://github.com/Yog-Sotho/LiSa-LLM/releases/tag/v1.0.0

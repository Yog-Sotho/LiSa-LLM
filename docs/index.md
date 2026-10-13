# 🛡️ LiSa-LLM — Documentation

Welcome! These pages take you from "I have a GGUF file" to a fast, locked-down, OpenAI-compatible
model server you can put on a network without sweating.

---

## What is LiSa-LLM?

LiSa-LLM is a small inference server for GGUF language models. You give it a model file, it gives
you the same API OpenAI's apps and SDKs already speak (`/v1/chat/completions`), so anything that
works with OpenAI works with your own model on your own hardware.

What makes it different is that **it doesn't trust the model file**. A GGUF file is a big blob of
binary data that a parser has to pick apart. A booby-trapped one could exploit a bug in that parser.
So LiSa never opens the model in the process that talks to the network. A separate worker process
does it, after locking itself into a padded cell: no files, no network, no other processes, no
privileges, and a system-call filter that kills it if it tries anything else.

And it is quick about it: continuous batching and a prompt cache make it faster than llama.cpp's own
`llama-server` when several clients talk to it at once, sandbox included.

---

## Who is this for?

- **Self-hosters** who want a private ChatGPT-style endpoint for their apps, agents or tools
- **Teams** putting a model on a shared network who need keys, rate limits, TLS and metrics
- **Security-minded people** who download models from the internet and would rather not run
  someone else's binary blob next to their SSH keys
- **The rest of the family**: [Brainbrew](https://github.com/Yog-Sotho/Brainbrew) can use LiSa as
  its teacher model, and GGUF files exported by
  [LLM Fine-Tuner](https://github.com/Yog-Sotho/LLM-fine-tuner) load straight into it

---

## Documentation map

| Guide | What you'll learn |
|---|---|
| [Installation](installation.md) | Release binaries, container images, building from source, verifying downloads |
| [Quick start](quick-start.md) | Your first answer from your own model in 5 minutes |
| [Using the API](api.md) | Chat and completion requests, streaming, JSON mode, errors, limits |
| [Configuration](configuration.md) | Every setting, its default and its environment variable |
| [Deployment](deployment.md) | Docker, Compose, TLS and mTLS, health checks, Prometheus, systemd |
| [Security and the sandbox](security.md) | What the sandbox does, what it needs, and how to expose the server safely |
| [Performance](performance.md) | Benchmarks, how batching and the prompt cache work, tuning |
| [Troubleshooting](troubleshooting.md) | The errors you are most likely to meet, and their fixes |
| [FAQ](faq.md) | Quick answers |

For contributors and the curious:

| Document | What's inside |
|---|---|
| [Threat model](project/threat_model.md) | The attackers the design assumes, every defence layer, known limits |
| [Codebase audit](project/codebase_audit.md) | The audit that started the rebuild, and the progress log of every phase |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Building, testing, fuzzing, style, releasing |
| [SECURITY.md](../SECURITY.md) | How to report a vulnerability privately |
| [CHANGELOG.md](../CHANGELOG.md) | What changed in each release |

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Linux, glibc 2.35+ (Ubuntu 22.04, Debian 12, Fedora 36, RHEL 10) | Kernel 5.13+ for Landlock |
| CPU | x86-64 with AVX2/FMA/F16C (Haswell, 2013+) or 64-bit Arm | More cores = more tokens per second |
| RAM | The model file's size plus its KV cache | Enough for every slot's KV cache too (see [Performance](performance.md#memory)) |
| GPU | None | NVIDIA, compute capability 7.5–9.0 for the CUDA image (sandbox off in GPU mode) |
| Sandbox | Unprivileged user namespaces and seccomp | On by default; see [Security](security.md#what-the-sandbox-needs) |

> **macOS or Windows?** Not supported: the sandbox is built on Linux kernel features. Run the
> container image in Docker Desktop or a Linux VM instead.

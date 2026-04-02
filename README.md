<div align="center">
  <img src="assets/Logo_LiSa.png" alt="LiSa Logo" width="420" style="margin-bottom: 20px;">
<h1># LiSa-LLM – Lightweight, Sandboxed LLM Inference Server</h1>

> *“Because your cat‑sized model deserves a lion‑sized sandbox.”*

<p><strong>LiSa-LLM is a **production‑grade**, self‑contained C++20 inference server for custom transformer models, using **GGML** for tensor computations. It features a **hardened Linux sandbox** (namespaces, seccomp, cgroup v2, pivot_root), **API key authentication**, **TLS support**, **streaming Server‑Sent Events**, and a **custom binary model format** (no llama.cpp required). Perfect for edge deployments or whenever you need to run untrusted models without losing sleep.
</strong></p>

  <p>
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    <img src="https://img.shields.io/badge/C++-ready-blue.svg" alt="C++ Ready">
    <img src="https://img.shields.io/badge/GPU-Required-green.svg" alt="GPU Required">
    <\p>
      <a href="https://github.com/sponsors/Yog-Sotho" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Sponsor❤️-30363D.svg?logo=githubsponsors&logoColor=EA4AAA" alt="Sponsor on GitHub">
  </a>
<\div>

<h2>## Features

- 🧠 **Pure GGML** transformer with multi‑head attention, RoPE, and KV cache (skeleton ready for your implementation).
- 🔒 **Sandboxed execution** – the model runs in a separate process with:
  - User, mount, PID, and network namespaces
  - `pivot_root` into an empty tmpfs
  - Strict seccomp‑bpF whitelist (only 30 syscalls)
  - cgroup v2 memory limit
  - Drop to `nobody` privileges
- 🌐 **HTTP API** with optional TLS (mTLS ready) and API key.
- 📡 **Real‑time streaming** via Server‑Sent Events.
- 📦 **No external ML frameworks** – just GGML, C++20, and a few header‑only libs.
- 🔁 **Model integrity checks** (SHA‑256) before loading.

<h2>## Build & Run<\h2>

### Dependencies

- CMake ≥ 3.20
- C++20 compiler (GCC 11+, Clang 14+)
- OpenSSL (for TLS and SHA‑256)
- libseccomp (Linux only, for sandbox)
- yaml-cpp
- (Optional) pthread, dl, rt

All other libraries (`ggml`, `cpp-httplib`, `nlohmann/json`) are bundled in `third_party/`.

<h2>### Compile<\h2>

```bash
mkdir build && cd build
cmake .. -DENABLE_TLS=ON -DENABLE_SANDBOX=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Made with love ❤️
Yog-Sotho

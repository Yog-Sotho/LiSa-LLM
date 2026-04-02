# LiSa-LLM – Lightweight, Sandboxed LLM Inference Server

> *“Because your cat‑sized model deserves a lion‑sized sandbox.”*

LiSa-LLM is a **production‑grade**, self‑contained C++20 inference server for custom transformer models, using **GGML** for tensor computations. It features a **hardened Linux sandbox** (namespaces, seccomp, cgroup v2, pivot_root), **API key authentication**, **TLS support**, **streaming Server‑Sent Events**, and a **custom binary model format** (no llama.cpp required). Perfect for edge deployments or whenever you need to run untrusted models without losing sleep.

## Features

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

## Build & Run

### Dependencies

- CMake ≥ 3.20
- C++20 compiler (GCC 11+, Clang 14+)
- OpenSSL (for TLS and SHA‑256)
- libseccomp (Linux only, for sandbox)
- yaml-cpp
- (Optional) pthread, dl, rt

All other libraries (`ggml`, `cpp-httplib`, `nlohmann/json`) are bundled in `third_party/`.

### Compile

```bash
mkdir build && cd build
cmake .. -DENABLE_TLS=ON -DENABLE_SANDBOX=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
Made with love ❤️
Yog-Sotho

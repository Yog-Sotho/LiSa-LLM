# 📦 Installation

There are three ways to get LiSa-LLM running. Pick the first one that fits:

| Way | Best for | Needs |
|---|---|---|
| [Release binary](#release-binaries) | Bare-metal Linux servers and desktops | Linux with glibc 2.35+ |
| [Container image](#container-images) | Docker, Compose, Kubernetes, macOS/Windows via Docker Desktop | Docker or Podman |
| [Build from source](#build-from-source) | NVIDIA GPUs other than the CUDA image's, tuning for your exact CPU, hacking | A C++20 compiler, CMake |

Then grab a model ([Getting a model](#getting-a-model)) and head to the [quick start](quick-start.md).

---

## Release binaries

Every release ships a ready-to-run archive for Linux **x86_64** and **aarch64** (64-bit Arm):

```bash
VERSION=1.0.0 ARCH=x86_64        # or aarch64
curl -LO https://github.com/Yog-Sotho/LiSa-LLM/releases/download/v$VERSION/lisa-$VERSION-linux-$ARCH.tar.gz
tar xzf lisa-$VERSION-linux-$ARCH.tar.gz
./lisa-$VERSION-linux-$ARCH/bin/lisa --version
```

What's inside:

| Path | What it is |
|---|---|
| `bin/lisa` | The server. Everything but glibc is linked in: no libraries to install |
| `config.yaml` | The sample configuration, every option commented |
| `doc/` | README, changelog, security policy, threat model |
| `THIRD-PARTY-NOTICES.md`, `licenses/` | Licences of everything linked into the binary |

- **Which Linux?** Any with glibc 2.35 or newer: Ubuntu 22.04, Debian 12, Fedora 36, RHEL 10
  and their successors. Check yours with `ldd --version`.
- **Which CPU?** The x86_64 build targets x86-64-v3: AVX2, FMA and F16C (Intel Haswell or AMD
  Excavator, 2013, or newer). On an older CPU it stops at startup with a clear message instead of
  crashing; [build from source](#build-from-source) to run there.
- Put `bin/lisa` anywhere on your `PATH`, e.g. `sudo install -m 755 lisa-*/bin/lisa /usr/local/bin/`.

Verify what you downloaded before running it: see [Verifying a release](#verifying-a-release).

---

## Container images

Images live on the GitHub Container Registry:

| Image | Platforms | What it runs |
|---|---|---|
| `ghcr.io/yog-sotho/lisa-llm:1.0.0` (also `:latest`) | linux/amd64, linux/arm64 | CPU inference |
| `ghcr.io/yog-sotho/lisa-llm:1.0.0-cuda` (also `:latest-cuda`) | linux/amd64 | NVIDIA GPUs, compute capability 7.5–9.0 (Turing to Hopper) |

```bash
KEY="$(openssl rand -hex 24)"; echo "Your API key: $KEY"
docker run --rm -p 8080:8080 -v ./models:/models:ro \
  -e LISA_API_KEYS="$KEY" -e LISA_API_KEY= \
  ghcr.io/yog-sotho/lisa-llm:1.0.0
```

- The model is expected at `/models/model.gguf` (mount a folder that has one, or set
  `LISA_MODEL_PATH`).
- `LISA_API_KEY=` clears the image's placeholder key. The server listens on `0.0.0.0` inside the
  container and refuses to start with a weak or placeholder key there.
- The images run as an unprivileged user (UID 10001) and have a `HEALTHCHECK` that asks `/readyz`.
- In the plain image the **sandbox is off**, because Docker's default security profiles block the
  user namespaces it needs. To turn it on, see [Deployment → Sandbox in a container](deployment.md#sandbox-in-a-container),
  or use the `lisa-sandboxed` Compose service.

For Docker Compose, Kubernetes and GPU setups, see [Deployment](deployment.md).

---

## Build from source

### 1. Install the tools

- CMake 3.28 or newer, and Ninja (or Make)
- A C++20 compiler: GCC 13+ or Clang 18+
- OpenSSL 3 (model SHA-256 and TLS), libseccomp (the sandbox), yaml-cpp (built from source if missing)
- Optional: the CUDA toolkit for NVIDIA GPU offload

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

llama.cpp, cpp-httplib, nlohmann/json and Catch2 are downloaded by CMake while it configures, each
pinned to a release tag whose commit (or archive SHA-256) is checked. Building offline? Point
`FETCHCONTENT_SOURCE_DIR_<NAME>` at local checkouts.

### 2. Build

```bash
git clone https://github.com/Yog-Sotho/LiSa-LLM.git
cd LiSa-LLM
cmake --preset release
cmake --build --preset release
ctest --preset release            # optional: the unit tests
./build/release/lisa --version
```

| Preset | What you get |
|---|---|
| `release` | Optimised build tuned for this machine's CPU |
| `portable` | The release-archive build: static dependencies, runs on other machines |
| `cuda` | NVIDIA GPU offload (needs the CUDA toolkit) |
| `dev`, `asan-ubsan`, `tsan`, `coverage`, `fuzz` | For contributors: see [CONTRIBUTING.md](../CONTRIBUTING.md) |

Main CMake options:

| Option | Default | Meaning |
|---|---|---|
| `LISA_CUDA` | `OFF` | CUDA offload |
| `LISA_SANDBOX` | `ON` | The sandboxed worker (needs libseccomp) |
| `LISA_TLS` | `ON` | HTTPS support (needs OpenSSL) |
| `LISA_NATIVE` | `ON` | Tune for the build machine's CPU |
| `LISA_HARDENING` | `ON` | Stack protector and clash protection, `_FORTIFY_SOURCE=3`, CET, full RELRO, PIE |
| `LISA_PORTABLE` | `OFF` | Link dependencies statically for distribution |
| `LISA_SANITIZERS` | empty | e.g. `address,undefined` |

### 3. Build the container yourself

```bash
docker build -t lisa-llm .                                   # CPU
docker build -t lisa-llm:cuda \
  --build-arg BUILD_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  --build-arg RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
  --build-arg LISA_CUDA=ON .                                 # GPU
```

The `lisa-cuda` service in [`docker-compose.yml`](../docker-compose.yml) has the same build
arguments, with the base images pinned by digest.

---

## Getting a model

LiSa runs any GGUF model llama.cpp supports: Llama, Mistral, Qwen, Gemma, Phi, DeepSeek and many
more, in every quantisation. The tokenizer and chat template come from the file itself.

```bash
mkdir -p models
# Tiny and fast, good for a first try (≈ 490 MB)
curl -L -o models/model.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
# A more capable chat model (≈ 2 GB)
curl -L -o models/model.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Fine-tuned your own with [LLM Fine-Tuner](https://github.com/Yog-Sotho/LLM-fine-tuner)? Its GGUF
export loads as is.

**Tip:** pin the exact file you meant to serve. Put its checksum in the config and LiSa refuses to
start on anything else:

```bash
sha256sum models/model.gguf       # → model_sha256: "<the 64 hex digits>"
```

---

## Verifying a release

Release archives and images are signed with [Sigstore](https://www.sigstore.dev/) by the release
workflow. The signing is keyless: the certificate names this repository's workflow and the tag,
so nobody can sign with it from anywhere else. Every artifact also carries SLSA build provenance
and an SPDX SBOM (the list of everything inside).

You need [cosign](https://docs.sigstore.dev/cosign/system_config/installation/) and, for the
provenance checks, the [GitHub CLI](https://cli.github.com/).

```bash
VERSION=1.0.0 FILE=lisa-1.0.0-linux-x86_64.tar.gz
IDENTITY="https://github.com/Yog-Sotho/LiSa-LLM/.github/workflows/release.yml@refs/tags/v$VERSION"
ISSUER=https://token.actions.githubusercontent.com

# Archive: checksum, signature, provenance
sha256sum --check --ignore-missing SHA256SUMS
cosign verify-blob "$FILE" --bundle "$FILE.sigstore.json" \
  --certificate-identity "$IDENTITY" --certificate-oidc-issuer "$ISSUER"
gh attestation verify "$FILE" --repo Yog-Sotho/LiSa-LLM

# Image: signature, provenance, SBOM
cosign verify ghcr.io/yog-sotho/lisa-llm:$VERSION \
  --certificate-identity "$IDENTITY" --certificate-oidc-issuer "$ISSUER"
gh attestation verify oci://ghcr.io/yog-sotho/lisa-llm:$VERSION --repo Yog-Sotho/LiSa-LLM
cosign verify-attestation --type spdxjson ghcr.io/yog-sotho/lisa-llm:$VERSION \
  --certificate-identity "$IDENTITY" --certificate-oidc-issuer "$ISSUER" > /dev/null
```

`SHA256SUMS`, the `.sigstore.json` bundles and the `.spdx.json` SBOMs are attached to each
[GitHub release](https://github.com/Yog-Sotho/LiSa-LLM/releases).

---

## Support matrix

| | Status |
|---|---|
| Linux x86_64 | Release binary (x86-64-v3) and images; developed and benchmarked here |
| Linux aarch64 | Release binary and image, built and unit-tested on GitHub's Arm runners |
| Kernel | 5.13+ recommended (Landlock); the sandbox needs unprivileged user namespaces and seccomp |
| NVIDIA GPUs | CUDA image for compute capability 7.5–9.0; any CUDA GPU when built from source; GPU mode runs without the sandbox |
| AMD/Intel GPUs, Apple Metal | Not yet (llama.cpp's Vulkan, ROCm and Metal backends aren't wired up) |
| macOS, Windows, BSD | Not supported natively (the sandbox is Linux-only); use the container image |

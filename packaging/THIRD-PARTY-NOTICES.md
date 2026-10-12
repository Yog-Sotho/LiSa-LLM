# Third-party notices

LiSa-LLM is MIT-licensed (see `LICENSE`). Release binaries and container images also
contain the following components; their full licence texts are in `licenses/` (the
release archive) or `/usr/share/doc/lisa-llm/licenses/` (the container image).

| Component | Licence | Source |
|---|---|---|
| llama.cpp / ggml (`libllama`) | MIT | https://github.com/ggml-org/llama.cpp |
| cpp-httplib | MIT | https://github.com/yhirose/cpp-httplib |
| nlohmann/json | MIT | https://github.com/nlohmann/json |
| yaml-cpp | MIT | https://github.com/jbeder/yaml-cpp |
| OpenSSL 3 | Apache-2.0 | https://www.openssl.org/ |
| libseccomp | LGPL-2.1 | https://github.com/seccomp/libseccomp |
| GCC runtime (libstdc++, libgcc) | GPL-3.0 with the GCC Runtime Library Exception | https://gcc.gnu.org/ |

Exact versions and commits are listed in the SBOM shipped with each release
(`*.spdx.json`).

**libseccomp (LGPL-2.1)** is linked statically into the portable binaries. To use a
modified libseccomp, rebuild LiSa-LLM from the corresponding release tag (the complete
source is public under the MIT licence) against your version of the library; see
`CONTRIBUTING.md` for build instructions. The container images link it dynamically from
the distribution package.

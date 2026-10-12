# syntax=docker/dockerfile:1
#
# CPU image (default; amd64 and arm64):
#   docker build -t lisa-llm .
# CUDA image (amd64):
#   docker build -t lisa-llm:cuda \
#     --build-arg BUILD_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a \
#     --build-arg RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:92906d87596d638d35015c6353053121bd299d25943b875763321653884ba924 \
#     --build-arg LISA_CUDA=ON .
# Run (see docker-compose.yml for the sandboxed and GPU variants):
#   docker run --rm -p 8080:8080 -v ./models:/models:ro \
#     -v ./my-config.yaml:/etc/lisa/config.yaml:ro lisa-llm
#
# Base images are pinned by digest (multi-arch index); bump them deliberately.

ARG BUILD_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
ARG RUNTIME_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# ---- Build -----------------------------------------------------------------------------
FROM ${BUILD_IMAGE} AS builder
ARG LISA_CUDA=OFF
# GPU generations compiled for CUDA builds: Turing, Ampere, Ada, Hopper.
ARG CUDA_ARCHITECTURES="75;80;86;89;90"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build pkg-config \
        libssl-dev libyaml-cpp-dev libseccomp-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLISA_NATIVE=OFF \
        -DLISA_CUDA=${LISA_CUDA} \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DLISA_BUILD_TESTS=OFF \
    && cmake --build build --target lisa \
    && strip build/lisa \
    && packaging/collect-licenses.sh build /licenses

# ---- Runtime ---------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
ARG VERSION=dev
LABEL org.opencontainers.image.title="LiSa-LLM" \
      org.opencontainers.image.description="Sandboxed, OpenAI-compatible LLM inference server for GGUF models" \
      org.opencontainers.image.source="https://github.com/Yog-Sotho/LiSa-LLM" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libseccomp2 libssl3t64 libyaml-cpp0.8 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 lisa \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin lisa
COPY --from=builder /src/build/lisa /usr/local/bin/lisa
COPY --from=builder /licenses /usr/share/doc/lisa-llm/licenses
COPY LICENSE packaging/THIRD-PARTY-NOTICES.md /usr/share/doc/lisa-llm/
COPY docker/config.yaml /etc/lisa/config.yaml
USER 10001:10001
WORKDIR /models
EXPOSE 8080
# Ready = model loaded and accepting requests (/readyz). Large models load slowly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/usr/local/bin/lisa", "--config", "/etc/lisa/config.yaml", "--health-check"]
ENTRYPOINT ["/usr/local/bin/lisa", "--config", "/etc/lisa/config.yaml"]

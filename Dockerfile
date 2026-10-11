# syntax=docker/dockerfile:1
#
# CPU image (default):
#   docker build -t lisa-llm .
# CUDA image:
#   docker build -t lisa-llm:cuda \
#     --build-arg BUILD_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04 \
#     --build-arg RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
#     --build-arg LISA_CUDA=ON .
# Run:
#   docker run --rm -p 8080:8080 -v ./models:/models:ro \
#     -v ./my-config.yaml:/etc/lisa/config.yaml:ro lisa-llm

ARG BUILD_IMAGE=ubuntu:24.04
ARG RUNTIME_IMAGE=ubuntu:24.04

# ---- Build -----------------------------------------------------------------------------
FROM ${BUILD_IMAGE} AS builder
ARG LISA_CUDA=OFF
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
        -DLISA_BUILD_TESTS=OFF \
    && cmake --build build --target lisa \
    && strip build/lisa

# ---- Runtime ---------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libseccomp2 libssl3t64 libyaml-cpp0.8 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 lisa \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin lisa
COPY --from=builder /src/build/lisa /usr/local/bin/lisa
COPY docker/config.yaml /etc/lisa/config.yaml
USER 10001:10001
WORKDIR /models
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/lisa", "--config", "/etc/lisa/config.yaml"]

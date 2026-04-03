# Stage 1: Builder
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config libyaml-cpp-dev libseccomp-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -B build -S . -DENABLE_CUDA=ON -DENABLE_TLS=ON -DENABLE_SANDBOX=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

# Stage 2: Runtime (minimal, sandbox-hardened)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/build/lisa /usr/local/bin/lisa
COPY config.yaml /etc/lisa/config.yaml
WORKDIR /models
EXPOSE 8080
CMD ["lisa"]

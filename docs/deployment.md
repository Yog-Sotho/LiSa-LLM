# 🚢 Deployment

How to run LiSa-LLM for real: in containers, behind TLS, watched by your monitoring, and stopped
without dropping anyone's request.

---

## Docker

```bash
KEY="$(openssl rand -hex 24)"; echo "Your API key: $KEY"
docker run -d --name lisa -p 8080:8080 \
  -v ./models:/models:ro \
  -e LISA_API_KEYS="$KEY" -e LISA_API_KEY= \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  ghcr.io/yog-sotho/lisa-llm:1.0.0
```

- The image expects the model at `/models/model.gguf` and reads `/etc/lisa/config.yaml`; mount
  your own config there, or use `LISA_*` environment variables.
- It runs as UID 10001 and needs nothing writable but `/tmp`, so `--read-only` and
  `--cap-drop ALL` cost nothing.
- `docker ps` shows `healthy` once the model is loaded: the image's `HEALTHCHECK` runs
  `lisa --health-check`, which asks `/readyz`. Big models take a while; the check allows two
  minutes before it counts failures.

### Sandbox in a container

The sandbox creates user namespaces, which Docker's default seccomp and AppArmor profiles forbid.
Relax **only Docker's own profiles**; the worker then installs a far stricter seccomp filter and
Landlock rules of its own:

```bash
docker run -d --name lisa -p 8080:8080 -v ./models:/models:ro \
  -e LISA_API_KEYS="$KEY" -e LISA_API_KEY= -e LISA_SANDBOX_ENABLED=true \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  ghcr.io/yog-sotho/lisa-llm:1.0.0
docker logs lisa | grep "Sandboxed worker ready"
```

On Ubuntu 23.10+ hosts, also allow unprivileged user namespaces on the host:
`sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`. CI runs exactly this setup on every
change.

### GPU

```bash
docker run -d --gpus all -p 8080:8080 -v ./models:/models:ro \
  -e LISA_API_KEYS="$KEY" -e LISA_API_KEY= \
  ghcr.io/yog-sotho/lisa-llm:1.0.0-cuda
```

Needs an NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
GPU drivers need device files and `ioctl` calls that the sandbox can't allow, so GPU mode runs
with the sandbox off: only serve models you trust there, and pin them with `model_sha256`.

---

## Docker Compose

[`docker-compose.yml`](https://github.com/Yog-Sotho/LiSa-LLM/blob/main/docker-compose.yml) has three ready-made services, all with a read-only
root, no capabilities and `no-new-privileges`:

```bash
echo "LISA_API_KEYS=$(openssl rand -hex 24)" > .env      # or a sha256: entry (lisa --hash-key)
cp /path/to/your-model.gguf models/model.gguf

docker compose up -d lisa                       # CPU
docker compose up -d lisa-sandboxed             # CPU, model parsed in the sandboxed worker
docker compose --profile gpu up -d lisa-cuda    # NVIDIA GPU
```

---

## TLS and mutual TLS

```yaml
enable_tls: true
cert_file: "/etc/lisa/tls/server.crt"     # PEM, full chain
key_file: "/etc/lisa/tls/server.key"
tls_min_version: "1.3"                    # or "1.2" (forward-secret AEAD suites only)
client_ca_file: "/etc/lisa/tls/clients-ca.crt"   # optional: require client certificates
```

With TLS on, responses carry `Strict-Transport-Security`. With `client_ca_file`, only clients with
a certificate signed by that CA can even complete the handshake; API keys are still checked on
top.

A quick self-signed setup for testing:

```bash
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 30 \
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost" \
  -keyout server.key -out server.crt
curl --cacert server.crt https://localhost:8080/readyz
```

### Behind a reverse proxy

Terminating TLS at nginx, Caddy or a load balancer works too. Two things matter: turn off response
buffering so streamed tokens arrive as they are generated, and allow long responses.

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_buffering off;            # stream tokens as they come
    proxy_read_timeout 300s;        # long generations
    proxy_set_header X-Request-Id $request_id;
}
```

Keep LiSa itself on `127.0.0.1` then. Behind a proxy every request comes from the proxy's address,
so rate limits are per key: give each client its own key.

---

## Health checks

| Check | Use it for | Answers |
|---|---|---|
| `GET /healthz` | Liveness: restart if this fails | `200` while the process runs |
| `GET /readyz` | Readiness: send traffic only when this passes | `200` when ready; `503` while draining or when the worker keeps crashing |
| `lisa --health-check` | Containers without curl (the image's `HEALTHCHECK`) | Exit 0 when `/readyz` says ready, 1 otherwise |

`--health-check` reads the same configuration as the server (address, port, TLS), so pass it the
same `--config` and environment.

Kubernetes probes (the container port is 8080):

```yaml
livenessProbe:
  httpGet: {path: /healthz, port: 8080}
  periodSeconds: 10
readinessProbe:
  httpGet: {path: /readyz, port: 8080}
  periodSeconds: 5
startupProbe:                      # loading a big model takes a while
  httpGet: {path: /readyz, port: 8080}
  periodSeconds: 5
  failureThreshold: 60
```

Use `scheme: HTTPS` in the probes when `enable_tls` is on.

---

## Monitoring

`GET /metrics` serves Prometheus metrics (it needs an API key when keys are configured):

```yaml
scrape_configs:
  - job_name: lisa
    authorization: {credentials_file: /etc/prometheus/lisa-key}
    static_configs: [{targets: ["lisa:8080"]}]
```

| Metric | Type | What it tells you |
|---|---|---|
| `lisa_ready` | gauge | 1 when ready, 0 while draining or when the worker is down |
| `lisa_requests_running`, `lisa_requests_queued` | gauge | Slots in use, requests waiting |
| `lisa_http_requests_total{route,code}` | counter | Requests by route and status |
| `lisa_http_request_duration_seconds` | histogram | Request latency |
| `lisa_queue_wait_seconds` | histogram | Time spent waiting for a slot |
| `lisa_time_to_first_token_seconds` | histogram | How long until the first token |
| `lisa_generation_tokens_per_second` | histogram | Generation speed per request |
| `lisa_prompt_tokens_total`, `lisa_completion_tokens_total` | counter | Tokens in and out |
| `lisa_prompt_cached_tokens_total` | counter | Prompt tokens served from the prompt cache |
| `lisa_generations_total` | counter | Completed generations |
| `lisa_requests_rejected_total{reason}` | counter | 429s and other refusals, by reason |
| `lisa_worker_deaths_total`, `lisa_worker_restarts_total` | counter | Crashed or killed inference workers |
| `lisa_build_info{version}` | gauge | The running version |

Good alerts to start with: `lisa_ready == 0` for more than a minute, any increase of
`lisa_worker_deaths_total` (a crash or a blocked system call: investigate the model), and a
growing `lisa_requests_rejected_total` (time for more `parallel_slots` or another instance).

---

## Stopping and upgrading

`SIGTERM` or `SIGINT` (Ctrl-C) starts a graceful drain:

1. `/readyz` turns `503`, so load balancers stop sending traffic.
2. New requests get `503` with `Retry-After`.
3. Running and queued requests finish, for up to `shutdown_timeout_seconds` (30 s); whatever is
   still running then is cancelled.

A second signal stops at once. For rolling upgrades, give the orchestrator a stop timeout longer
than `shutdown_timeout_seconds` (Kubernetes: `terminationGracePeriodSeconds: 45`).

---

## systemd

```ini
# /etc/systemd/system/lisa.service
[Unit]
Description=LiSa-LLM inference server
After=network-online.target
Wants=network-online.target

[Service]
User=lisa
Group=lisa
ExecStart=/usr/local/bin/lisa --config /etc/lisa/config.yaml
Restart=on-failure
TimeoutStopSec=45
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo useradd --system --no-create-home lisa
sudo systemctl daemon-reload && sudo systemctl enable --now lisa
journalctl -u lisa -f -o cat
```

Keep the configuration and key files readable only by `lisa` (`chmod 600`; LiSa warns about a
plaintext key file that group or others can read). If you add more
systemd sandboxing, leave user namespaces allowed (no `RestrictNamespaces=` or `PrivateUsers=`):
LiSa's own sandbox needs them. For hard memory limits, add `Delegate=yes` and point
`sandbox_cgroup` at the delegated cgroup folder.

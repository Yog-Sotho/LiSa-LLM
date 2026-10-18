# 🚀 Quick start

From zero to your own model answering questions, in about five minutes. This uses the release
binary; with Docker, see the one-liner in [Installation → Container images](installation.md#container-images)
and skip to [step 4](#4-ask-it-something).

---

## 1. Download LiSa-LLM

```bash
VERSION=1.0.1 ARCH=x86_64        # or aarch64
curl -LO https://github.com/Yog-Sotho/LiSa-LLM/releases/download/v$VERSION/lisa-$VERSION-linux-$ARCH.tar.gz
tar xzf lisa-$VERSION-linux-$ARCH.tar.gz
sudo install -m 755 lisa-$VERSION-linux-$ARCH/bin/lisa /usr/local/bin/
```

## 2. Download a model

```bash
mkdir -p models
curl -L -o models/model.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Any GGUF chat model works; this one is small enough to run fast on any laptop.

## 3. Start the server

Make yourself an API key and start LiSa with it:

```bash
export LISA_API_KEYS="$(openssl rand -hex 24)"
echo "Your API key: $LISA_API_KEYS"
lisa --model models/model.gguf
```

You'll see a few lines of JSON logs. These two mean you're good to go:

```json
{"landlock_abi":7,"level":"info","message":"Sandboxed worker ready","pid":5456,"timestamp":1791764108919}
{"api_keys":1,"bind":"127.0.0.1:8080","level":"info","message":"Starting server","model":"model","mtls":false,"timestamp":1791764108919,"tls":false}
```

"Sandboxed worker ready" means the model was loaded inside the locked-down worker, not in the server
itself. If you get an error about **user namespaces** instead, your system blocks the sandbox; the
message says how to allow it, and [Troubleshooting](troubleshooting.md#-sandbox-user-namespaces-look-blocked)
has the details.

> **Port taken?** `lisa --model models/model.gguf --port 9000`.

## 4. Ask it something

In a second terminal (with the same `LISA_API_KEYS` value):

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $LISA_API_KEYS" -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain sandboxing to a cat."}], "max_tokens": 120}'
```

Or from Python, with the official OpenAI SDK (`pip install openai`):

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=os.environ["LISA_API_KEYS"])
stream = client.chat.completions.create(
    model="model",                        # any name works: one model is loaded
    messages=[{"role": "user", "content": "Write a haiku about sandboxes."}],
    max_tokens=100,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

Words appear as they are generated. 🎉 That's it: you have a private, OpenAI-compatible model server.

## 5. Make it yours

Copy the sample configuration and edit it; every option is commented:

```bash
cp lisa-$VERSION-linux-$ARCH/config.yaml .
lisa --config config.yaml --check-config      # validates settings and the model, then exits
lisa --config config.yaml
```

Things people usually change first:

| Want to… | Set |
|---|---|
| Serve more people at once | `parallel_slots` (default 4); see [Performance](performance.md) |
| Give the model a longer memory | `max_context` (default 2048 tokens; the sample config uses 4096) |
| Reach it from other machines | `listen_addr: "0.0.0.0"` plus TLS and a strong key; see [Security](security.md#exposing-the-server) |
| Keep keys out of the config | `lisa --hash-key` and `sha256:` entries; see [Configuration](configuration.md#api-keys) |
| Watch it in Grafana | Scrape `/metrics`; see [Deployment](deployment.md#monitoring) |

`Ctrl-C` stops the server gracefully: requests already running finish first. Press it twice to stop
immediately.

---

## Plug it into your tools

Anything that lets you set an "OpenAI base URL" works with LiSa: point it at
`http://<host>:8080/v1` and give it your key.

- **[Brainbrew](https://github.com/Yog-Sotho/Brainbrew)**: pick *Custom URL* as the model endpoint
  and enter `http://localhost:8080/v1`. Brainbrew asks for JSON-schema output first; LiSa answers
  that with a clear `400`, and Brainbrew falls back to JSON mode, which LiSa supports.
- **[LLM Fine-Tuner](https://github.com/Yog-Sotho/LLM-fine-tuner)**: export your fine-tuned model
  as GGUF and serve that file with LiSa. Its `synthesize` command can also use LiSa as the
  writer model, since that only needs an OpenAI-compatible server.
- **Your own code**: the OpenAI SDKs for Python, JavaScript, Go and others work unchanged. See
  [Using the API](api.md) for what's supported.

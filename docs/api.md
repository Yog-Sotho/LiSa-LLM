# 📡 Using the API

LiSa-LLM speaks the **OpenAI API**, so the official SDKs and most tools built for OpenAI work by
changing two settings: the base URL (`http://<host>:8080/v1`) and the API key.

---

## Routes

| Route | Key needed? | What it does |
|---|---|---|
| `POST /v1/chat/completions` | yes | Chat, formatted with the model's chat template |
| `POST /v1/completions` | yes | Plain text completion of a raw prompt |
| `GET /v1/models` | yes | The loaded model (one per server) |
| `GET /metrics` | yes, when keys are configured | Prometheus metrics ([Deployment → Monitoring](deployment.md#monitoring)) |
| `GET /healthz` | no | Liveness: the process is up |
| `GET /readyz` | no | Readiness: `200 {"status":"ready"}`, or `503` while draining or when the worker keeps crashing |

## Authentication

Send your key in either header:

```
Authorization: Bearer <key>
X-API-Key: <key>
```

A wrong or missing key gets `401` with code `invalid_api_key`. If no key is configured at all,
authentication is off; the server only allows that quietly on a loopback address and warns
otherwise. How to set keys: [Configuration → API keys](configuration.md#api-keys).

---

## Chat completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
        "messages": [
          {"role": "system", "content": "You are a terse assistant."},
          {"role": "user", "content": "Name three sandboxing techniques."}
        ],
        "max_tokens": 80,
        "temperature": 0.2
      }'
```

The answer (the text depends on your model):

```json
{
  "id": "chatcmpl-19cf438a9ae4f27f8b3a6e04",
  "object": "chat.completion",
  "created": 1791881393,
  "model": "qwen",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Namespaces, seccomp filters and Landlock."},
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 29, "completion_tokens": 11, "total_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 0}}
}
```

- **Roles:** `system`, `developer` (treated like `system`), `user`, `assistant`.
- **Content:** a string, or an array of `{"type": "text", "text": "…"}` parts. Images and audio are
  not supported.
- **Chat templates:** the template stored in the GGUF file is used automatically. llama.cpp's 54
  built-in formats are recognised (Llama 2/3/4, ChatML, Mistral, Gemma, Phi, DeepSeek, Qwen,
  Command-R, …). Force one with `chat_template` in the config (e.g. `llama3`, `chatml`); a model
  without a template falls back to ChatML.
- **No forged roles:** message text that spells a special token (like `<|im_end|>`) is rejected,
  so users can't fake a system message by typing one.

## Text completions

```bash
curl http://localhost:8080/v1/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time, in a sandbox,", "max_tokens": 40}'
```

The prompt is used as is, with special-token parsing off: text in it is always text.

---

## Request fields

| Field | Range / values | Default |
|---|---|---|
| `messages` (chat) / `prompt` (completions) | required | |
| `max_tokens`, `max_completion_tokens` | ≥ 1, must fit the context with the prompt | `max_new_tokens_default`, capped by the free context |
| `temperature` | 0–2; `0` = greedy | `temperature_default` (0.8) |
| `top_p` | (0, 1] | `top_p_default` (0.95) |
| `top_k` *(llama.cpp extension)* | 0–1000; 0 = off | 0 |
| `min_p` *(llama.cpp extension)* | 0–1 | 0 |
| `presence_penalty`, `frequency_penalty` | −2 to 2 | 0 |
| `repeat_penalty` *(llama.cpp extension)* | (0, 2] | off |
| `seed` | integer ≥ 0 | random |
| `stop` | a string or up to 4 strings of ≤ 256 bytes | none |
| `stream` | `true` / `false` | `false` |
| `stream_options.include_usage` | `true` / `false` | `false` |
| `response_format` | `{"type": "text"}` or `{"type": "json_object"}` | text |
| `model` | any string; ignored, since one model is loaded | responses name the model's alias |

`finish_reason` is `stop` (end of text or a stop sequence) or `length` (hit `max_tokens` or the end
of the context). Stop sequences are cut from the output, also when they arrive split across
tokens in a stream.

### Not supported, and rejected rather than ignored

`n` or `best_of` other than 1, `logprobs`, `top_logprobs`, `echo`, `suffix`, `tools`, `functions`,
`logit_bias` and `response_format: json_schema` get a `400` that names the field. That's on
purpose: a client that relies on one of them finds out straight away instead of getting different
behaviour without noticing.

---

## Streaming

With `"stream": true` the answer arrives as [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events),
one `data:` line per piece, then `data: [DONE]`:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],…}
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Name"},"finish_reason":null}],…}
…
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],…}
data: [DONE]
```

Add `"stream_options": {"include_usage": true}` for a final chunk with the token counts. Pieces are
always whole UTF-8 characters. If the client disconnects, generation stops and the slot is freed
for the next request.

## JSON mode

```json
{"messages": [{"role": "user", "content": "A JSON object with a cat's name and age."}],
 "response_format": {"type": "json_object"}}
```

The output is constrained by a JSON grammar while it is generated, so it is always valid JSON
(unless it is cut short by `max_tokens`: check that `finish_reason` is `stop`). As with OpenAI, also
ask for JSON in the prompt, or the model may produce a valid but unhelpful object. JSON schemas
(`json_schema`) are not supported yet.

## Prompt caching

There's nothing to switch on. When a request starts with tokens that are already in the cache
(the same system prompt, or the earlier turns of a chat), they are reused instead of evaluated
again, even if another slot holds them. The answer says how many in
`usage.prompt_tokens_details.cached_tokens`. Keep shared text at the **start** of the prompt to
benefit.

---

## Errors

Errors use OpenAI's format, so SDKs raise their usual exceptions:

```json
{"error": {"message": "response_format must be {\"type\": \"text\"} or {\"type\": \"json_object\"}",
           "type": "invalid_request_error", "param": "response_format", "code": null}}
```

| Status | When | What to do |
|---|---|---|
| `400` | Invalid or unsupported field; prompt too long (`code: context_length_exceeded`) | Fix the request; `param` names the field |
| `401` | Missing or wrong API key (`code: invalid_api_key`) | Check the key |
| `413` | Body larger than 1 MiB | Send less text |
| `429` | Your key is over `rate_limit_per_minute`, or the queue is full (`max_queued_requests`) | Wait for `Retry-After` seconds (the OpenAI SDKs do this for you) |
| `503` | The server is draining for shutdown | Retry against another instance after `Retry-After` |
| `500` | Generation failed (for example the worker crashed mid-request, or keeps crashing) | Retry; check the server log for the request id, and `/readyz` |

## Request ids

Every response carries an `X-Request-Id` header. Send your own (short; letters, digits, `-`, `_`,
`.`, `:`) to follow a request through your systems; otherwise the server makes a random one. The same id
is in the server's access log line for that request:

```json
{"client":"key-0061cbd2","duration_us":68,"level":"info","message":"request","method":"GET","path":"/v1/models","remote_addr":"127.0.0.1","request_id":"my-trace.42","status":200,"timestamp":1791881460379}
```

Keys are never logged: `client` is a short, stable fingerprint of the key, so you can tell clients
apart. Successful `/healthz`, `/readyz` and `/metrics` calls are not logged, to keep probes from
flooding the log.

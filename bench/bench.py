#!/usr/bin/env python3
"""Throughput and latency benchmark for OpenAI-compatible completion servers.

Drives /v1/completions with streaming requests at several concurrency levels and
reports aggregate generation throughput, time to first token (TTFT) and latency.
Standard library only, so it runs anywhere Python 3.9+ does.

    python3 bench/bench.py --url http://127.0.0.1:8080 --key $KEY --concurrency 1,2,4,8

Every request gets a distinct prompt (a numbered prefix) so no server can answer from
a prompt cache; pass --shared-prefix to measure prefix caching instead.
"""

import argparse
import http.client
import json
import itertools
import statistics
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

STORY = (
    "Write a long, detailed story about a lighthouse keeper who discovers that the "
    "light is being used to send messages to ships that do not exist. Describe the "
    "keeper, the island, the weather, the messages and what happens next."
)
SYSTEM_PREFIX = (
    "You are a careful assistant. Answer in plain English. Keep the facts straight, "
    "explain your reasoning briefly and avoid repeating yourself. " * 8
)


def one_request(url, key, prompt, max_tokens, timeout):
    """Streams one completion. Returns (ttft_s, total_s, completion_tokens)."""
    parsed = urllib.parse.urlparse(url)
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    start = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body, headers=headers)
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
    ttft = None
    chunks = 0
    usage_tokens = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage_tokens = event["usage"].get("completion_tokens")
        for choice in event.get("choices") or []:
            if choice.get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks += 1
    total = time.perf_counter() - start
    conn.close()
    return (ttft if ttft is not None else total), total, (usage_tokens if usage_tokens is not None else chunks)


def percentile(values, q):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


_counter = itertools.count(1)  # global, so no prompt repeats across levels


def run_level(args, concurrency):
    def task(_):
        n = next(_counter)
        prefix = SYSTEM_PREFIX if args.shared_prefix else f"Request {n}. "
        return one_request(args.url, args.key, prefix + STORY, args.max_tokens, args.timeout)

    requests = max(args.requests, concurrency)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(task, range(requests)))
    wall = time.perf_counter() - start
    ttfts = [r[0] for r in results]
    totals = [r[1] for r in results]
    tokens = sum(r[2] for r in results)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "completion_tokens": tokens,
        "wall_s": wall,
        "throughput_tok_s": tokens / wall,
        "ttft_mean_s": statistics.mean(ttfts),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "latency_p50_s": percentile(totals, 0.5),
        "latency_p95_s": percentile(totals, 0.95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--key", default="", help="API key (Bearer)")
    parser.add_argument("--concurrency", default="1,2,4,8", help="comma-separated levels")
    parser.add_argument("--requests", type=int, default=16, help="requests per level")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--shared-prefix", action="store_true",
                        help="all prompts share a long prefix (measures prompt caching)")
    parser.add_argument("--label", default="server", help="name shown in the table")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    one_request(args.url, args.key, "Warm up.", 4, args.timeout)  # load caches, threads
    rows = [run_level(args, int(c)) for c in args.concurrency.split(",")]
    if args.json:
        print(json.dumps({"label": args.label, "results": rows}, indent=2))
        return
    print(f"| {args.label} | concurrency | tok/s (aggregate) | TTFT mean | TTFT p95 | latency p50 | latency p95 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| | {r['concurrency']} | {r['throughput_tok_s']:.1f} | {r['ttft_mean_s']:.2f} s "
              f"| {r['ttft_p95_s']:.2f} s | {r['latency_p50_s']:.2f} s | {r['latency_p95_s']:.2f} s |")


if __name__ == "__main__":
    main()

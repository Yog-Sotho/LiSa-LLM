# LiSa-LLM threat model

## What is protected, and from what

The main attacker this design assumes is a **malicious model file**: a crafted GGUF that
exploits a memory-safety bug in the model parser or tensor code and gets code execution in
the process that loads it. The sandbox's goal is that such code can do nothing useful: no
host files, no network, no other processes, no persistence, no privilege escalation.

The second attacker is a **network client** sending hostile prompts or requests:
- **Validation:** the broker checks every request (types, ranges, sizes, a 1 MiB body limit)
  before anything reaches the model. The request decoder is fuzzed.
- **Prompts:** they are untrusted data that never become control tokens. Raw prompts are
  tokenized with special-token parsing off, and chat messages that spell a special token
  are rejected.
- **Authentication:** API keys are held only as SHA-256 digests and compared in constant
  time. Off loopback, the server refuses placeholder or short keys.
- **Exhaustion:** clients can't exhaust the broker. There is a bounded first-come,
  first-served queue (429 when full), per-key token-bucket rate limits, read, write and
  keep-alive timeouts, a capped thread pool and a cap on pending connections.
- **TLS:** forward-secret AEAD suites only on TLS 1.2, optional TLS 1.3 minimum, optional
  client certificates (mTLS), HSTS.

The third case is a **compromised worker**, the expected result of a successful model
exploit. The broker treats everything the worker sends as untrusted:
- **Framing:** frames are size-checked (16 MiB) and parsed as JSON. Requests are
  multiplexed: every reply must carry the id of a request the broker sent, or the worker
  is killed.
- **Wrong shapes:** a message of the wrong shape counts as a protocol violation. The worker
  is killed and restarted rather than letting the channel fall out of step.
- **Testing:** this decoder is fuzzed too, and a test runs a deliberately hostile worker.
- **Output:** the worker can still return arbitrary text, which the broker forwards to the
  client as model output.

## Architecture

```
            network (HTTP/TLS, API key)
                     │
   ┌─────────────────▼─────────────────┐   trusted: never parses the model
   │ broker  (lisa)                     │   opens the model fd, verifies SHA-256,
   │  OpenAI API, validation, auth      │   restarts crashed workers
   └─────────────────┬─────────────────┘
        socketpair (length-prefixed JSON, 16 MiB frame limit) + model fd
   ┌─────────────────▼─────────────────┐
   │ reaper  (outside the PID ns)       │   seccomp: wait4/exit only
   └─────────────────┬─────────────────┘
   ┌─────────────────▼─────────────────┐   untrusted input: parses the GGUF
   │ worker  (PID 1 of its namespaces)  │
   └───────────────────────────────────┘
```

The worker is started by re-executing the server binary (`/proc/self/exe --sandbox-worker`)
with exactly two descriptors (the IPC socket on fd 3, the model on fd 4) and an empty
environment. It locks itself down **before** reading the model.

## Worker isolation layers

| Layer | What it does |
|---|---|
| User namespace | Runs as "root" only inside its own namespace, mapped to the broker's unprivileged UID. |
| Mount namespace + `pivot_root` | The root is a 16 KiB read-only tmpfs. It contains only the model, bind-mounted read-only, `nosuid`/`nodev`/`noexec`. The old root is detached, and there is no `/proc`, `/sys` or `/dev`. |
| Model binding | The verified descriptor is reopened inside the namespace, and its inode must match. A path swapped after verification is refused. |
| Network namespace | No interfaces except a downed loopback. |
| PID, IPC, UTS, cgroup namespaces | Can't see or signal host processes. Own hostname, own SysV IPC. |
| Capabilities | Bounding, ambient, effective, permitted and inheritable sets are all empty. |
| `no_new_privs`, non-dumpable | setuid binaries don't help (none are reachable anyway). The process can't be ptraced by peers. |
| Landlock (kernel ≥ 5.13) | All filesystem access is denied except reading the model. TCP bind/connect is denied (ABI 4). Abstract Unix sockets and signals to processes outside the sandbox are denied (ABI 6). Best effort, depending on the kernel's ABI. |
| seccomp-bpf, stage 1 (load) | Allow-list for loading the model. Default action: kill the whole process. |
| seccomp-bpf, stage 2 (serve) | A narrower list once the model is loaded: computing, memory, futexes and the IPC socket. `openat` fails with EACCES. No `socket`, `connect`, `execve`, `fork`, `mount`, `unshare`, `ptrace`, `kill` or `setuid`. Threads only: `clone3` returns ENOSYS so glibc falls back to `clone`, and `clone` is allowed only with `CLONE_THREAD`. |
| Resource limits | `RLIMIT_DATA` = `memory_limit_bytes` (worker heap), 64 open files, no core dumps. With `sandbox_cgroup`: `memory.max`, `memory.swap.max=0`, `pids.max`. |
| Lifetime | `PR_SET_PDEATHSIG`: the worker dies with the broker. The broker logs every worker death (a seccomp kill shows up as SIGSYS) and restarts it at most 3 times a minute. |

The allow-lists were derived from `strace` runs of real workloads: F32 and Q4_K models,
1–4 threads, streaming, cancellation and chat templates. The test suite runs escape probes
inside a live worker and expects every one to be killed by seccomp.

## What it does not protect against

- **Kernel vulnerabilities** reachable through the allowed system calls. User namespaces
  themselves widen the kernel attack surface. That is why some distributions restrict them,
  and the sandbox needs them.
- **Denial of service by the model**: a malicious model can burn CPU or return wrong
  tokens. Memory is capped (`RLIMIT_DATA`, or a cgroup when configured), and CPU time is not.
  Network clients are rate-limited and queued, but a large volume of traffic from many
  addresses without keys still needs a front-end limiter.
- **The broker itself**: it handles network input with cpp-httplib and is not sandboxed. Run
  it as an unprivileged user. The Docker image runs as UID 10001. Its own request decoding
  is fuzzed, and its memory safety is checked under ASan/UBSan in CI.
- **Model confidentiality or integrity at rest**: `model_sha256` checks the file once at
  startup. Anyone who can write to the model file can change what gets loaded on the next start.
- **GPU inference**: GPU drivers need device nodes and `ioctl`, so sandboxed CUDA inference is
  refused at startup. Use `sandbox_enabled: false` for GPU offload, ideally inside a
  container.

## Requirements and deployment notes

- Linux with unprivileged user namespaces.
  - Ubuntu 23.10+ restricts them through AppArmor. Run
    `sysctl kernel.apparmor_restrict_unprivileged_userns=0` or install an AppArmor profile
    for `lisa`.
  - In Docker, pass `--security-opt seccomp=unconfined --security-opt apparmor=unconfined`.
    These only relax Docker's own profiles; LiSa installs a stricter filter in the worker.
  - Otherwise set `sandbox_enabled: false`, which logs a warning.
- If the sandbox can't be set up, the server **refuses to start**. It never silently runs
  unsandboxed.
- `sandbox_seccomp_log: true` logs blocked system calls (audit log) instead of killing the
  worker. Use it only to diagnose a kill on an unusual platform.

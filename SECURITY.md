# Security policy

LiSa-LLM's purpose is to run untrusted models safely, so security reports are very welcome.

## Supported versions

| Version | Supported |
|---|---|
| 1.x (latest release) | ✅ |
| older / unreleased `main` snapshots | best effort |

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through GitHub:
[**Report a vulnerability**](https://github.com/Yog-Sotho/LiSa-LLM/security/advisories/new)
(Security tab → Advisories → *Report a vulnerability*).

Include what you can of: the affected version or commit, your configuration (sandbox on or
off, TLS, Docker or bare metal, kernel version), steps or a proof of concept, and the impact
you see. A malicious GGUF file that demonstrates the problem is ideal; please attach it to the
private advisory rather than publishing it.

What to expect: an acknowledgement as soon as a maintainer sees the report, a fix developed
in a private fork, a coordinated disclosure date agreed with you, and credit in the advisory
and the changelog unless you prefer otherwise.

## Scope

In scope — please report:

- **Sandbox escapes**: code running in the inference worker reaching files, the network,
  other processes or more privileges than [the threat model](docs/project/threat_model.md)
  allows.
- **Broker compromise**: memory-safety bugs, authentication bypass, request smuggling or
  crashes in the network-facing process, including through messages from a compromised
  worker.
- **Secret exposure**: API keys, TLS keys or other configuration reaching logs, the worker
  or clients.
- **Supply chain**: problems with the release artifacts, signatures or provenance.

Out of scope (documented non-goals, see the threat model): kernel vulnerabilities reachable
through the allowed system calls, CPU exhaustion by a model, deployments with
`sandbox_enabled: false`, GPU inference (which can't be sandboxed), and denial of service
by high request volume without rate limits configured.

## Verifying releases

Release archives and container images are signed with Sigstore (keyless, tied to this
repository's release workflow) and carry SLSA build provenance and an SPDX SBOM. See
[Verifying a release](README.md#verifying-a-release).

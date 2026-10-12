#!/usr/bin/env python3
"""SPDX 2.3 SBOM for a portable LiSa-LLM binary.

Generic scanners can't see what CMake's FetchContent compiles in, so the SBOM is built
from the pins in cmake/Dependencies.cmake (tag + verified commit, or URL + SHA-256) plus
the system libraries linked statically, whose package versions the caller passes:

    packaging/sbom.py --version 1.0.0 --arch x86_64 \\
        --system openssl=$(dpkg-query -W -f='${Version}' libssl-dev) ... > lisa.spdx.json
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "https://github.com/Yog-Sotho/LiSa-LLM"

# Licences of what the binary contains (SPDX identifiers).
LICENSES = {
    "llama_cpp": "MIT",
    "httplib": "MIT",
    "nlohmann_json": "MIT",
    "yaml-cpp": "MIT",
    "openssl": "Apache-2.0",
    "libseccomp": "LGPL-2.1-only",
    "libstdc++": "GPL-3.0-or-later WITH GCC-exception-3.1",
}
# Debian/Ubuntu source packages of the statically linked system libraries.
DEB_PACKAGES = {"openssl": "openssl", "libseccomp": "libseccomp", "libstdc++": "gcc-12"}


def fetched_dependencies():
    """(name, url, tag, commit) of lisa_fetch_git() calls and URL archives."""
    text = (ROOT / "cmake" / "Dependencies.cmake").read_text()
    deps = []
    for name, url, tag, commit in re.findall(
        r"lisa_fetch_git\((\S+)\s+(\S+)\s+(\S+)\s+([0-9a-f]{40})\)", text
    ):
        deps.append({"name": name, "url": url, "version": tag, "commit": commit})
    for name, url, sha in re.findall(
        r"FetchContent_Declare\((\S+)\s+URL\s+(\S+)\s+URL_HASH\s+SHA256=([0-9a-f]{64})", text
    ):
        version = re.search(r"/v?(\d+\.\d+\.\d+)/", url)
        deps.append({"name": name, "url": url, "version": version.group(1) if version else "",
                     "sha256": sha})
    return deps


def spdx_id(name):
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--distro", default="ubuntu-22.04")
    parser.add_argument("--system", action="append", default=[], metavar="NAME=VERSION",
                        help="statically linked system library and its package version")
    args = parser.parse_args()

    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0")) or None
    created = (datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc) if epoch
               else datetime.datetime.now(datetime.timezone.utc))
    name = f"lisa-{args.version}-linux-{args.arch}"
    lisa = {
        "SPDXID": "SPDXRef-Package-lisa",
        "name": "lisa-llm",
        "versionInfo": args.version,
        "downloadLocation": f"git+{REPO}.git@v{args.version}",
        "licenseConcluded": "MIT",
        "licenseDeclared": "MIT",
        "copyrightText": "NOASSERTION",
        "filesAnalyzed": False,
        "supplier": "Organization: Yog-Sotho",
        "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                          "referenceLocator": f"pkg:github/Yog-Sotho/LiSa-LLM@v{args.version}"}],
    }
    packages = [lisa]
    for dep in fetched_dependencies():
        package = {
            "SPDXID": spdx_id(dep["name"]),
            "name": dep["name"],
            "versionInfo": dep["version"],
            "licenseConcluded": LICENSES.get(dep["name"], "NOASSERTION"),
            "licenseDeclared": LICENSES.get(dep["name"], "NOASSERTION"),
            "copyrightText": "NOASSERTION",
            "filesAnalyzed": False,
        }
        if "commit" in dep:
            package["downloadLocation"] = f"git+{dep['url']}@{dep['commit']}"
            owner_repo = re.search(r"github\.com/([^/]+/[^/.]+)", dep["url"])
            if owner_repo:
                package["externalRefs"] = [{
                    "referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                    "referenceLocator": f"pkg:github/{owner_repo.group(1)}@{dep['commit']}"}]
        else:
            package["downloadLocation"] = dep["url"]
            package["checksums"] = [{"algorithm": "SHA256", "checksumValue": dep["sha256"]}]
        packages.append(package)
    for entry in args.system:
        lib, _, version = entry.partition("=")
        packages.append({
            "SPDXID": spdx_id(lib),
            "name": lib,
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "licenseConcluded": LICENSES.get(lib, "NOASSERTION"),
            "licenseDeclared": LICENSES.get(lib, "NOASSERTION"),
            "copyrightText": "NOASSERTION",
            "filesAnalyzed": False,
            "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                              "referenceLocator": f"pkg:deb/ubuntu/{DEB_PACKAGES.get(lib, lib)}"
                                                  f"@{version}?distro={args.distro}"}],
        })

    relationships = [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
                      "relatedSpdxElement": lisa["SPDXID"]}]
    relationships += [{"spdxElementId": lisa["SPDXID"], "relationshipType": "CONTAINS",
                       "relatedSpdxElement": p["SPDXID"]} for p in packages[1:]]
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": name,
        "documentNamespace": f"{REPO}/sbom/{name}-{uuid.uuid5(uuid.NAMESPACE_URL, REPO + name)}",
        "creationInfo": {"created": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "creators": ["Tool: LiSa-LLM packaging/sbom.py"]},
        "packages": packages,
        "relationships": relationships,
    }
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()

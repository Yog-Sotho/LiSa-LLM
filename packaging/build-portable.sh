#!/usr/bin/env bash
# Builds the portable release binary and packages it:
#   dist/lisa-<version>-linux-<arch>.tar.gz and its .sha256
#
# Every dependency except glibc is linked statically (LISA_PORTABLE). The glibc floor is
# the build system's, so release builds run on Ubuntu 22.04 (glibc 2.35); see
# .github/workflows/release.yml. Usage: packaging/build-portable.sh [build-dir] [dist-dir]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly root
readonly build="${1:-${root}/build/portable}"
readonly dist="${2:-${root}/dist}"

cmake -S "${root}" -B "${build}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLISA_PORTABLE=ON \
    -DLISA_NATIVE=OFF \
    -DLISA_WARNINGS_AS_ERRORS=ON \
    -DGGML_OPENMP=OFF
cmake --build "${build}"
ctest --test-dir "${build}" --output-on-failure

binary="${build}/lisa"
version="$("${binary}" --version | awk '{print $2}')"
arch="$(uname -m)"
name="lisa-${version}-linux-${arch}"

# Nothing but glibc (and the kernel's vDSO) may be loaded at run time.
if ldd "${binary}" | grep -vE 'linux-vdso|ld-linux|libc\.so|libm\.so' | grep -q '=>'; then
    echo "error: ${binary} links shared libraries other than glibc:" >&2
    ldd "${binary}" >&2
    exit 1
fi
glibc="$(objdump -T "${binary}" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
echo "${name}: needs ${glibc} or newer"

staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT
mkdir -p "${staging}/${name}/bin" "${staging}/${name}/doc"
install -m 0755 "${binary}" "${staging}/${name}/bin/lisa"
strip "${staging}/${name}/bin/lisa"
install -m 0644 "${root}/config.yaml" "${staging}/${name}/config.yaml"
install -m 0644 "${root}/LICENSE" "${root}/README.md" "${root}/CHANGELOG.md" \
    "${root}/SECURITY.md" "${root}/docs/project/threat_model.md" "${staging}/${name}/doc/"
install -m 0644 "${root}/packaging/THIRD-PARTY-NOTICES.md" "${staging}/${name}/"
"${root}/packaging/collect-licenses.sh" "${build}" "${staging}/${name}/licenses"

mkdir -p "${dist}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "${root}" log -1 --format=%ct)}"

# SBOM: pinned dependencies plus the versions of the statically linked system libraries.
if command -v dpkg-query >/dev/null; then
    gcc_major="$("${CXX:-c++}" -dumpversion | cut -d. -f1)"
    "${root}/packaging/sbom.py" --version "${version}" --arch "${arch}" \
        --distro "$(
            # shellcheck source=/dev/null
            . /etc/os-release && echo "${ID}-${VERSION_ID}"
        )" \
        --system "openssl=$(dpkg-query -W -f='${Version}' libssl-dev)" \
        --system "libseccomp=$(dpkg-query -W -f='${Version}' libseccomp-dev)" \
        --system "libstdc++=$(dpkg-query -W -f='${Version}' "libstdc++-${gcc_major}-dev")" \
        > "${dist}/${name}.spdx.json"
fi

# Reproducible archive: fixed owner, order and timestamps.
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime="@${SOURCE_DATE_EPOCH}" \
    -C "${staging}" -czf "${dist}/${name}.tar.gz" "${name}"
(cd "${dist}" && sha256sum "${name}.tar.gz" > "${name}.tar.gz.sha256")
echo "${dist}/${name}.tar.gz"

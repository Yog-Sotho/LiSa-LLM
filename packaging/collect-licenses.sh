#!/usr/bin/env bash
# Copies the licence texts of everything linked into lisa into <dest>.
# Usage: packaging/collect-licenses.sh <cmake-build-dir> <dest>
set -euo pipefail
readonly build="$1" dest="$2"
mkdir -p "${dest}"

copy() {  # copy <source> <name>; every listed licence must exist
    if [[ ! -f "$1" ]]; then
        echo "error: licence file $1 not found" >&2
        exit 1
    fi
    install -m 0644 "$1" "${dest}/$2"
}

copy "${build}/_deps/llama_cpp-src/LICENSE" llama.cpp.txt
copy "${build}/_deps/httplib-src/LICENSE" cpp-httplib.txt
copy "${build}/_deps/nlohmann_json-src/LICENSE.MIT" nlohmann-json.txt
# yaml-cpp: fetched for portable builds, a distribution package otherwise.
if [[ -f "${build}/_deps/yaml-cpp-src/LICENSE" ]]; then
    copy "${build}/_deps/yaml-cpp-src/LICENSE" yaml-cpp.txt
else
    copy /usr/share/doc/libyaml-cpp-dev/copyright yaml-cpp.copyright.txt
fi
# Debian/Ubuntu copyright files of the system libraries, with the licences they cite.
copy /usr/share/doc/libssl-dev/copyright openssl.copyright.txt
copy /usr/share/common-licenses/Apache-2.0 Apache-2.0.txt
copy /usr/share/doc/libseccomp-dev/copyright libseccomp.copyright.txt
copy /usr/share/common-licenses/LGPL-2.1 LGPL-2.1.txt
gcc_base="$(find /usr/share/doc -maxdepth 1 -name 'gcc-*-base' | sort -V | tail -1)"
copy "${gcc_base}/copyright" gcc-runtime.copyright.txt

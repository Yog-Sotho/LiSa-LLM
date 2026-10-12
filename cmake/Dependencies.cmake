# Third-party dependencies. Every source is pinned: git dependencies by tag *and* commit
# (the commit is verified after the shallow clone), archives by SHA-256.
#
# Offline / air-gapped builds: point FETCHCONTENT_SOURCE_DIR_<NAME> at a local checkout
# (e.g. -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/src/llama.cpp); the commit check is then skipped.

# Portable binaries link OpenSSL statically. Set before anything (cpp-httplib included)
# looks OpenSSL up, because the result is cached.
if(LISA_PORTABLE)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

include(FetchContent)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

# lisa_fetch_git(<name> <repository> <tag> <expected commit>)
function(lisa_fetch_git name repository tag expected_commit)
    FetchContent_Declare(${name}
        GIT_REPOSITORY "${repository}"
        GIT_TAG "${tag}"
        GIT_SHALLOW TRUE
        GIT_PROGRESS FALSE
        EXCLUDE_FROM_ALL
        SYSTEM
    )
    FetchContent_MakeAvailable(${name})

    string(TOUPPER "${name}" upper_name)
    if(FETCHCONTENT_SOURCE_DIR_${upper_name})
        message(STATUS "${name}: using local source override, commit pin not verified")
        return()
    endif()

    FetchContent_GetProperties(${name} SOURCE_DIR source_dir)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD
        OUTPUT_VARIABLE actual_commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE git_result
    )
    if(NOT git_result EQUAL 0 OR NOT actual_commit STREQUAL expected_commit)
        message(FATAL_ERROR
            "${name}: tag ${tag} resolved to '${actual_commit}', expected ${expected_commit}. "
            "Refusing to build an unverified dependency.")
    endif()
    message(STATUS "${name}: ${tag} (${expected_commit}) verified")
endfunction()

# --- llama.cpp (inference engine: GGUF loader, tokenizer, ggml backends, samplers) ----------
set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_COMMON OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_OPENSSL OFF)
set(GGML_NATIVE ${LISA_NATIVE})
set(GGML_CUDA ${LISA_CUDA})
lisa_fetch_git(llama_cpp
    https://github.com/ggml-org/llama.cpp.git
    v0.5.0
    7fe450e19305b828c199d602c23a8337aaa1f03b)

# --- cpp-httplib (HTTP/1.1 server) -----------------------------------------------------------
# Compression support stays off: no decompression-bomb surface on request bodies.
set(HTTPLIB_INSTALL OFF)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF)
set(HTTPLIB_REQUIRE_OPENSSL ${LISA_TLS})
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF)
# getaddrinfo_a runs lookups on threads glibc creates internally, which ThreadSanitizer
# cannot track (it crashes); the server only resolves its bind address once anyway.
set(HTTPLIB_USE_NON_BLOCKING_GETADDRINFO OFF)
lisa_fetch_git(httplib
    https://github.com/yhirose/cpp-httplib.git
    v0.60.1
    00afaaed6aaba592e4d079742dfecfba2dcd1196)

# --- nlohmann/json ---------------------------------------------------------------------------
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(nlohmann_json)

# --- yaml-cpp (system package preferred; built from source, static, for portable binaries) ---
if(NOT LISA_PORTABLE)
    find_package(yaml-cpp 0.7 CONFIG QUIET)
endif()
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp: using system package ${yaml-cpp_VERSION}")
else()
    set(YAML_CPP_BUILD_TESTS OFF)
    set(YAML_CPP_BUILD_TOOLS OFF)
    set(YAML_CPP_BUILD_CONTRIB OFF)
    set(YAML_CPP_INSTALL OFF)
    set(YAML_CPP_FORMAT_SOURCE OFF)
    lisa_fetch_git(yaml-cpp
        https://github.com/jbeder/yaml-cpp.git
        yaml-cpp-0.9.0
        56e3bb550c91fd7005566f19c079cb7a503223cf)
endif()

# --- OpenSSL (model SHA-256; TLS when LISA_TLS) ------------------------------------------
if(LISA_TLS)
    find_package(OpenSSL 3.0 REQUIRED COMPONENTS Crypto SSL)
else()
    find_package(OpenSSL 3.0 REQUIRED COMPONENTS Crypto)
endif()

# --- libseccomp (sandbox) --------------------------------------------------------------------
if(LISA_SANDBOX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBSECCOMP REQUIRED IMPORTED_TARGET libseccomp)
    if(LISA_PORTABLE)
        find_library(LISA_SECCOMP_STATIC NAMES libseccomp.a REQUIRED)
        add_library(lisa_seccomp INTERFACE)
        target_link_libraries(lisa_seccomp INTERFACE "${LISA_SECCOMP_STATIC}")
        target_include_directories(lisa_seccomp INTERFACE ${LIBSECCOMP_INCLUDE_DIRS})
    else()
        add_library(lisa_seccomp ALIAS PkgConfig::LIBSECCOMP)
    endif()
endif()

cmake_minimum_required(VERSION 3.20)
project(llm_client LANGUAGES C CXX)

# -------------------------------------------------
# 1️⃣ Global options
# -------------------------------------------------
option(STATIC_BUILD "Link everything statically" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(STATIC_BUILD)
    set(CMAKE_EXE_LINKER_FLAGS "-static -s")
endif()

# -------------------------------------------------
# 2️⃣ GGML static library
# -------------------------------------------------
add_library(ggml_static STATIC
    third_party/ggml/ggml.c
    third_party/ggml/ggml-alloc.c
    third_party/ggml/ggml-nn.c
)
target_include_directories(ggml_static PUBLIC third_party/ggml)

# -------------------------------------------------
# 3️⃣ Project sources
# -------------------------------------------------
set(SRC
    src/main.cpp
    src/sandbox.cpp
    src/inference.cpp
    src/model_loader.cpp
)

add_executable(llm_client ${SRC})
target_include_directories(llm_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(llm_client PRIVATE ggml_static)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(llm_client PRIVATE
        -pthread
        -lrt
        -ldl
        -lseccomp   # libseccomp for sandbox
        -lcrypto    # OpenSSL for SHA‑256 checksum
    )
endif()

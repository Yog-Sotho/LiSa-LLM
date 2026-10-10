# Warnings, sanitizers and hardening flags. Applied to first-party targets only, so
# third-party code fetched in Dependencies.cmake is never built with -Werror.

set(LISA_GCC_CLANG_WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wcast-align
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

function(lisa_configure_target target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
        if(LISA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
        return()
    endif()

    target_compile_options(${target} PRIVATE ${LISA_GCC_CLANG_WARNINGS})
    if(LISA_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()

    if(LISA_SANITIZERS)
        target_compile_options(${target} PRIVATE
            -fsanitize=${LISA_SANITIZERS} -fno-omit-frame-pointer -fno-sanitize-recover=all)
        target_link_options(${target} PRIVATE -fsanitize=${LISA_SANITIZERS})
    endif()

    if(LISA_HARDENING)
        target_compile_options(${target} PRIVATE -fstack-protector-strong -fstack-clash-protection)
        # FORTIFY_SOURCE needs optimisation and conflicts with ASan's interceptors.
        if(NOT LISA_SANITIZERS)
            target_compile_options(${target} PRIVATE
                $<$<NOT:$<CONFIG:Debug>>:-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3>)
        endif()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target} PRIVATE -fcf-protection=full)
        endif()
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "EXECUTABLE")
            target_link_options(${target} PRIVATE -pie -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack)
        endif()
    endif()
endfunction()

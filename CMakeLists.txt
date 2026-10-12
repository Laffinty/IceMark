cmake_minimum_required(VERSION 3.15)
project(IceMark VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# A benchmark that silently builds at -O0 measures the compiler, not the CPU.
# Default to Release when the caller did not choose, and refuse Debug outright.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_BUILD_TYPE_RELEASE "Release" CACHE STRING "" FORCE)

if(MSVC)
    # /utf-8 is mandatory, not cosmetic: without it MSVC decodes source with
    # the system codepage and every Chinese UI string becomes a compile error.
    add_compile_options(/W4 /utf-8 /fp:precise /permissive-)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_executable(icemark WIN32
    src/main.cpp
    src/cpu.cpp
    src/weights.cpp
    src/kernel.cpp
    src/kernel_ref.cpp
    src/kernel_avx2.cpp
    src/harness.cpp
    src/scoring.cpp
    src/gui.cpp
)

target_include_directories(icemark PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MSVC)
    # The AVX2 translation unit is the only one that may emit AVX2, and it is
    # guarded at runtime by a CPUID check.
    set_source_files_properties(src/kernel_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    target_link_libraries(icemark PRIVATE user32 gdi32 comctl32 comdlg32 shlwapi shell32)
    target_compile_definitions(icemark PRIVATE
        UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    set_source_files_properties(src/kernel_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2 -mfma")
    target_link_libraries(icemark PRIVATE pthread m)
endif()
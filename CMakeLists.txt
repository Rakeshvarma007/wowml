cmake_minimum_required(VERSION 3.20)
project(wowml_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ---- Performance & Architecture Flags ----
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT APPLE)
        add_compile_options(-O3 -march=native -Wall -Wextra)
    else()
        add_compile_options(-O3 -Wall -Wextra)
    endif()
endif()

# ---- Multithreading via OpenMP ----
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found. Multi-threading enabled.")
endif()

# ---- Optional OpenBLAS / LAPACKE Fast Paths ----
option(WOWML_USE_BLAS "Use OpenBLAS and LAPACKE fast paths" ON)
set(WOWML_BLAS_LIBS "")
set(WOWML_BLAS_INCS "")

if(WOWML_USE_BLAS)
    if(APPLE)
        find_package(BLAS)
        find_package(LAPACK)
        if(BLAS_FOUND AND LAPACK_FOUND)
            message(STATUS "Apple Accelerate BLAS/LAPACK found.")
            add_compile_definitions(WOWML_USE_BLAS=1)
            list(APPEND WOWML_BLAS_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        endif()
    else()
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(OPENBLAS openblas)
        endif()

        if(OPENBLAS_FOUND)
            message(STATUS "OpenBLAS found via pkg-config.")
            add_compile_definitions(WOWML_USE_BLAS=1)
            list(APPEND WOWML_BLAS_LIBS ${OPENBLAS_LIBRARIES})
            list(APPEND WOWML_BLAS_INCS ${OPENBLAS_INCLUDE_DIRS})
        else()
            find_path(CBLAS_INCLUDE_DIR cblas.h
                PATHS
                    /usr/include
                    /usr/include/x86_64-linux-gnu
                    /usr/include/openblas
                    /usr/local/include
                    "C:/openblas/include"
                    "$ENV{OPENBLAS_HOME}/include"
            )
            find_library(OPENBLAS_LIB
                NAMES openblas libopenblas
                PATHS
                    /usr/lib
                    /usr/lib/x86_64-linux-gnu
                    /usr/local/lib
                    "C:/openblas/lib"
                    "$ENV{OPENBLAS_HOME}/lib"
            )
            if(CBLAS_INCLUDE_DIR AND OPENBLAS_LIB)
                message(STATUS "OpenBLAS found: ${OPENBLAS_LIB} (include: ${CBLAS_INCLUDE_DIR})")
                add_compile_definitions(WOWML_USE_BLAS=1)
                list(APPEND WOWML_BLAS_LIBS ${OPENBLAS_LIB})
                list(APPEND WOWML_BLAS_INCS ${CBLAS_INCLUDE_DIR})
            else()
                find_package(BLAS)
                if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
                    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
                    add_compile_definitions(WOWML_USE_BLAS=1)
                    list(APPEND WOWML_BLAS_LIBS ${BLAS_LIBRARIES})
                    list(APPEND WOWML_BLAS_INCS ${CBLAS_INCLUDE_DIR})
                else()
                    message(WARNING "OpenBLAS/CBLAS not found. Falling back to pure C11 reference implementations.")
                    set(WOWML_USE_BLAS OFF CACHE BOOL "Use OpenBLAS and LAPACKE fast paths" FORCE)
                endif()
            endif()
        endif()
    endif()
endif()

# ---- Core C Shared Library ----
file(GLOB WOWML_SOURCES csrc/src/*.c)
add_library(wowml_c SHARED ${WOWML_SOURCES})
target_include_directories(wowml_c PUBLIC csrc/include)

if(UNIX AND NOT APPLE)
    target_link_libraries(wowml_c PRIVATE m)
endif()

if(OpenMP_C_FOUND)
    target_link_libraries(wowml_c PRIVATE OpenMP::OpenMP_C)
endif()

if(WOWML_USE_BLAS)
    if(WOWML_BLAS_INCS)
        target_include_directories(wowml_c PRIVATE ${WOWML_BLAS_INCS})
    endif()
    if(WOWML_BLAS_LIBS)
        target_link_libraries(wowml_c PRIVATE ${WOWML_BLAS_LIBS})
    endif()
endif()

set_target_properties(wowml_c PROPERTIES PREFIX "" OUTPUT_NAME "wowml_c")

# ---- C Unit Test Harnesses ----
option(BUILD_TESTS "Build C unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    set(C_TEST_NAMES test_tensor test_ops test_matmul test_linalg)
    foreach(test_name ${C_TEST_NAMES})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/tests/${test_name}.c")
            add_executable(${test_name} csrc/tests/${test_name}.c)
            target_link_libraries(${test_name} PRIVATE wowml_c)
            if(UNIX AND NOT APPLE)
                target_link_libraries(${test_name} PRIVATE m)
            endif()
            target_include_directories(${test_name} PRIVATE csrc/include)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endif()
    endforeach()
endif()

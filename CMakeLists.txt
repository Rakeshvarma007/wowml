cmake_minimum_required(VERSION 3.20)
project(wowml_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ---- Compiler Flags ----
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W3)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -Wall -Wextra)
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
        else()
            message(WARNING "Accelerate BLAS/LAPACK not found. Falling back to pure C11 reference implementations.")
            set(WOWML_USE_BLAS OFF CACHE BOOL "Use OpenBLAS and LAPACKE fast paths" FORCE)
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
            find_path(CBLAS_INCLUDE_DIR
                NAMES cblas.h
                PATHS
                    /usr/include
                    /usr/include/x86_64-linux-gnu
                    /usr/include/openblas
                    /usr/local/include
                    "C:/openblas/include"
                    "$ENV{OPENBLAS_HOME}/include"
            )
            find_path(LAPACKE_INCLUDE_DIR
                NAMES lapacke.h
                PATHS
                    /usr/include
                    /usr/include/x86_64-linux-gnu
                    /usr/include/lapacke
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
                if(LAPACKE_INCLUDE_DIR)
                    list(APPEND WOWML_BLAS_INCS ${LAPACKE_INCLUDE_DIR})
                endif()
            else()
                message(WARNING "OpenBLAS/CBLAS not found. Falling back to pure C11 reference implementations.")
                set(WOWML_USE_BLAS OFF CACHE BOOL "Use OpenBLAS and LAPACKE fast paths" FORCE)
            endif()
        endif()
    endif()
endif()

list(REMOVE_DUPLICATES WOWML_BLAS_INCS)

# ---- Core C Shared Library ----
file(GLOB WOWML_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/src/*.c")
if(NOT WOWML_SOURCES)
    message(FATAL_ERROR "No C source files found in ${CMAKE_CURRENT_SOURCE_DIR}/csrc/src")
endif()

add_library(wowml_c SHARED ${WOWML_SOURCES})
target_include_directories(wowml_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include")

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

# Keep wowml_c.dll alongside test executables on Windows
if(WIN32)
    set_target_properties(wowml_c PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# ---- C Unit Test Harnesses ----
option(BUILD_TESTS "Build C unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    set(C_TEST_NAMES test_tensor test_ops test_matmul test_linalg)
    foreach(test_name ${C_TEST_NAMES})
        set(test_src "${CMAKE_CURRENT_SOURCE_DIR}/csrc/tests/${test_name}.c")
        if(EXISTS "${test_src}")
            add_executable(${test_name} "${test_src}")
            target_link_libraries(${test_name} PRIVATE wowml_c)
            if(UNIX AND NOT APPLE)
                target_link_libraries(${test_name} PRIVATE m)
            endif()
            if(WIN32)
                set_target_properties(${test_name} PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                )
            endif()
            add_test(NAME ${test_name} COMMAND ${test_name})
        endif()
    endforeach()
endif()
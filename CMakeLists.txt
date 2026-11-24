cmake_minimum_required(VERSION 3.20)
project(wowml_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- C Kernel shared library ----
file(GLOB WOWML_SOURCES csrc/src/*.c)
add_library(wowml_c SHARED ${WOWML_SOURCES})
target_include_directories(wowml_c PUBLIC csrc/include)
set_target_properties(wowml_c PROPERTIES PREFIX "" OUTPUT_NAME "wowml_c")

# ---- C Tests (optional) ----
option(BUILD_TESTS "Build C unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SOURCES csrc/tests/*.c)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} wowml_c)
        target_include_directories(${test_name} PRIVATE csrc/include)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

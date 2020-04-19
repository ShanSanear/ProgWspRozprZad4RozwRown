cmake_minimum_required(VERSION 3.16)
project(ProgWspRozprZad4RozwRown)

set(CMAKE_CXX_STANDARD 20)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -fopenmp ${GCC_COVERAGE_COMPILE_FLAGS}" )
add_library(cxxopts INTERFACE)
add_library(cxxopts::cxxopts ALIAS cxxopts)
set(CMAKE_EXE_LINKER_FLAGS " -static")
target_link_libraries(cxxopts INTERFACE -static-libgcc -static-libstdc++)
target_include_directories(cxxopts INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

add_executable(ProgWspRozprZad4RozwRown src/main.cpp)
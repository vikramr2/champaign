cmake_minimum_required(VERSION 3.12)
project(champaign_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../lib)

# Create Python module
pybind11_add_module(champaign
    ${CMAKE_SOURCE_DIR}/../lib/python_bindings.cpp
)

# Link OpenMP
target_link_libraries(champaign PRIVATE OpenMP::OpenMP_CXX)

# Set output directory (use CMAKE_LIBRARY_OUTPUT_DIRECTORY from setup.py if provided)
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(champaign PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

# Installation
install(TARGETS champaign
    LIBRARY DESTINATION .
)

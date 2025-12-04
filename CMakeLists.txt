cmake_minimum_required(VERSION 3.12)
project(champaign_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# Find OpenMP package (required for parallel graph processing)
find_package(OpenMP REQUIRED)

# Add the library directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Set compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Option to build Python module
option(BUILD_PYTHON_MODULE "Build Python module" OFF)

# ============================================================================
# Target 1: C++ Executable
# ============================================================================
add_executable(champaign_exe app/champaign.cpp)
target_link_libraries(champaign_exe PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(champaign_exe PROPERTIES OUTPUT_NAME champaign)

# ============================================================================
# Target 2: Python Module (only if BUILD_PYTHON_MODULE is ON)
# ============================================================================
if(BUILD_PYTHON_MODULE)
    # Find Python and pybind11
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    # Create Python module
    pybind11_add_module(champaign_py
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/python_bindings.cpp
    )

    # Link OpenMP
    target_link_libraries(champaign_py PRIVATE OpenMP::OpenMP_CXX)

    # Set module name to 'champaign' for Python import
    set_target_properties(champaign_py PROPERTIES OUTPUT_NAME champaign)

    # Set output directory (use CMAKE_LIBRARY_OUTPUT_DIRECTORY from setup.py if provided)
    if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(champaign_py PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        )
    endif()

    # Installation
    install(TARGETS champaign_py
        LIBRARY DESTINATION .
    )
endif()

cmake_minimum_required(VERSION 3.16)
project(HighFidelityAIModelSimulation VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Include directories
include_directories(include)

# Source files (exclude main.cpp for library)
file(GLOB SOURCES "src/hardware_simulator.cpp" "src/ai_model.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

# Create library first
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# Main executable
add_executable(ai_simulation src/main.cpp)
target_link_libraries(ai_simulation ${PROJECT_NAME})

# Optional: Enable testing
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# Optional: Add examples  
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# Install targets
install(TARGETS ai_simulation DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
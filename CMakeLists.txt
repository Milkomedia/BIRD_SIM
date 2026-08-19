cmake_minimum_required(VERSION 3.16)
project(bird_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(mujoco CONFIG REQUIRED PATHS "$ENV{HOME}/.local/mujoco")
find_package(glfw3 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(
  run
  main.cpp
  src/Servo.cpp
  src/ELRS.cpp
  src/MST.cpp
  src/mmap_manager.cpp
)

target_include_directories(
  run
  PRIVATE
    include
    src
)

target_link_libraries(
  run
  PRIVATE
    mujoco::mujoco
    glfw
    Eigen3::Eigen
    Threads::Threads
)

target_compile_definitions(
  run
  PRIVATE BIRD_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(CheckIPOSupported)
check_ipo_supported(RESULT BIRD_IPO_SUPPORTED OUTPUT BIRD_IPO_ERROR)
if(BIRD_IPO_SUPPORTED)
  set_property(TARGET run PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
